/**
 * Index schedule policy: decides whether a *background* refresh (auto-refresh
 * tick, fs-watch refresh, `autorag watch`) may run right now, from the power
 * source and the local day/time. Explicit `autorag refresh` is never gated.
 *
 * Invariants:
 *  - Pure decision: {@link evaluateIndexSchedule} takes the clock and power
 *    source as arguments, so it is deterministic.
 *  - Fail open on `power: "ac"` when the power source cannot be read (a
 *    desktop or a failed probe must not silently stop indexing).
 *  - A window is `[start, end)` in local time; `start > end` wraps midnight and
 *    belongs to the day it starts on.
 */
import { execFile } from "node:child_process";
import { readdir, readFile } from "node:fs/promises";
import { promisify } from "node:util";

export type PowerPolicy = "any" | "ac";
export type PowerSource = "ac" | "battery" | "unknown";
export type ScheduleDay = "sun" | "mon" | "tue" | "wed" | "thu" | "fri" | "sat";

export interface IndexSchedule {
	/** `any` (default) ignores power; `ac` pauses while on battery and runs only when plugged in. */
	readonly power?: PowerPolicy;
	/** Local-time windows `HH:MM-HH:MM`; indexing is allowed inside any of them. Omitted = all day. */
	readonly windows?: readonly string[];
	/** Local weekdays indexing is allowed on. Omitted = every day. */
	readonly days?: readonly ScheduleDay[];
}

export type ScheduleDenyReason = "battery" | "outside-window" | "outside-day";

export type ScheduleDecision =
	| { readonly allowed: true }
	| { readonly allowed: false; readonly reason: ScheduleDenyReason };

const DAYS: readonly ScheduleDay[] = ["sun", "mon", "tue", "wed", "thu", "fri", "sat"];
const POWER_POLICIES: readonly PowerPolicy[] = ["any", "ac"];
const SCHEDULE_KEYS: Record<string, true> = { power: true, windows: true, days: true };
const WINDOW_PATTERN = /^([01]\d|2[0-3]):([0-5]\d)-([01]\d|2[0-3]):([0-5]\d)$/;

function parseWindow(window: string): { start: number; end: number } {
	const match = WINDOW_PATTERN.exec(window);
	if (match === null) throw new Error(`refresh.schedule.windows entry "${window}" must look like HH:MM-HH:MM`);
	const start = Number(match[1]) * 60 + Number(match[2]);
	const end = Number(match[3]) * 60 + Number(match[4]);
	if (start === end) throw new Error(`refresh.schedule.windows entry "${window}" has equal start and end`);
	return { start, end };
}

/** Validate and normalize a raw `refresh.schedule` value. Throws a plain Error naming the field. */
export function validateIndexSchedule(raw: unknown): IndexSchedule {
	if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
		throw new Error("refresh.schedule must be an object");
	}
	const record = raw as Record<string, unknown>;
	for (const key of Object.keys(record)) {
		if (!Object.hasOwn(SCHEDULE_KEYS, key)) throw new Error(`refresh.schedule.${key} is not a recognized field`);
	}
	const power = record.power;
	if (power !== undefined && !POWER_POLICIES.includes(power as PowerPolicy)) {
		throw new Error(`refresh.schedule.power must be one of ${POWER_POLICIES.join(", ")}`);
	}
	let windows: string[] | undefined;
	if (record.windows !== undefined) {
		if (!Array.isArray(record.windows) || record.windows.some((w) => typeof w !== "string")) {
			throw new Error("refresh.schedule.windows must be an array of HH:MM-HH:MM strings");
		}
		windows = (record.windows as string[]).map((w) => w.trim());
		for (const w of windows) parseWindow(w);
	}
	let days: ScheduleDay[] | undefined;
	if (record.days !== undefined) {
		if (!Array.isArray(record.days) || record.days.length === 0) {
			throw new Error(`refresh.schedule.days must be a non-empty array of ${DAYS.join(", ")}`);
		}
		days = (record.days as unknown[]).map((d) =>
			typeof d === "string" ? d.trim().toLowerCase() : d,
		) as ScheduleDay[];
		for (const d of days) {
			if (!DAYS.includes(d))
				throw new Error(`refresh.schedule.days entry ${JSON.stringify(d)} is not one of ${DAYS.join(", ")}`);
		}
	}
	return {
		...(power !== undefined ? { power: power as PowerPolicy } : {}),
		...(windows !== undefined ? { windows } : {}),
		...(days !== undefined ? { days } : {}),
	};
}

/** Decide whether a background refresh may run at `now` given the detected `source`. */
export function evaluateIndexSchedule(
	schedule: IndexSchedule | undefined,
	now: Date,
	source: PowerSource,
): ScheduleDecision {
	if (schedule === undefined) return { allowed: true };
	if (schedule.power === "ac" && source === "battery") return { allowed: false, reason: "battery" };

	const today = DAYS[now.getDay()] as ScheduleDay;
	const yesterday = DAYS[(now.getDay() + 6) % 7] as ScheduleDay;
	const dayAllowed = (day: ScheduleDay): boolean => schedule.days === undefined || schedule.days.includes(day);

	if (schedule.windows === undefined || schedule.windows.length === 0) {
		return dayAllowed(today) ? { allowed: true } : { allowed: false, reason: "outside-day" };
	}
	const minutes = now.getHours() * 60 + now.getMinutes();
	for (const window of schedule.windows) {
		const { start, end } = parseWindow(window);
		if (start < end) {
			if (minutes >= start && minutes < end && dayAllowed(today)) return { allowed: true };
		} else if ((minutes >= start && dayAllowed(today)) || (minutes < end && dayAllowed(yesterday))) {
			return { allowed: true };
		}
	}
	return { allowed: false, reason: "outside-window" };
}

/** Map raw probe output to a power source. Returns `unknown` for anything unrecognized. */
export function parsePowerSourceOutput(platform: string, output: string): PowerSource {
	if (platform === "darwin") {
		if (/Now drawing from 'AC Power'/i.test(output)) return "ac";
		if (/Now drawing from 'Battery Power'/i.test(output)) return "battery";
		return "unknown";
	}
	if (platform === "win32") {
		const status = Number.parseInt(output.trim(), 10);
		if (!Number.isFinite(status)) return "unknown";
		return status === 1 ? "battery" : "ac";
	}
	return "unknown";
}

const execFileAsync = promisify(execFile);

export interface PowerProbeDeps {
	readonly platform?: NodeJS.Platform;
	/** Run a command and return stdout. */
	readonly run?: (command: string, args: readonly string[]) => Promise<string>;
	readonly read?: (path: string) => Promise<string>;
	readonly list?: (dir: string) => Promise<string[]>;
}

const POWER_SUPPLY_DIR = "/sys/class/power_supply";

async function detectLinuxPowerSource(
	read: (path: string) => Promise<string>,
	list: (dir: string) => Promise<string[]>,
): Promise<PowerSource> {
	const entries = await list(POWER_SUPPLY_DIR);
	let hasBattery = false;
	let hasMains = false;
	let mainsOnline = false;
	let batteryDischarging = false;
	for (const entry of entries) {
		const base = `${POWER_SUPPLY_DIR}/${entry}`;
		const type = (await read(`${base}/type`).catch(() => "")).trim();
		if (type === "Battery") {
			hasBattery = true;
			const status = (await read(`${base}/status`).catch(() => "")).trim();
			if (status === "Discharging") batteryDischarging = true;
		} else if (type === "Mains" || type === "USB") {
			hasMains = true;
			if ((await read(`${base}/online`).catch(() => "")).trim() === "1") mainsOnline = true;
		}
	}
	if (!hasBattery) return "unknown";
	if (mainsOnline) return "ac";
	return hasMains || batteryDischarging ? "battery" : "unknown";
}

/** Detect the current power source. Never throws: any probe failure yields `unknown`. */
export async function detectPowerSource(deps: PowerProbeDeps = {}): Promise<PowerSource> {
	const platform = deps.platform ?? process.platform;
	try {
		if (platform === "linux") {
			return await detectLinuxPowerSource(
				deps.read ?? ((path) => readFile(path, "utf8")),
				deps.list ?? ((dir) => readdir(dir)),
			);
		}
		const run =
			deps.run ??
			(async (command, args) =>
				(await execFileAsync(command, [...args], { timeout: 5000, windowsHide: true })).stdout);
		if (platform === "darwin") return parsePowerSourceOutput(platform, await run("pmset", ["-g", "batt"]));
		if (platform === "win32") {
			return parsePowerSourceOutput(
				platform,
				await run("powershell.exe", [
					"-NoProfile",
					"-NonInteractive",
					"-Command",
					"(Get-CimInstance Win32_Battery | Select-Object -First 1).BatteryStatus",
				]),
			);
		}
		return "unknown";
	} catch {
		return "unknown";
	}
}
