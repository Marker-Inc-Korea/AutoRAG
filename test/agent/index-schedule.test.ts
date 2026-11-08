import { describe, expect, it } from "vitest";
import {
	detectPowerSource,
	evaluateIndexSchedule,
	parsePowerSourceOutput,
	validateIndexSchedule,
} from "../../src/agent/index-schedule.ts";

/** Local-time constructor: 2026-11-09 is a Monday. */
const at = (day: number, hh: number, mm = 0): Date => new Date(2026, 10, day, hh, mm, 0);
const MON = 9;
const TUE = 10;

describe("validateIndexSchedule", () => {
	it("accepts power, windows and days and trims them", () => {
		expect(validateIndexSchedule({ power: "ac", windows: [" 22:00-07:00 "], days: ["Mon", "tue"] })).toEqual({
			power: "ac",
			windows: ["22:00-07:00"],
			days: ["mon", "tue"],
		});
	});

	it.each([
		[{ power: "solar" }, /power/],
		[{ windows: ["25:00-26:00"] }, /windows/],
		[{ windows: ["09:00-09:00"] }, /windows/],
		[{ windows: ["9am-5pm"] }, /windows/],
		[{ windows: "22:00-07:00" }, /windows/],
		[{ days: ["funday"] }, /days/],
		[{ days: [] }, /days/],
		[{ nope: 1 }, /nope/],
	])("rejects %j", (raw, message) => {
		expect(() => validateIndexSchedule(raw)).toThrow(message);
	});
});

describe("evaluateIndexSchedule", () => {
	it("allows everything when no policy is set", () => {
		expect(evaluateIndexSchedule(undefined, at(MON, 3), "battery")).toEqual({ allowed: true });
		expect(evaluateIndexSchedule({}, at(MON, 3), "battery")).toEqual({ allowed: true });
	});

	it("power=ac pauses on battery, runs on AC, and fails open when the source is unknown", () => {
		const policy = { power: "ac" as const };
		expect(evaluateIndexSchedule(policy, at(MON, 12), "battery")).toMatchObject({
			allowed: false,
			reason: "battery",
		});
		expect(evaluateIndexSchedule(policy, at(MON, 12), "ac")).toEqual({ allowed: true });
		expect(evaluateIndexSchedule(policy, at(MON, 12), "unknown")).toEqual({ allowed: true });
	});

	it("power=battery is no longer a valid policy", () => {
		expect(() => validateIndexSchedule({ power: "battery" })).toThrow(/power/);
	});

	it("power=any ignores the power source", () => {
		expect(evaluateIndexSchedule({ power: "any" }, at(MON, 12), "battery")).toEqual({ allowed: true });
	});

	it("same-day window is [start, end)", () => {
		const policy = { windows: ["09:00-17:00"] };
		expect(evaluateIndexSchedule(policy, at(MON, 8, 59), "ac")).toMatchObject({
			allowed: false,
			reason: "outside-window",
		});
		expect(evaluateIndexSchedule(policy, at(MON, 9, 0), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(MON, 16, 59), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(MON, 17, 0), "ac").allowed).toBe(false);
	});

	it("overnight window wraps midnight", () => {
		const policy = { windows: ["22:00-07:00"] };
		expect(evaluateIndexSchedule(policy, at(MON, 23), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(TUE, 6, 59), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(TUE, 7, 0), "ac").allowed).toBe(false);
		expect(evaluateIndexSchedule(policy, at(MON, 21, 59), "ac").allowed).toBe(false);
	});

	it("any of several windows opens the gate", () => {
		const policy = { windows: ["01:00-02:00", "13:00-14:00"] };
		expect(evaluateIndexSchedule(policy, at(MON, 13, 30), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(MON, 3), "ac").allowed).toBe(false);
	});

	it("days restrict by local weekday; an overnight window belongs to the day it starts", () => {
		const policy = { windows: ["22:00-07:00"], days: ["mon" as const] };
		expect(evaluateIndexSchedule(policy, at(MON, 23), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(TUE, 3), "ac").allowed).toBe(true); // Monday's night tail
		expect(evaluateIndexSchedule(policy, at(TUE, 23), "ac").allowed).toBe(false);
		expect(evaluateIndexSchedule(policy, at(MON, 3), "ac").allowed).toBe(false); // Sunday's night tail
	});

	it("days without windows allow the whole day", () => {
		const policy = { days: ["tue" as const] };
		expect(evaluateIndexSchedule(policy, at(TUE, 15), "ac").allowed).toBe(true);
		expect(evaluateIndexSchedule(policy, at(MON, 15), "ac")).toMatchObject({ allowed: false, reason: "outside-day" });
	});

	it("power and time must both pass", () => {
		const policy = { power: "ac" as const, windows: ["09:00-17:00"] };
		expect(evaluateIndexSchedule(policy, at(MON, 10), "battery").allowed).toBe(false);
		expect(evaluateIndexSchedule(policy, at(MON, 20), "ac").allowed).toBe(false);
		expect(evaluateIndexSchedule(policy, at(MON, 10), "ac").allowed).toBe(true);
	});
});

describe("parsePowerSourceOutput", () => {
	it("reads macOS pmset", () => {
		expect(
			parsePowerSourceOutput("darwin", "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged"),
		).toBe("ac");
		expect(parsePowerSourceOutput("darwin", "Now drawing from 'Battery Power'\n -InternalBattery-0")).toBe("battery");
		expect(parsePowerSourceOutput("darwin", "garbage")).toBe("unknown");
	});

	it("reads the Windows BatteryStatus (1 = discharging, empty = no battery)", () => {
		expect(parsePowerSourceOutput("win32", "1\r\n")).toBe("battery");
		expect(parsePowerSourceOutput("win32", "2\r\n")).toBe("ac");
		expect(parsePowerSourceOutput("win32", "")).toBe("unknown");
	});
});

describe("detectPowerSource", () => {
	it("returns unknown when the probe command fails", async () => {
		const result = await detectPowerSource({
			platform: "darwin",
			run: async () => {
				throw new Error("pmset missing");
			},
		});
		expect(result).toBe("unknown");
	});

	it("derives Linux state from power_supply entries", async () => {
		const files: Record<string, string> = {
			"/sys/class/power_supply/AC/type": "Mains\n",
			"/sys/class/power_supply/AC/online": "0\n",
			"/sys/class/power_supply/BAT0/type": "Battery\n",
		};
		const read = async (path: string): Promise<string> => {
			const hit = files[path];
			if (hit === undefined) throw new Error("ENOENT");
			return hit;
		};
		const list = async (): Promise<string[]> => ["AC", "BAT0"];
		expect(await detectPowerSource({ platform: "linux", read, list })).toBe("battery");
		files["/sys/class/power_supply/AC/online"] = "1\n";
		expect(await detectPowerSource({ platform: "linux", read, list })).toBe("ac");
		expect(await detectPowerSource({ platform: "linux", read, list: async () => [] })).toBe("unknown");
	});
});
