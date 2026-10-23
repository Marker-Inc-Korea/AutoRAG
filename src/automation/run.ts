import { spawn } from "node:child_process";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { resolveAutoRAGHome, resolveUserHome } from "../config/home.ts";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { terminateProcessTree } from "../process/terminate-tree.ts";
import type { Automation, AutomationEvent } from "./config.ts";

export const AUTOMATION_MAX_APPEND_FILES = 100;
const MAX_CAPTURE_BYTES = 16 * 1024;
const MAX_LOG_FILE_BYTES = 1024 * 1024;

export interface AutomationRunRecord {
	readonly id: string;
	readonly startedAt: string;
	readonly durationMs: number;
	readonly eventCount: number;
	readonly filesOverflow: boolean;
	readonly exitCode: number | null;
	readonly signal: NodeJS.Signals | null;
	readonly timedOut: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly error?: string;
}

export interface RunAutomationOptions {
	readonly home?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** Override `[nodePath, cliEntry]` used when `run[0] === "autorag"`. */
	readonly selfCommand?: readonly string[];
}

export function automationLogPath(home: string = resolveAutoRAGHome(), id: string): string {
	return join(home, "automations", "logs", `${id}.jsonl`);
}

export function appendAutomationRunLog(home: string, record: AutomationRunRecord): void {
	const logPath = automationLogPath(home, record.id);
	mkdirSync(dirname(logPath), { recursive: true });
	try {
		if (existsSync(logPath) && statSync(logPath).size >= MAX_LOG_FILE_BYTES) {
			renameSync(logPath, `${logPath}.1`);
		}
	} catch {
		// Best-effort rotation; still append the run record below.
	}
	appendFileSync(logPath, `${JSON.stringify(record)}\n`, "utf8");
}

export function readLastAutomationRun(home: string, id: string): AutomationRunRecord | undefined {
	const logPath = automationLogPath(home, id);
	if (!existsSync(logPath)) return undefined;
	const lines = readFileSync(logPath, "utf8").trim().split("\n");
	const last = lines.at(-1);
	if (!last) return undefined;
	try {
		return JSON.parse(last) as AutomationRunRecord;
	} catch {
		return undefined;
	}
}

/**
 * `[runtime, cliEntry]` for this AutoRAG install, found by walking up from this
 * module to the package root. `process.argv[1]` is not used: inside the MCP
 * server or a library host it is not the AutoRAG CLI. Bundled installs run
 * `dist/cli/index.js`; a source checkout without a build runs the TypeScript entry.
 */
export function autoragCliCommand(): readonly string[] {
	let dir = dirname(fileURLToPath(import.meta.url));
	for (let depth = 0; depth < 6; depth += 1) {
		for (const entry of [join(dir, "dist", "cli", "index.js"), join(dir, "src", "cli", "index.ts")]) {
			if (existsSync(join(dir, "package.json")) && existsSync(entry)) return [process.execPath, entry];
		}
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return ["autorag"];
}

function resolveSpawnArgv(
	run: readonly string[],
	events: readonly AutomationEvent[],
	appendFiles: boolean,
	selfCommand: readonly string[],
): { readonly command: string; readonly args: readonly string[]; readonly filesOverflow: boolean } {
	let head: readonly string[];
	if (run[0] === "autorag") {
		head = [...selfCommand, ...run.slice(1)];
	} else if (run[0] === "node") {
		head = [process.execPath, ...run.slice(1)];
	} else {
		head = run;
	}

	const filesOverflow = appendFiles && events.length > AUTOMATION_MAX_APPEND_FILES;
	const fileArgs = appendFiles ? events.slice(0, AUTOMATION_MAX_APPEND_FILES).map((ev) => ev.path) : [];
	const [rawCmd, ...rawArgs] = [...head, ...fileArgs];
	const portable = portableSpawnCommand(rawCmd, rawArgs);
	return { command: portable.command, args: portable.args, filesOverflow };
}

function appendBounded(current: string, chunk: Buffer): string {
	const next = current + chunk.toString("utf8");
	return next.length <= MAX_CAPTURE_BYTES ? next : next.slice(next.length - MAX_CAPTURE_BYTES);
}

export async function runAutomation(
	automation: Automation,
	events: readonly AutomationEvent[],
	options: RunAutomationOptions = {},
): Promise<AutomationRunRecord> {
	const env = options.env ?? process.env;
	const home = options.home ?? resolveAutoRAGHome(env);
	const selfCommand = options.selfCommand ?? autoragCliCommand();
	const { command, args, filesOverflow } = resolveSpawnArgv(
		automation.run,
		events,
		automation.appendFiles,
		selfCommand,
	);
	const cwd = existsSync(automation.paths[0]) ? automation.paths[0] : resolveUserHome(env);
	const startedAtMs = Date.now();
	const startedAt = new Date(startedAtMs).toISOString();

	const childEnv: NodeJS.ProcessEnv = {
		...env,
		AUTORAG_AUTOMATION_ID: automation.id,
		AUTORAG_EVENT_COUNT: String(events.length),
		...(filesOverflow ? { AUTORAG_FILES_OVERFLOW: "true" } : {}),
	};

	let resolve!: (record: AutomationRunRecord) => void;
	const promise = new Promise<AutomationRunRecord>((r) => {
		resolve = r;
	});
	let stdout = "";
	let stderr = "";
	let timedOut = false;
	let settled = false;

	const finish = (exitCode: number | null, signal: NodeJS.Signals | null, error?: string): void => {
		if (settled) return;
		settled = true;
		clearTimeout(timer);
		const record: AutomationRunRecord = {
			id: automation.id,
			startedAt,
			durationMs: Math.max(0, Date.now() - startedAtMs),
			eventCount: events.length,
			filesOverflow,
			exitCode: timedOut ? null : exitCode,
			signal,
			timedOut,
			stdout,
			stderr,
			...(error ? { error } : {}),
		};
		appendAutomationRunLog(home, record);
		resolve(record);
	};

	const child = spawn(command, args, {
		cwd,
		env: childEnv,
		stdio: ["pipe", "pipe", "pipe"],
		detached: process.platform !== "win32",
		windowsHide: true,
	});

	const timer = setTimeout(() => {
		timedOut = true;
		terminateProcessTree(child, "SIGKILL");
	}, automation.timeoutMs);
	timer.unref?.();

	child.stdout.on("data", (chunk: Buffer) => {
		stdout = appendBounded(stdout, chunk);
	});
	child.stderr.on("data", (chunk: Buffer) => {
		stderr = appendBounded(stderr, chunk);
	});
	child.stdin.on("error", () => {
		// Child exited without reading stdin (EPIPE / ERR_STREAM_DESTROYED).
	});
	child.stdin.end(`${JSON.stringify(events)}\n`);

	child.on("error", (err) => {
		finish(null, null, err.message);
	});
	child.on("close", (code, sig) => {
		finish(code, sig);
	});

	return promise;
}
