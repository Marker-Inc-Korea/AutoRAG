import { mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Automation, AutomationEvent } from "../../src/automation/config.ts";
import {
	AUTOMATION_MAX_APPEND_FILES,
	automationLogPath,
	readLastAutomationRun,
	runAutomation,
} from "../../src/automation/run.ts";

let root: string;
let home: string;
let watched: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-run-"));
	home = join(root, "home");
	watched = join(root, "watched");
	mkdirSync(home, { recursive: true });
	mkdirSync(watched, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function rule(overrides: Partial<Automation>): Automation {
	return {
		id: "test",
		paths: [watched],
		events: ["create", "update", "delete"],
		include: ["**"],
		exclude: [],
		run: ["node"],
		appendFiles: false,
		debounceMs: 1000,
		timeoutMs: 10_000,
		...overrides,
	};
}

function writeScript(name: string, body: string): string {
	const path = join(root, name);
	writeFileSync(path, body, "utf8");
	return path;
}

describe("runAutomation", () => {
	it("passes events on stdin, sets env, runs in paths[0], and appends a JSONL log record", async () => {
		const script = writeScript(
			"echo.ts",
			`
const raw: string = await new Response(process.stdin).text();
const events: Array<{ type: string; path: string }> = JSON.parse(raw);
console.log(JSON.stringify({
	id: process.env.AUTORAG_AUTOMATION_ID,
	count: process.env.AUTORAG_EVENT_COUNT,
	overflow: process.env.AUTORAG_FILES_OVERFLOW ?? null,
	cwd: process.cwd(),
	argv: process.argv.slice(2),
	events,
}));
`,
		);
		const events: AutomationEvent[] = [{ type: "create", path: join(watched, "a.pdf") }];
		const result = await runAutomation(rule({ run: ["node", script, "arg1"] }), events, { home });

		expect(result.exitCode).toBe(0);
		expect(result.timedOut).toBe(false);
		const payload = JSON.parse(result.stdout.trim());
		expect(payload).toEqual({
			id: "test",
			count: "1",
			overflow: null,
			cwd: realpathSync(watched),
			argv: ["arg1"],
			events,
		});
		expect(readLastAutomationRun(home, "test")?.exitCode).toBe(0);
		expect(readFileSync(automationLogPath(home, "test"), "utf8").trim().split("\n")).toHaveLength(1);
	});

	it("caps appendFiles at AUTOMATION_MAX_APPEND_FILES and sets AUTORAG_FILES_OVERFLOW while keeping all events on stdin", async () => {
		const script = writeScript(
			"cap.mjs",
			`
let s = "";
for await (const c of process.stdin) s += c;
console.log(JSON.stringify({
	argvLen: process.argv.slice(2).length,
	stdinLen: JSON.parse(s).length,
	overflow: process.env.AUTORAG_FILES_OVERFLOW,
}));
`,
		);
		const events: AutomationEvent[] = Array.from({ length: AUTOMATION_MAX_APPEND_FILES + 5 }, (_, i) => ({
			type: "create",
			path: join(watched, `f${i}.pdf`),
		}));
		const result = await runAutomation(rule({ run: ["node", script], appendFiles: true }), events, { home });

		expect(result.filesOverflow).toBe(true);
		expect(JSON.parse(result.stdout.trim())).toEqual({
			argvLen: AUTOMATION_MAX_APPEND_FILES,
			stdinLen: AUTOMATION_MAX_APPEND_FILES + 5,
			overflow: "true",
		});
	});

	it("maps run[0] === 'autorag' through selfCommand", async () => {
		const fakeCli = writeScript("cli.mjs", "console.log(JSON.stringify(process.argv.slice(2)));");
		const result = await runAutomation(
			rule({ run: ["autorag", "search", "summarize"], appendFiles: true }),
			[{ type: "create", path: "/x/a.pdf" }],
			{ home, selfCommand: [process.execPath, fakeCli] },
		);
		expect(result.exitCode).toBe(0);
		expect(JSON.parse(result.stdout.trim())).toEqual(["search", "summarize", "/x/a.pdf"]);
	});

	it("terminates the process tree on timeout including pipe-holding grandchildren", async () => {
		const pidFile = join(root, "grandchild.pid");
		const script = writeScript(
			"hang.mjs",
			`
import { spawn } from "node:child_process";
import { writeFileSync } from "node:fs";
const child = spawn(process.execPath, ["-e", "process.stdin.resume()"], { stdio: ["pipe", "inherit", "inherit"] });
writeFileSync(${JSON.stringify(pidFile)}, String(child.pid));
`,
		);
		const result = await runAutomation(rule({ run: ["node", script], timeoutMs: 250 }), [], { home });
		expect(result.timedOut).toBe(true);
		expect(result.exitCode).toBeNull();

		const grandchildPid = Number(readFileSync(pidFile, "utf8").trim());
		expect(() => process.kill(grandchildPid, 0)).toThrow();
	});

	it("records spawn failures (missing command) without throwing", async () => {
		const result = await runAutomation(rule({ run: [join(root, "does-not-exist-bin")] }), [], { home });
		expect(result.exitCode).toBeNull();
		expect(result.error).toMatch(/ENOENT/);
		expect(readLastAutomationRun(home, "test")?.error).toMatch(/ENOENT/);
	});

	it("does not fail with EPIPE when the command exits before reading stdin", async () => {
		const script = writeScript("fast.mjs", "process.exit(0);");
		const events: AutomationEvent[] = Array.from({ length: 500 }, (_, i) => ({
			type: "create",
			path: join(watched, `${"x".repeat(200)}-${i}.pdf`),
		}));
		const result = await runAutomation(rule({ run: ["node", script] }), events, { home });
		expect(result.exitCode).toBe(0);
	});
});
