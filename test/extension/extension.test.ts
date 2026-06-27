import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import autoragExtension from "../../src/extension.ts";

let cwd: string;

beforeEach(() => {
	cwd = mkdtempSync(join(tmpdir(), "autorag-ext-"));
});

afterEach(() => {
	rmSync(cwd, { recursive: true, force: true });
});

interface Handlers {
	[event: string]: (event: unknown, ctx: unknown) => unknown;
}

function makePi(): {
	pi: ExtensionAPI;
	registerTool: ReturnType<typeof vi.fn>;
	setActiveTools: ReturnType<typeof vi.fn>;
	commands: Record<string, (args: string, ctx: unknown) => Promise<void>>;
	handlers: Handlers;
	appendEntry: ReturnType<typeof vi.fn>;
} {
	const registerTool = vi.fn();
	const setActiveTools = vi.fn();
	const handlers: Handlers = {};
	const commands: Record<string, (args: string, ctx: unknown) => Promise<void>> = {};
	const on = vi.fn((eventName: string, handler: Handlers[string]) => {
		handlers[eventName] = handler;
	});
	const registerCommand = vi.fn(
		(name: string, options: { handler: (args: string, ctx: unknown) => Promise<void> }) => {
			commands[name] = options.handler;
		},
	);
	const pi = {
		registerTool,
		on,
		setActiveTools,
		registerCommand,
		appendEntry: vi.fn(),
	} as Partial<ExtensionAPI> as ExtensionAPI;
	return {
		pi,
		registerTool,
		setActiveTools,
		commands,
		handlers,
		appendEntry: pi.appendEntry as ReturnType<typeof vi.fn>,
	};
}

describe("autoragExtension", () => {
	it("is a function", () => {
		expect(typeof autoragExtension).toBe("function");
	});

	it("keeps Pi builtin tools and registers only check_memory", () => {
		const { pi, registerTool, handlers } = makePi();
		autoragExtension(pi);

		const registeredNames = registerTool.mock.calls.map((call) => (call[0] as { name: string }).name);
		expect(registeredNames).toEqual(["check_memory"]);

		const events = Object.keys(handlers);
		expect(events).toContain("session_start");
		expect(events).toContain("tool_result");
		expect(events).toContain("before_agent_start");
		expect(events).toContain("message_end");
	});

	it("sets the active tool surface to Pi builtins plus bash and check_memory", async () => {
		const { pi, setActiveTools, handlers } = makePi();
		autoragExtension(pi);

		await handlers.before_agent_start({ systemPrompt: "base" }, { cwd });

		expect(setActiveTools).toHaveBeenCalledTimes(1);
		const requested = setActiveTools.mock.calls[0][0] as string[];
		expect([...requested].sort()).toEqual(["bash", "check_memory", "find", "grep", "ls", "read"].sort());
		expect(requested).toContain("bash");
		for (const banned of ["edit", "write"]) {
			expect(requested).not.toContain(banned);
		}
	});

	it("registers an autorag-refresh command that parses real source directories", async () => {
		const { pi, commands, handlers, appendEntry } = makePi();
		autoragExtension(pi);
		expect(typeof commands["autorag-refresh"]).toBe("function");

		const docs = join(cwd, "docs");
		mkdirSync(docs, { recursive: true });
		const file = join(docs, "x.txt");
		writeFileSync(file, "AAAA\n");
		utimesSync(file, 1_700_000_000, 1_700_000_000);
		mkdirSync(join(cwd, ".autorag"), { recursive: true });
		writeFileSync(join(cwd, ".autorag", "sources.json"), JSON.stringify([docs]));

		await handlers.session_start({}, { cwd });

		const before = statSync(file, { bigint: true });
		writeFileSync(file, "BBBB\n");
		expect(statSync(file, { bigint: true }).size).toBe(before.size);
		utimesSync(file, 1_700_000_000, 1_700_000_000);

		await commands["autorag-refresh"]("", {});

		const refreshEntry = appendEntry.mock.calls.find((c: unknown[]) => c[0] === "autorag_refresh");
		expect(refreshEntry).toBeDefined();
		expect((refreshEntry?.[1] as { parsed: { written: number } }).parsed.written).toBe(1);
	});

	it("registers autorag-jikji-refresh when explicit config enables Jikji", async () => {
		const { pi, commands, handlers, appendEntry } = makePi();
		autoragExtension(pi);
		const docs = join(cwd, "docs");
		const binaryPath = join(cwd, "fake-jikji.mjs");
		const logPath = join(cwd, "jikji-prepare.jsonl");
		mkdirSync(docs, { recursive: true });
		mkdirSync(join(cwd, ".autorag"), { recursive: true });
		writeFileSync(join(cwd, ".autorag", "sources.json"), JSON.stringify([docs]));
		writeFileSync(
			binaryPath,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, envMedia: process.env.JIKJI_ENABLE_MEDIA_INDEX ?? null }) + "\\n");
console.log(JSON.stringify({ prepared: true }));
`,
		);
		chmodSync(binaryPath, 0o755);
		writeFileSync(
			join(cwd, ".autorag", "jikji.json"),
			JSON.stringify({
				enabled: true,
				binaryPath,
				includeHidden: true,
				includeSensitive: true,
				parseTimeout: 5,
				maxFiles: 10,
				staleAfterSeconds: 60,
				exclude: ["private/**"],
			}),
		);

		await handlers.session_start({}, { cwd });
		await commands["autorag-jikji-refresh"]("", {});

		const logged = JSON.parse(readFileSync(logPath, "utf8").trim()) as { args: string[]; envMedia: string | null };
		expect(logged.args).toEqual([
			"prepare",
			docs,
			"--json",
			"--include-hidden",
			"--include-sensitive",
			"--parse-timeout",
			"5",
			"--max-files",
			"10",
			"--stale-after-seconds",
			"60",
			"--exclude",
			"private/**",
		]);
		expect(logged.envMedia).toBeNull();
		const refreshEntry = appendEntry.mock.calls.find((c: unknown[]) => c[0] === "autorag_jikji_refresh");
		expect(refreshEntry).toBeDefined();
		expect(JSON.stringify(refreshEntry?.[1])).toContain("success");
		expect(JSON.stringify(logged.args)).not.toContain("enable-media");
	});

	it("ignores invalid Jikji config without changing active tools", async () => {
		const { pi, commands, handlers, setActiveTools, appendEntry } = makePi();
		autoragExtension(pi);
		mkdirSync(join(cwd, ".autorag"), { recursive: true });
		writeFileSync(join(cwd, ".autorag", "jikji.json"), "[]");

		await handlers.session_start({}, { cwd });
		await commands["autorag-jikji-refresh"]("", {});
		await handlers.before_agent_start({ systemPrompt: "base" }, { cwd });

		expect(appendEntry.mock.calls.find((c: unknown[]) => c[0] === "autorag_jikji_refresh")).toBeUndefined();
		expect(setActiveTools.mock.calls[0][0]).toEqual(["grep", "find", "read", "ls", "check_memory", "bash"]);
	});
});
