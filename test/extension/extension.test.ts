import { mkdirSync, mkdtempSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ACTIVE_TOOLS, AGENTDIR_TOOL_NAMES } from "../../src/agentdir/tools.ts";
import { clearWorkspaceCache } from "../../src/agentdir/workspace.ts";
import autoragExtension from "../../src/extension.ts";

let cwd: string;

beforeEach(() => {
	cwd = mkdtempSync(join(tmpdir(), "autorag-ext-"));
	clearWorkspaceCache();
});

afterEach(() => {
	clearWorkspaceCache();
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

	it("replaces builtin grep/find/read/ls with agentdir tools and registers check_memory", () => {
		const { pi, registerTool, handlers } = makePi();
		autoragExtension(pi);

		const registeredNames = registerTool.mock.calls.map((call) => (call[0] as { name: string }).name);
		// check_memory + organize + all agentdir tools (same names override builtins grep/find/read/ls)
		expect(registeredNames).toContain("check_memory");
		expect(registeredNames).toContain("organize");
		for (const name of AGENTDIR_TOOL_NAMES) {
			expect(registeredNames).toContain(name);
		}
		expect(registerTool).toHaveBeenCalledTimes(AGENTDIR_TOOL_NAMES.length + 2);

		const events = Object.keys(handlers);
		expect(events).toContain("session_start");
		expect(events).toContain("tool_result");
		expect(events).toContain("before_agent_start");
		expect(events).toContain("message_end");
	});

	it("closes the active tool surface to ACTIVE_TOOLS, excluding builtins (AC-5)", async () => {
		const { pi, setActiveTools, handlers } = makePi();
		autoragExtension(pi);

		await handlers.before_agent_start({ systemPrompt: "base" }, { cwd });

		expect(setActiveTools).toHaveBeenCalledTimes(1);
		const requested = setActiveTools.mock.calls[0][0] as string[];
		expect([...requested].sort()).toEqual([...ACTIVE_TOOLS].sort());
		for (const banned of ["bash", "edit", "write"]) {
			expect(requested).not.toContain(banned);
		}
	});

	it("registers an autorag-refresh command that hash-verifies (issue #2 / AC-7)", async () => {
		const { pi, commands, handlers, appendEntry } = makePi();
		autoragExtension(pi);
		expect(typeof commands["autorag-refresh"]).toBe("function");

		// project cwd with a source mapped via .autorag/sources.json
		const docs = join(cwd, "docs");
		mkdirSync(docs, { recursive: true });
		const file = join(docs, "x.txt");
		writeFileSync(file, "AAAA\n");
		utimesSync(file, 1_700_000_000, 1_700_000_000);
		mkdirSync(join(cwd, ".autorag"), { recursive: true });
		writeFileSync(join(cwd, ".autorag", "sources.json"), JSON.stringify([docs]));

		await handlers.session_start({}, { cwd });

		// same-size + same-mtime content swap
		const before = statSync(file, { bigint: true });
		writeFileSync(file, "BBBB\n");
		expect(statSync(file, { bigint: true }).size).toBe(before.size);
		utimesSync(file, 1_700_000_000, 1_700_000_000);

		await commands["autorag-refresh"]("", {});

		const refreshEntry = appendEntry.mock.calls.find((c: unknown[]) => c[0] === "autorag_refresh");
		expect(refreshEntry).toBeDefined();
		expect((refreshEntry?.[1] as { summary: { refreshed: number } }).summary.refreshed).toBeGreaterThanOrEqual(1);
	});
});
