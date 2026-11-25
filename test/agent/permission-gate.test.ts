import { mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { BashOperations, ExtensionAPI, ExtensionContext, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	createPermissionGate,
	cyclePermissionMode,
	DEFAULT_PERMISSION_MODE,
	describeReadOnlyBlock,
	PERMISSION_MODES,
	PermissionController,
	type PermissionMode,
	parsePermissionMode,
	READONLY_PROMPT_ADDENDUM,
} from "../../src/agent/permission-gate.ts";
import type { ReadOnlySandbox } from "../../src/agent/readonly-sandbox.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

type Handler = (event: Record<string, unknown>, ctx: ExtensionContext) => unknown;

interface HarnessOptions {
	readonly mode?: PermissionMode;
	readonly hasUI?: boolean;
	readonly trustedPaths?: readonly string[];
	readonly answers?: (string | undefined)[];
	readonly sandbox?: ReadOnlySandbox;
}

interface Verdict {
	block?: boolean;
	reason?: string;
	systemPrompt?: string;
	operations?: BashOperations;
}

function fakeSandbox(scratchDir: string): {
	readonly sandbox: ReadOnlySandbox;
	readonly execCalls: string[];
	readonly disposed: () => boolean;
} {
	const execCalls: string[] = [];
	let isDisposed = false;
	const sandbox: ReadOnlySandbox = {
		backend: "seatbelt",
		scratchDir,
		operations: {
			exec: async (command) => {
				execCalls.push(command);
				return { exitCode: 0 };
			},
		},
		dispose() {
			isDisposed = true;
		},
	};
	return { sandbox, execCalls, disposed: () => isDisposed };
}

function harness(options: HarnessOptions = {}) {
	const handlers = new Map<string, Handler[]>();
	const commands = new Map<
		string,
		{
			handler: (args: string, ctx: ExtensionContext) => Promise<void>;
			getArgumentCompletions?: (prefix: string) => { value: string; label: string; description?: string }[] | null;
		}
	>();
	const registeredTools = new Map<string, ToolDefinition>();
	const controller = new PermissionController(options.mode ?? "readonly");
	const pi = {
		on: (event: string, handler: Handler) => {
			handlers.set(event, [...(handlers.get(event) ?? []), handler]);
			return () => undefined;
		},
		registerTool: (tool: ToolDefinition) => {
			registeredTools.set(tool.name, tool);
		},
		registerCommand: (
			name: string,
			command: {
				handler: (args: string, ctx: ExtensionContext) => Promise<void>;
				getArgumentCompletions?: (
					prefix: string,
				) => { value: string; label: string; description?: string }[] | null;
			},
		) => {
			commands.set(name, command);
		},
		getSettings: () => ({}),
	} as unknown as ExtensionAPI;

	const answers = [...(options.answers ?? [])];
	const terminalListeners = new Set<(data: string) => { consume?: boolean } | undefined>();
	const statuses: (string | undefined)[] = [];
	const notices: { message: string; level?: string }[] = [];
	const select = vi.fn(async () => answers.shift());
	const ctx = {
		hasUI: options.hasUI ?? true,
		cwd: root,
		ui: {
			select,
			confirm: vi.fn(async () => false),
			input: vi.fn(async () => undefined),
			notify: (message: string, level?: string) => void notices.push({ message, level }),
			setStatus: (_key: string, text: string | undefined) => void statuses.push(text),
			onTerminalInput: (listener: (data: string) => { consume?: boolean } | undefined) => {
				terminalListeners.add(listener);
				return () => terminalListeners.delete(listener);
			},
			theme: { fg: (_color: string, text: string) => text },
		},
	} as unknown as ExtensionContext;

	createPermissionGate({
		controller,
		...(options.trustedPaths === undefined ? {} : { trustedPaths: options.trustedPaths }),
		...(options.sandbox === undefined ? {} : { createSandbox: () => options.sandbox as ReadOnlySandbox }),
	})(pi);

	const emit = async (event: string, payload: Record<string, unknown> = {}) => {
		let result: unknown;
		for (const handler of handlers.get(event) ?? []) {
			const out = await handler({ type: event, ...payload }, ctx);
			if (out !== undefined && out !== null) result = out;
		}
		return result as Verdict | undefined;
	};

	return {
		controller,
		registeredTools,
		commands,
		statuses,
		notices,
		select,
		ctx,
		emit,
		call: (toolName: string, input: Record<string, unknown>) =>
			emit("tool_call", { toolName, toolCallId: "c1", input }),
		start: () => emit("session_start", { reason: "startup" }),
		shutdown: () => emit("session_shutdown", { reason: "quit" }),
		press: (data: string) => [...terminalListeners].some((listener) => listener(data)?.consume === true),
		listenerCount: () => terminalListeners.size,
	};
}

describe("permission mode helpers", () => {
	it("defines only readonly and full, defaulting to readonly", () => {
		expect(PERMISSION_MODES).toEqual(["readonly", "full"]);
		expect(DEFAULT_PERMISSION_MODE).toBe("readonly");
	});

	it("parses modes case-insensitively, accepts read-only alias, and rejects everything else", () => {
		expect(parsePermissionMode("readonly")).toBe("readonly");
		expect(parsePermissionMode(" READONLY ")).toBe("readonly");
		expect(parsePermissionMode("read-only")).toBe("readonly");
		expect(parsePermissionMode("FULL")).toBe("full");
		expect(parsePermissionMode("auto")).toBeUndefined();
		expect(parsePermissionMode("ask")).toBeUndefined();
		expect(parsePermissionMode("yolo")).toBeUndefined();
		expect(parsePermissionMode(undefined)).toBeUndefined();
	});

	it("cycles between readonly and full", () => {
		expect(cyclePermissionMode("readonly")).toBe("full");
		expect(cyclePermissionMode("full")).toBe("readonly");
	});
});

describe("PermissionController", () => {
	it("notifies subscribers on mode change and ignores no-op sets", () => {
		const controller = new PermissionController("readonly");
		const seen: PermissionMode[] = [];
		const unsub = controller.subscribe((mode) => seen.push(mode));
		expect(controller.isReadOnly).toBe(true);
		controller.set("readonly");
		expect(seen).toEqual([]);
		controller.cycle();
		expect(controller.mode).toBe("full");
		expect(controller.isReadOnly).toBe(false);
		controller.cycle();
		expect(controller.mode).toBe("readonly");
		unsub();
		controller.set("full");
		expect(seen).toEqual(["full", "readonly"]);
	});
});

describe("describeReadOnlyBlock", () => {
	it("blocks write, edit, convert_files, powershell, and mutating automation/ltr actions", () => {
		const trusted = new Set<string>();
		expect(describeReadOnlyBlock("write", { path: "a.txt" }, root, trusted, undefined)).toBe("write (a.txt)");
		expect(describeReadOnlyBlock("edit", { path: "a.txt" }, root, trusted, undefined)).toBe("edit (a.txt)");
		expect(describeReadOnlyBlock("convert_files", { inputs: ["a.pdf"], to: "md" }, root, trusted, undefined)).toBe(
			"convert_files",
		);
		expect(describeReadOnlyBlock("powershell", { command: "ls" }, root, trusted, undefined)).toBe("powershell");
		expect(describeReadOnlyBlock("manage_automations", { action: "save" }, root, trusted, undefined)).toBe(
			"manage_automations (save)",
		);
		expect(describeReadOnlyBlock("manage_automations", { action: "delete" }, root, trusted, undefined)).toBe(
			"manage_automations (delete)",
		);
		expect(describeReadOnlyBlock("manage_ltr", { action: "set" }, root, trusted, undefined)).toBe("manage_ltr (set)");
		expect(describeReadOnlyBlock("manage_ltr", { action: "train" }, root, trusted, undefined)).toBe(
			"manage_ltr (train)",
		);
	});

	it("allows read-only tools, listing/status queries, trusted paths, and scratch-dir writes", () => {
		const configFile = join(root, "config.json");
		const scratch = join(root, "scratch");
		const trusted = new Set([configFile]);
		expect(describeReadOnlyBlock("read", { path: "a.txt" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("grep", { pattern: "x" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("bash", { command: "ls" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("search_all_documents", { query: "q" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("manage_automations", { action: "list" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("manage_ltr", { action: "status" }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("write", { path: configFile }, root, trusted, scratch)).toBeUndefined();
		expect(describeReadOnlyBlock("edit", { path: configFile }, root, trusted, scratch)).toBeUndefined();
		expect(
			describeReadOnlyBlock("write", { path: join(scratch, "tmp.txt") }, root, trusted, scratch),
		).toBeUndefined();
		expect(
			describeReadOnlyBlock(
				"convert_files",
				{ inputs: ["a.pdf"], to: "md", outputDir: scratch },
				root,
				trusted,
				scratch,
			),
		).toBeUndefined();
	});
});

describe("createPermissionGate extension", () => {
	it("registers a mode-aware bash tool override and routes user_bash through the sandbox in readonly", async () => {
		const sb = fakeSandbox(join(root, "scratch"));
		const h = harness({ mode: "readonly", sandbox: sb.sandbox });
		expect(h.registeredTools.has("bash")).toBe(true);

		const userBashReadOnly = await h.emit("user_bash", { command: "ls", cwd: root });
		expect(userBashReadOnly?.operations).toBe(sb.sandbox.operations);

		h.controller.set("full");
		const userBashFull = await h.emit("user_bash", { command: "ls", cwd: root });
		expect(userBashFull).toBeUndefined();

		await h.shutdown();
		expect(sb.disposed()).toBe(true);
	});

	it("blocks write/edit/mutating tools in readonly and allows everything in full", async () => {
		const h = harness({ mode: "readonly" });
		const w = await h.call("write", { path: join(root, "x.txt"), content: "hi" });
		expect(w?.block).toBe(true);
		expect(w?.reason).toMatch(/read-only/i);
		expect(w?.reason).toMatch(/--permission-mode full/);

		const e = await h.call("edit", { path: join(root, "x.txt"), edits: [] });
		expect(e?.block).toBe(true);

		const c = await h.call("convert_files", { inputs: ["a.pdf"], to: "md" });
		expect(c?.block).toBe(true);

		const a = await h.call("manage_automations", { action: "save", id: "r1" });
		expect(a?.block).toBe(true);

		const aList = await h.call("manage_automations", { action: "list" });
		expect(aList).toBeUndefined();

		const bashCall = await h.call("bash", { command: "ls" });
		expect(bashCall).toBeUndefined();

		// Switch to full: previously blocked calls now pass.
		h.controller.set("full");
		expect(await h.call("write", { path: join(root, "x.txt"), content: "hi" })).toBeUndefined();
		expect(await h.call("edit", { path: join(root, "x.txt"), edits: [] })).toBeUndefined();
		expect(await h.call("convert_files", { inputs: ["a.pdf"], to: "md" })).toBeUndefined();
		expect(await h.call("manage_automations", { action: "save", id: "r1" })).toBeUndefined();
	});

	it("still allows editing trustedPaths (such as config.json) in readonly mode", async () => {
		const cfg = join(root, "config.json");
		writeFileSync(cfg, "{}");
		const h = harness({ mode: "readonly", trustedPaths: [cfg] });
		expect(await h.call("edit", { path: cfg, edits: [] })).toBeUndefined();
		expect(await h.call("write", { path: cfg, content: "{}" })).toBeUndefined();
		const other = await h.call("write", { path: join(root, "other.json"), content: "{}" });
		expect(other?.block).toBe(true);
	});

	it("appends READONLY_PROMPT_ADDENDUM in readonly mode and nothing in full mode", async () => {
		const h = harness({ mode: "readonly" });
		const ro = await h.emit("before_agent_start", { systemPrompt: "BASE" });
		expect(ro?.systemPrompt).toContain("BASE");
		expect(ro?.systemPrompt).toContain(READONLY_PROMPT_ADDENDUM);

		h.controller.set("full");
		const full = await h.emit("before_agent_start", { systemPrompt: "BASE" });
		expect(full).toBeUndefined();
	});

	it("binds TUI footer status, Shift+Tab toggling, and /permissions command cleanly", async () => {
		const h = harness({ mode: "readonly", answers: ["full - full read and write access (no restrictions)"] });
		await h.start();
		expect(h.statuses.at(-1)).toBe("permissions: read-only");
		expect(h.listenerCount()).toBe(1);

		// Shift+Tab cycles readonly -> full -> readonly.
		expect(h.press("\x1b[Z")).toBe(true);
		expect(h.controller.mode).toBe("full");
		expect(h.statuses.at(-1)).toBe("permissions: full");
		expect(h.notices.at(-1)?.message).toMatch(/Permission mode: full/);

		expect(h.press("\x1b[Z")).toBe(true);
		expect(h.controller.mode).toBe("readonly");
		expect(h.statuses.at(-1)).toBe("permissions: read-only");

		// `/permissions` picker
		const cmd = h.commands.get("permissions");
		expect(cmd).toBeDefined();
		expect(cmd?.getArgumentCompletions?.("")?.map((c) => c.value)).toEqual(["readonly", "full"]);

		await cmd?.handler("", h.ctx);
		expect(h.controller.mode).toBe("full");

		await cmd?.handler("readonly", h.ctx);
		expect(h.controller.mode).toBe("readonly");

		await cmd?.handler("auto", h.ctx);
		expect(h.controller.mode).toBe("readonly");
		expect(h.notices.at(-1)?.level).toBe("warning");
		expect(h.notices.at(-1)?.message).toMatch(/readonly, full/);

		await h.shutdown();
		expect(h.listenerCount()).toBe(0);
	});
});
