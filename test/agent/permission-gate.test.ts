import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import {
	createPermissionGate,
	cyclePermissionMode,
	PERMISSION_GATE_QUESTION_ID,
	PERMISSION_MODES,
	PermissionController,
	type PermissionMode,
	parsePermissionMode,
} from "../../src/agent/permission-gate.ts";
import { UserRequestLog } from "../../src/agent/user-requests.ts";

const ALLOW_ONCE = "Allow once";
const ALLOW_SESSION_PREFIX = "Allow";
const DENY = "Deny";

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
	readonly jev?: JevBackend;
	/** What the user said so far; the gate shows it to Jev. */
	readonly requests?: UserRequestLog;
	readonly maxStateChars?: number;
	/** Dialog answers in order; a function sees the title and options. */
	readonly answers?: (string | undefined | ((title: string, options: string[]) => string | undefined))[];
}

function harness(options: HarnessOptions = {}) {
	const handlers = new Map<string, Handler[]>();
	const commands = new Map<string, { handler: (args: string, ctx: ExtensionContext) => Promise<void> }>();
	const controller = new PermissionController(options.mode ?? "ask");
	const judge = options.jev === undefined ? undefined : createJevJudge({ backend: options.jev });
	const pi = {
		on: (event: string, handler: Handler) => {
			handlers.set(event, [...(handlers.get(event) ?? []), handler]);
			return () => undefined;
		},
		registerCommand: (name: string, command: { handler: (args: string, ctx: ExtensionContext) => Promise<void> }) =>
			void commands.set(name, command),
	} as unknown as ExtensionAPI;
	const answers = [...(options.answers ?? [])];
	const dialogs: { title: string; options: string[] }[] = [];
	const terminalListeners = new Set<(data: string) => { consume?: boolean } | undefined>();
	const statuses: (string | undefined)[] = [];
	const notices: string[] = [];
	const select = vi.fn(async (title: string, choices: string[]) => {
		dialogs.push({ title, options: choices });
		const next = answers.shift();
		return typeof next === "function" ? next(title, choices) : next;
	});
	const ctx = {
		hasUI: options.hasUI ?? true,
		cwd: root,
		signal: undefined,
		ui: {
			select,
			input: vi.fn(async () => "use a copy"),
			notify: (message: string) => void notices.push(message),
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
		...(judge === undefined ? {} : { judge }),
		...(options.requests === undefined ? {} : { requests: options.requests }),
		...(options.maxStateChars === undefined ? {} : { maxStateChars: options.maxStateChars }),
	})(pi);
	let nextId = 0;
	const emit = async (event: string, payload: Record<string, unknown>) => {
		let result: unknown;
		for (const handler of handlers.get(event) ?? []) {
			const out = await handler({ type: event, ...payload }, ctx);
			if (out !== undefined && out !== null) result = out;
		}
		return result as { block?: boolean; reason?: string; systemPrompt?: string } | undefined;
	};
	return {
		controller,
		dialogs,
		select,
		notices,
		statuses,
		ctx,
		emit,
		commands,
		/** Runs one tool call through the gate; on success also reports its result like pi does. */
		async call(toolName: string, input: Record<string, unknown>, execute?: () => void) {
			const toolCallId = `call-${nextId++}`;
			const result = await emit("tool_call", { toolName, toolCallId, input });
			if (result?.block !== true) {
				execute?.();
				await emit("tool_result", { toolName, toolCallId, input, isError: false });
			}
			return result;
		},
		bash: (command: string, execute?: () => void) => emitBash(command, execute),
		start: () => emit("session_start", { reason: "startup" }),
		press: (data: string) => [...terminalListeners].some((listener) => listener(data)?.consume === true),
		listenerCount: () => terminalListeners.size,
	};

	function emitBash(command: string, execute?: () => void) {
		return (async () => {
			const toolCallId = `call-${nextId++}`;
			const result = await emit("tool_call", { toolName: "bash", toolCallId, input: { command } });
			if (result?.block !== true) {
				execute?.();
				await emit("tool_result", { toolName: "bash", toolCallId, input: { command }, isError: false });
			}
			return result;
		})();
	}
}

/** Unique file content, so a test can prove it never leaves the machine. */
const FILE_MARKER = "FILE-CONTENT-MARKER-7f3a";

function original(name: string): string {
	const path = join(root, name);
	writeFileSync(path, FILE_MARKER);
	return path;
}

/** Jev answering the gate question with `answer`. */
function gateJev(answer: "allow" | "deny", confidence = 0.95): JevBackend {
	return new MockBackend({
		[PERMISSION_GATE_QUESTION_ID]: {
			answer,
			distribution: {
				allow: answer === "allow" ? confidence : 1 - confidence,
				deny: answer === "deny" ? confidence : 1 - confidence,
			},
			confidence,
		},
	});
}

describe("permission mode helpers", () => {
	it("parses modes case-insensitively and rejects everything else", () => {
		expect(parsePermissionMode("Ask")).toBe("ask");
		expect(parsePermissionMode(" AUTO ")).toBe("auto");
		expect(parsePermissionMode("full")).toBe("full");
		expect(parsePermissionMode("yolo")).toBeUndefined();
		expect(parsePermissionMode(undefined)).toBeUndefined();
	});

	it("cycles ask -> auto -> full -> ask", () => {
		expect(PERMISSION_MODES).toEqual(["ask", "auto", "full"]);
		expect(cyclePermissionMode("ask")).toBe("auto");
		expect(cyclePermissionMode("auto")).toBe("full");
		expect(cyclePermissionMode("full")).toBe("ask");
	});
});

describe("full mode", () => {
	it("never gates or prompts", async () => {
		const h = harness({ mode: "full" });
		const target = original("a.txt");
		expect(await h.bash(`rm ${target}`)).toBeUndefined();
		expect(await h.call("edit", { path: target })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});
});

describe("ask mode", () => {
	it("lets reads, searches, copies, and new files through without a prompt", async () => {
		const h = harness();
		const a = original("a.pdf");
		expect(await h.bash(`ls ${root} && grep -rn x ${root} | head`)).toBeUndefined();
		expect(await h.bash(`cp ${a} ${join(root, "copy.pdf")}`)).toBeUndefined();
		expect(await h.bash(`mkdir ${join(root, "out")} && echo hi > ${join(root, "out", "n.txt")}`)).toBeUndefined();
		expect(await h.call("write", { path: join(root, "fresh.md") })).toBeUndefined();
		expect(await h.call("read", { path: a })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("asks before deleting, and runs the command only after approval", async () => {
		const h = harness({ answers: [ALLOW_ONCE] });
		const a = original("a.pdf");
		const result = await h.bash(`rm ${a}`);
		expect(result).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toContain("delete");
		expect(h.dialogs[0]?.title).toContain(a);
		expect(h.dialogs[0]?.options).toContain(ALLOW_ONCE);
		expect(h.dialogs[0]?.options).toContain(DENY);
	});

	it("blocks a denied operation and tells the model to work on a copy", async () => {
		const h = harness({ answers: [DENY] });
		const a = original("a.pdf");
		const result = await h.bash(`mv ${a} ${join(root, "b.pdf")}`);
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/denied/i);
		expect(result?.reason).toMatch(/copy/i);
	});

	it("passes the user's note to the model when they deny with feedback", async () => {
		const h = harness({
			answers: [(_title, options) => options.find((option) => option.startsWith("Deny") && option !== DENY)],
		});
		const result = await h.call("edit", { path: original("a.md") });
		expect(result?.block).toBe(true);
		expect(result?.reason).toContain("use a copy");
	});

	it("treats a dismissed dialog as a denial", async () => {
		const h = harness({ answers: [undefined] });
		const result = await h.bash(`rm ${original("a.txt")}`);
		expect(result?.block).toBe(true);
	});

	it("asks before editing or overwriting an original document", async () => {
		const h = harness({ answers: [DENY, DENY, DENY] });
		const a = original("a.md");
		expect((await h.call("edit", { path: a }))?.block).toBe(true);
		expect((await h.call("write", { path: a }))?.block).toBe(true);
		expect((await h.bash(`sed -i 's/a/b/' ${a}`))?.block).toBe(true);
		expect(h.dialogs.map((dialog) => dialog.title).join("\n")).toMatch(/edit/);
		expect(h.dialogs.map((dialog) => dialog.title).join("\n")).toMatch(/overwrite/);
	});

	it("asks for a rename and a move", async () => {
		const h = harness({ answers: [DENY, DENY] });
		const a = original("a.txt");
		mkdirSync(join(root, "archive"));
		expect((await h.bash(`mv ${a} ${join(root, "renamed.txt")}`))?.block).toBe(true);
		expect((await h.bash(`mv ${a} ${join(root, "archive")}`))?.block).toBe(true);
		expect(h.dialogs[0]?.title).toContain("rename");
		expect(h.dialogs[1]?.title).toContain("move");
	});

	it("asks when the target is only known at run time", async () => {
		const h = harness({ answers: [DENY] });
		expect((await h.bash("rm $TARGET"))?.block).toBe(true);
		expect(h.dialogs[0]?.title).toContain("rm $TARGET");
	});

	it("blocks without a prompt when nobody can answer", async () => {
		const h = harness({ hasUI: false });
		const result = await h.bash(`rm ${original("a.txt")}`);
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/approval/i);
		expect(result?.reason).toMatch(/--permission-mode/);
		expect(h.select).not.toHaveBeenCalled();
	});

	it("remembers 'allow for this session' per operation kind only", async () => {
		const h = harness({
			answers: [
				(_title, options) =>
					options.find((option) => option.startsWith(ALLOW_SESSION_PREFIX) && option !== ALLOW_ONCE),
				DENY,
			],
		});
		const a = original("a.txt");
		const b = original("b.txt");
		const c = original("c.txt");
		expect(await h.bash(`rm ${a}`)).toBeUndefined();
		expect(await h.bash(`rm ${b}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect((await h.bash(`mv ${c} ${join(root, "d.txt")}`))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(2);
	});

	it("never gates files the agent created itself, so copy-then-edit needs no approval", async () => {
		const h = harness({ answers: [DENY] });
		const a = original("a.md");
		const copy = join(root, "a.edited.md");
		await h.bash(`cp ${a} ${copy}`, () => writeFileSync(copy, "original"));
		expect(await h.call("edit", { path: copy })).toBeUndefined();
		expect(await h.call("write", { path: copy })).toBeUndefined();
		expect(await h.bash(`sed -i 's/a/b/' ${copy}`)).toBeUndefined();
		expect(await h.bash(`mv ${copy} ${join(root, "final.md")}`)).toBeUndefined();
		expect(await h.bash(`rm ${join(root, "final.md")}`)).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
		// The original it was copied from is still protected.
		expect((await h.bash(`rm ${a}`))?.block).toBe(true);
		expect(h.select).toHaveBeenCalledTimes(1);
	});

	it("trusts files created earlier in the same command, and everything inside a folder it made", async () => {
		const h = harness({ answers: [DENY] });
		const out = join(root, "sorted");
		expect(
			await h.bash(`mkdir ${out} && echo x > ${join(out, "n.txt")} && rm ${join(out, "n.txt")}`, () =>
				mkdirSync(out),
			),
		).toBeUndefined();
		const nested = join(out, "deep.txt");
		await h.call("write", { path: nested }, () => writeFileSync(nested, "x"));
		expect(await h.call("edit", { path: nested })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("does not trust a path whose creating command failed", async () => {
		const h = harness({ answers: [DENY] });
		const target = join(root, "never-made.txt");
		const toolCallId = "failed-1";
		await h.emit("tool_call", { toolName: "bash", toolCallId, input: { command: `echo x > ${target}` } });
		await h.emit("tool_result", { toolName: "bash", toolCallId, input: {}, isError: true });
		writeFileSync(target, "someone else's file");
		expect((await h.call("edit", { path: target }))?.block).toBe(true);
	});

	it("ignores tools it does not manage, including external MCP tools", async () => {
		const h = harness({ answers: [DENY] });
		expect(await h.call("mcp__files__delete", { command: "rm -rf /", path: "/x" })).toBeUndefined();
		expect(await h.call("search_all_documents", { query: "rm a" })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("gates a saved automation whose command deletes, not one that only searches", async () => {
		const h = harness({ answers: [DENY] });
		const save = (run: string[]) =>
			h.call("manage_automations", { action: "save", id: "r", automation: { paths: [root], run } });
		expect(await save(["autorag", "search", "summarize new file"])).toBeUndefined();
		expect((await save(["sh", "-c", `rm ${original("old.txt")}`]))?.block).toBe(true);
		expect(await h.call("manage_automations", { action: "list" })).toBeUndefined();
		expect(h.select).toHaveBeenCalledTimes(1);
	});

	it("shows several operations of one command in a single prompt", async () => {
		const h = harness({ answers: [ALLOW_ONCE] });
		const a = original("a.txt");
		const b = original("b.txt");
		await h.bash(`rm ${a} ${b}`);
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toContain(a);
		expect(h.dialogs[0]?.title).toContain(b);
	});

	it("asks one question at a time when calls overlap", async () => {
		const releases: ((answer: string) => void)[] = [];
		const h = harness();
		h.select.mockImplementation(
			(title: string, options: string[]) =>
				new Promise<string | undefined>((resolve) => {
					h.dialogs.push({ title, options });
					releases.push(resolve);
				}),
		);
		const first = h.bash(`rm ${original("a.txt")}`);
		const second = h.bash(`rm ${original("b.txt")}`);
		await vi.waitFor(() => expect(releases).toHaveLength(1));
		// The second call must wait for the first dialog instead of stacking another one.
		await Promise.resolve();
		expect(h.dialogs).toHaveLength(1);
		releases[0]?.(ALLOW_ONCE);
		await vi.waitFor(() => expect(releases).toHaveLength(2));
		releases[1]?.(ALLOW_ONCE);
		await Promise.all([first, second]);
		expect(h.dialogs).toHaveLength(2);
	});
});

describe("auto mode", () => {
	it("lets Jev approve without bothering the user, and says so", async () => {
		const h = harness({ mode: "auto", jev: gateJev("allow") });
		const a = original("a.txt");
		expect(await h.bash(`rm ${a}`)).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
		expect(h.notices.join("\n")).toMatch(/jev/i);
		expect(h.notices.join("\n")).toContain("rm");
	});

	it("falls back to asking the user when Jev rejects, showing why", async () => {
		const h = harness({ mode: "auto", jev: gateJev("deny"), answers: [ALLOW_ONCE] });
		expect(await h.bash(`rm ${original("a.txt")}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toMatch(/jev/i);
		expect(h.dialogs[0]?.title).toMatch(/reject|denied|declined/i);
	});

	it("blocks when Jev rejects and the user also denies", async () => {
		const h = harness({ mode: "auto", jev: gateJev("deny"), answers: [DENY] });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
	});

	it("asks the user when Jev is not confident enough", async () => {
		const h = harness({ mode: "auto", jev: gateJev("allow", 0.55), answers: [DENY] });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(1);
	});

	it("asks the user when Jev is unreachable or unconfigured", async () => {
		const broken = new MockBackend();
		broken.judge = async () => {
			throw new Error("network down");
		};
		const unreachable = harness({ mode: "auto", jev: broken, answers: [DENY] });
		expect((await unreachable.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		expect(unreachable.dialogs).toHaveLength(1);

		const none = harness({ mode: "auto", answers: [DENY] });
		expect((await none.bash(`rm ${original("b.txt")}`))?.block).toBe(true);
		expect(none.dialogs).toHaveLength(1);
	});

	it("blocks without a UI when Jev does not approve", async () => {
		const h = harness({ mode: "auto", hasUI: false, jev: gateJev("deny") });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		const approved = harness({ mode: "auto", hasUI: false, jev: gateJev("allow") });
		expect(await approved.bash(`rm ${original("b.txt")}`)).toBeUndefined();
	});

	it("never sends secrets or file contents to Jev, only the operation and the request", async () => {
		const seen: string[] = [];
		const backend: JevBackend = {
			name: "spy",
			judge: async (request) => {
				seen.push(JSON.stringify(request));
				return {
					answers: request.questions.map((question) => ({
						id: question.id,
						answer: "allow",
						distribution: { allow: 0.99, deny: 0.01 },
						confidence: 0.99,
					})),
				} as never;
			},
		};
		const requests = new UserRequestLog();
		requests.record("clean up my downloads");
		const h = harness({ mode: "auto", jev: backend, requests });
		await h.bash(`rm ${original("a.txt")} # sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef`);
		expect(seen).toHaveLength(1);
		expect(seen[0]).toContain("clean up my downloads");
		expect(seen[0]).not.toContain("sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef");
		expect(seen[0]).not.toContain(FILE_MARKER);
	});
});

/** A Jev that approves everything and keeps the state it was shown. */
function spyJev(states: string[]): JevBackend {
	return {
		name: "spy",
		judge: async (request) => {
			states.push(typeof request.state === "string" ? request.state : JSON.stringify(request.state));
			return {
				answers: request.questions.map(() => ({
					answer: "allow",
					distribution: { allow: 0.99, deny: 0.01 },
					confidence: 0.99,
				})),
			};
		},
	};
}

describe("what Jev is told about the request", () => {
	it("shows every user message of the conversation in order, corrections marked", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		requests.record("organize my downloads folder");
		requests.record("the invoices from 2024 are obsolete, remove them", "request");
		requests.record("not the 2025 ones", "steer");
		const h = harness({ mode: "auto", jev: spyJev(states), requests });
		await h.bash(`rm ${original("invoice-2024.pdf")}`);
		expect(states).toHaveLength(1);
		const state = states[0] ?? "";
		const first = state.indexOf("[1] organize my downloads folder");
		const second = state.indexOf("[2] the invoices from 2024 are obsolete, remove them");
		const third = state.indexOf("[3] (correction while the agent was working) not the 2025 ones");
		expect(first).toBeGreaterThan(-1);
		expect(second).toBeGreaterThan(first);
		expect(third).toBeGreaterThan(second);
	});

	it("does not cut a long request at 2000 characters", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		const long = `please tidy up ${"details ".repeat(1500)}THE-END-MARKER`;
		requests.record(long);
		const h = harness({ mode: "auto", jev: spyJev(states), requests });
		await h.bash(`rm ${original("a.txt")}`);
		expect(long.length).toBeGreaterThan(10_000);
		expect(states[0]).toContain(long);
	});

	it("keeps the whole state under Jev's context limit, dropping the middle of a very long conversation", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		requests.record("OPENING-REQUEST organize everything");
		for (let i = 0; i < 400; i++) requests.record(`filler message ${i} ${"z".repeat(300)}`);
		requests.record("LATEST-STEER only the pdfs", "steer");
		const h = harness({ mode: "auto", jev: spyJev(states), requests, maxStateChars: 8000 });
		await h.bash(`rm ${original("a.txt")}`);
		const state = states[0] ?? "";
		expect(state.length).toBeLessThanOrEqual(8000);
		expect(state).toContain("OPENING-REQUEST");
		expect(state).toContain("LATEST-STEER only the pdfs");
		expect(state).toMatch(/earlier messages omitted/);
		expect(state).toContain(`${root}`);
		expect(state).toContain("rm ");
	});

	it("by default leaves room for the operations and Jev's own question under its 30k-token ceiling", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		requests.record(`start ${"word ".repeat(80_000)} end`);
		const h = harness({ mode: "auto", jev: spyJev(states), requests });
		await h.bash(`rm ${original("a.txt")}`);
		// jev-use estimates ~4 characters per token and hands the question back above 30,000 tokens.
		expect(Math.ceil((states[0] ?? "").length / 4)).toBeLessThan(30_000);
		expect(states).toHaveLength(1);
		expect(states[0]).toContain("rm ");
	});

	it("still answers fast when a message holds a huge unbroken blob, and still redacts around it", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		const key = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef";
		// jev-use's redaction slows down quadratically on long runs without whitespace (108k took ~19s).
		requests.record(`delete the old export, my key is ${key} and the blob is ${"A".repeat(300_000)} thanks`);
		const h = harness({ mode: "auto", jev: spyJev(states), requests });
		const started = performance.now();
		await h.bash(`rm ${original("export.csv")}`);
		expect(performance.now() - started).toBeLessThan(5000);
		const state = states[0] ?? "";
		expect(state).toContain("delete the old export");
		expect(state).toContain("thanks");
		expect(state).not.toContain(key);
		expect(state).toContain("[redacted]");
		expect(state.length).toBeLessThan(10_000);
	}, 30_000);

	it("says so when no user message is known instead of inventing one", async () => {
		const states: string[] = [];
		const h = harness({ mode: "auto", jev: spyJev(states) });
		await h.bash(`rm ${original("a.txt")}`);
		expect(states[0]).toContain("(no user message available)");
	});

	it("never shows Jev AutoRAG's internal phase prompts", async () => {
		const states: string[] = [];
		const requests = new UserRequestLog();
		requests.record("delete old-notes.txt");
		const h = harness({ mode: "auto", jev: spyJev(states), requests });
		// A phase prompt reaches pi as a before_agent_start of source "rpc"/"extension"; it never enters the log.
		await h.emit("before_agent_start", {
			messageType: "user",
			prompt: "Answer this query directly from your own general knowledge: delete old-notes.txt",
			systemPrompt: "base",
		});
		await h.bash(`rm ${original("old-notes.txt")}`);
		expect(states[0]).toContain("[1] delete old-notes.txt");
		expect(states[0]).not.toContain("Answer this query directly");
	});
});

describe("switching modes", () => {
	it("applies the new mode to the very next call and clears session grants", async () => {
		const h = harness({
			answers: [(_t, options) => options.find((o) => o !== ALLOW_ONCE && o.startsWith("Allow")), DENY],
		});
		await h.bash(`rm ${original("a.txt")}`);
		h.controller.set("full");
		expect(await h.bash(`rm ${original("b.txt")}`)).toBeUndefined();
		h.controller.set("ask");
		expect((await h.bash(`rm ${original("c.txt")}`))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(2);
	});

	it("shows the mode in the status bar and updates it on change", async () => {
		const h = harness({ mode: "ask" });
		await h.start();
		expect(h.statuses.at(-1)).toMatch(/ask/i);
		h.controller.cycle();
		expect(h.statuses.at(-1)).toMatch(/auto/i);
		h.controller.cycle();
		expect(h.statuses.at(-1)).toMatch(/full/i);
	});

	it("cycles on Shift+Tab, consuming the key, and installs the listener once across restarts", async () => {
		const h = harness({ mode: "ask" });
		await h.start();
		await h.start();
		expect(h.listenerCount()).toBe(1);
		expect(h.press("\u001b[Z")).toBe(true);
		expect(h.controller.mode).toBe("auto");
		expect(h.press("\u001b[Z")).toBe(true);
		expect(h.controller.mode).toBe("full");
		expect(h.press("a")).toBe(false);
		expect(h.controller.mode).toBe("full");
	});

	it("ignores Shift+Tab while an approval dialog is open", async () => {
		let resolveDialog: (value: string) => void = () => undefined;
		const h = harness();
		h.select.mockImplementationOnce(
			() => new Promise<string | undefined>((resolve) => (resolveDialog = resolve as (value: string) => void)),
		);
		await h.start();
		const pending = h.bash(`rm ${original("a.txt")}`);
		await vi.waitFor(() => expect(h.select).toHaveBeenCalled());
		expect(h.press("\u001b[Z")).toBe(true);
		expect(h.controller.mode).toBe("ask");
		resolveDialog(ALLOW_ONCE);
		await pending;
	});

	it("supports /permissions with an argument and rejects unknown modes", async () => {
		const h = harness({ mode: "ask" });
		const command = h.commands.get("permissions");
		expect(command).toBeDefined();
		await command?.handler("full", h.ctx);
		expect(h.controller.mode).toBe("full");
		await command?.handler("nonsense", h.ctx);
		expect(h.controller.mode).toBe("full");
		expect(h.notices.at(-1)).toMatch(/ask, auto, full|full, ask, auto/);
	});

	it("opens a picker for /permissions without an argument", async () => {
		const h = harness({
			mode: "ask",
			answers: [(_t, options) => options.find((option) => option.toLowerCase().startsWith("auto"))],
		});
		await h.commands.get("permissions")?.handler("", h.ctx);
		expect(h.controller.mode).toBe("auto");
	});
});

describe("system prompt", () => {
	it("tells the model to copy before editing in ask and auto, and says nothing in full", async () => {
		const prompt = async (mode: PermissionMode) =>
			(
				await harness({ mode }).emit("before_agent_start", {
					messageType: "user",
					prompt: "hi",
					systemPrompt: "BASE",
				})
			)?.systemPrompt;
		for (const mode of ["ask", "auto"] as const) {
			const text = await prompt(mode);
			expect(text).toContain("BASE");
			expect(text).toMatch(/copy/i);
			// An explicit request must reach the approval prompt instead of being refused by the model.
			expect(text).toMatch(/explicitly asks/i);
			expect(text).toMatch(/do not refuse/i);
		}
		expect(await prompt("full")).toBeUndefined();
	});
});
