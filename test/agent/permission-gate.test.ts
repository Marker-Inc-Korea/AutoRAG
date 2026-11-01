import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import type { BackendRequest } from "jev-use/dist/backends/types.js";
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
import {
	PERMISSION_KINDS,
	type PermissionKind,
	permissionKindQuestionId,
} from "../../src/agent/permission-judgment.ts";
import { UserRequestLog } from "../../src/agent/user-requests.ts";

const ALLOW_ONCE = "Allow once";
const DENY = "Deny";

/** Unique file content, so a test can prove it never leaves the machine. */
const FILE_MARKER = "FILE-CONTENT-MARKER-7f3a";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Jev test doubles. The real Jev classifies; these answer the way it would for one scripted call.
// ---------------------------------------------------------------------------

interface AllowVerdict {
	readonly answer: "allow" | "deny";
	readonly confidence?: number;
}

/** Jev saying each listed category is likely (default probability for the rest: 0.02) and `allow` for auto mode. */
function jevSays(flags: Partial<Record<PermissionKind, number>>, allow?: AllowVerdict) {
	const script: Record<
		string,
		{ answer: number | string; distribution?: Record<string, number>; confidence?: number }
	> = {};
	for (const kind of PERMISSION_KINDS) script[permissionKindQuestionId(kind)] = { answer: flags[kind] ?? 0.02 };
	const confidence = allow?.confidence ?? 0.95;
	script[PERMISSION_GATE_QUESTION_ID] = {
		answer: allow?.answer ?? "deny",
		distribution:
			(allow?.answer ?? "deny") === "allow"
				? { allow: confidence, deny: 1 - confidence }
				: { allow: 1 - confidence, deny: confidence },
		confidence,
	};
	return new MockBackend(script);
}

/** Wraps a backend so a test can read every request it received. */
function spy(inner: JevBackend) {
	const requests: BackendRequest[] = [];
	const backend: JevBackend = {
		name: inner.name,
		judge: async (request) => {
			requests.push(request);
			return inner.judge(request);
		},
	};
	const states = () =>
		requests.map((request) => (typeof request.state === "string" ? request.state : JSON.stringify(request.state)));
	return { backend, requests, states };
}

const SAFE = () => jevSays({});

/** The proposed-call part of a state Jev was shown: everything from the call's own heading on. */
function callPartOf(state: string | undefined): string {
	const text = state ?? "";
	return text.slice(text.lastIndexOf("Proposed tool call"));
}

// ---------------------------------------------------------------------------
// Harness: the gate against a minimal pi API.
// ---------------------------------------------------------------------------

type Handler = (event: Record<string, unknown>, ctx: ExtensionContext) => unknown;

interface HarnessOptions {
	readonly mode?: PermissionMode;
	readonly hasUI?: boolean;
	readonly jev?: JevBackend;
	/** What the user said so far; the gate shows it to Jev. */
	readonly requests?: UserRequestLog;
	readonly maxStateChars?: number;
	readonly trustedPaths?: readonly string[];
	/** Dialog answers in order; a function sees the title and options. */
	readonly answers?: (string | undefined | ((title: string, options: string[]) => string | undefined))[];
}

interface Verdict {
	block?: boolean;
	reason?: string;
	systemPrompt?: string;
}

/** Jev for a harness that names none: `ask`, `full`, and trusted paths never reach it, and any call fails. */
function unreachedJev(): JevBackend {
	return {
		name: "unreached",
		async judge() {
			throw new Error("Jev was not expected to be asked");
		},
	};
}

function harness(options: HarnessOptions = {}) {
	const handlers = new Map<string, Handler[]>();
	const commands = new Map<string, { handler: (args: string, ctx: ExtensionContext) => Promise<void> }>();
	const controller = new PermissionController(options.mode ?? "auto");
	const judge = createJevJudge({ backend: options.jev ?? unreachedJev() });
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
	/** The session's branch, as pi's session manager returns it. */
	const branch: unknown[] = [];
	const select = vi.fn(async (title: string, choices: string[]) => {
		dialogs.push({ title, options: choices });
		const next = answers.shift();
		return typeof next === "function" ? next(title, choices) : next;
	});
	const ctx = {
		hasUI: options.hasUI ?? true,
		cwd: root,
		signal: undefined,
		sessionManager: { getBranch: () => branch },
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
		judge,
		...(options.requests === undefined ? {} : { requests: options.requests }),
		...(options.maxStateChars === undefined ? {} : { maxStateChars: options.maxStateChars }),
		...(options.trustedPaths === undefined ? {} : { trustedPaths: options.trustedPaths }),
	})(pi);
	let nextId = 0;
	const emit = async (event: string, payload: Record<string, unknown>) => {
		let result: unknown;
		for (const handler of handlers.get(event) ?? []) {
			const out = await handler({ type: event, ...payload }, ctx);
			if (out !== undefined && out !== null) result = out;
		}
		return result as Verdict | undefined;
	};
	/** One tool call as pi runs it: the gate may block it, otherwise `execute` runs and the result is recorded. */
	async function call(
		toolName: string,
		input: Record<string, unknown>,
		execute?: () => void,
	): Promise<Verdict | undefined> {
		const toolCallId = `call-${nextId++}`;
		const result = await emit("tool_call", { toolName, toolCallId, input });
		const ok = result?.block !== true;
		if (ok) {
			execute?.();
			await emit("tool_result", { toolName, toolCallId, input, isError: false });
		}
		branch.push({
			type: "message",
			id: toolCallId,
			message: {
				role: "assistant",
				content: [{ type: "toolCall", id: toolCallId, name: toolName, arguments: input }],
			},
		});
		branch.push({
			type: "message",
			id: `${toolCallId}-result`,
			message: { role: "toolResult", toolCallId, toolName, isError: !ok, content: [] },
		});
		return result;
	}
	return {
		controller,
		dialogs,
		select,
		notices,
		statuses,
		ctx,
		emit,
		commands,
		branch,
		call,
		bash: (command: string, execute?: () => void) => call("bash", { command }, execute),
		start: () => emit("session_start", { reason: "startup" }),
		press: (data: string) => [...terminalListeners].some((listener) => listener(data)?.consume === true),
		listenerCount: () => terminalListeners.size,
	};
}

function original(name: string): string {
	const path = join(root, name);
	writeFileSync(path, FILE_MARKER);
	return path;
}

// ---------------------------------------------------------------------------

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
	it("never asks Jev and never prompts", async () => {
		const jev = spy(jevSays({ delete: 0.99 }));
		const h = harness({ mode: "full", jev: jev.backend });
		const target = original("a.txt");
		expect(await h.bash(`rm ${target}`)).toBeUndefined();
		expect(await h.call("edit", { path: target })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
		expect(jev.requests).toHaveLength(0);
	});
});

describe("which tool calls Jev classifies", () => {
	it("classifies bash, write, edit, and saved automations, and nothing else", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.bash("ls");
		await h.call("write", { path: join(root, "new.md"), content: "x" });
		await h.call("edit", { path: original("a.md"), edits: [] });
		await h.call("manage_automations", {
			action: "save",
			id: "r",
			automation: { paths: [root], run: ["echo", "hi"] },
		});
		expect(jev.requests).toHaveLength(4);

		await h.call("read", { path: original("r.md") });
		await h.call("grep", { pattern: "x" });
		await h.call("search_all_documents", { query: "rm a" });
		await h.call("mcp__files__delete", { path: "/x" });
		await h.call("manage_automations", { action: "list" });
		expect(jev.requests).toHaveLength(4);
	});
});

/** Jev answering with each backend in turn, one per request; the last one repeats. */
function sequence(...backends: JevBackend[]): JevBackend {
	let next = 0;
	return {
		name: "sequence",
		judge: async (request) => {
			const backend = backends[Math.min(next, backends.length - 1)] as JevBackend;
			next += 1;
			return backend.judge(request);
		},
	};
}

describe("ask mode", () => {
	it("asks about every bash, write, and edit call, even one Jev would find safe, and never asks Jev", async () => {
		const jev = spy(SAFE());
		const h = harness({ mode: "ask", jev: jev.backend, answers: [ALLOW_ONCE, ALLOW_ONCE, ALLOW_ONCE] });
		expect(await h.bash("ls")).toBeUndefined();
		expect(await h.call("write", { path: join(root, "fresh.md"), content: "x" })).toBeUndefined();
		expect(await h.call("edit", { path: original("a.md") })).toBeUndefined();
		expect(h.dialogs).toHaveLength(3);
		expect(jev.requests).toHaveLength(0);
	});

	it("asks without saying anything about classification", async () => {
		const h = harness({ mode: "ask", answers: [DENY] });
		expect((await h.bash("ls"))?.block).toBe(true);
		expect(h.dialogs[0]?.title).not.toMatch(/classify|Jev/i);
	});

	it("shows the tool and the command, and blocks a denied call with advice to use a copy", async () => {
		const h = harness({ mode: "ask", answers: [DENY] });
		const a = original("a.pdf");
		const result = await h.bash(`rm ${a}`);
		expect(h.dialogs[0]?.title).toContain("bash");
		expect(h.dialogs[0]?.title).toContain(`rm ${a}`);
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/copy/i);
	});

	it("does not ask about tools that never touch files", async () => {
		const h = harness({ mode: "ask" });
		expect(await h.call("read", { path: original("a.md") })).toBeUndefined();
		expect(await h.call("grep", { pattern: "x" })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("lets the user allow every such call for the session, and that ends with the mode", async () => {
		const allowSession = (_title: string, options: string[]) =>
			options.find((option) => option.startsWith("Allow all"));
		const h = harness({ mode: "ask", answers: [allowSession, DENY] });
		expect(await h.bash("ls")).toBeUndefined();
		expect(await h.call("write", { path: join(root, "n.md"), content: "x" })).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		h.controller.set("auto");
		h.controller.set("ask");
		expect((await h.bash("pwd"))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(2);
	});

	it("blocks every gated call when nobody can answer, pointing at the way out", async () => {
		const h = harness({ mode: "ask", hasUI: false });
		const result = await h.bash("ls");
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/--permission-mode/);
		expect(h.select).not.toHaveBeenCalled();
	});

	it("still lets a trusted file be written without asking", async () => {
		const config = original("config.json");
		const h = harness({ mode: "ask", trustedPaths: [config] });
		expect(await h.call("write", { path: config, content: "x" })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});
});

describe("default mode (auto): what Jev flags", () => {
	it("lets a call through silently when Jev finds no category", async () => {
		const h = harness({ jev: SAFE() });
		expect(await h.bash(`ls ${root} && cp a.pdf copy.pdf`)).toBeUndefined();
		expect(await h.call("write", { path: join(root, "fresh.md"), content: "x" })).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("asks when Jev says the call deletes, naming the category and showing the command", async () => {
		const h = harness({ jev: jevSays({ delete: 0.97 }), answers: [ALLOW_ONCE] });
		const a = original("a.pdf");
		expect(await h.bash(`rm ${a}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toMatch(/delete/);
		expect(h.dialogs[0]?.title).toContain(`rm ${a}`);
		expect(h.dialogs[0]?.options).toContain(ALLOW_ONCE);
		expect(h.dialogs[0]?.options).toContain(DENY);
	});

	it("asks for every category Jev flags, and says which", async () => {
		const h = harness({
			jev: jevSays({ move: 0.9, rename: 0.8, overwrite: 0.7, edit: 0.9, unclear: 0.9 }),
			answers: [DENY],
		});
		await h.bash("python3 reorganize.py");
		const title = h.dialogs[0]?.title ?? "";
		for (const kind of ["move", "rename", "overwrite", "edit", "unclear"]) expect(title).toContain(kind);
		expect(title).not.toMatch(/\bdelete\b/);
	});

	it("blocks a denied call and tells the model to work on a copy", async () => {
		const h = harness({ jev: jevSays({ move: 0.95 }), answers: [DENY] });
		const result = await h.bash(`mv ${original("a.pdf")} ${join(root, "b.pdf")}`);
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/denied/i);
		expect(result?.reason).toMatch(/copy/i);
	});

	it("passes the user's note to the model when they deny with feedback", async () => {
		const h = harness({
			jev: jevSays({ edit: 0.95 }),
			answers: [(_title, options) => options.find((option) => option.startsWith("Deny") && option !== DENY)],
		});
		const result = await h.call("edit", { path: original("a.md") });
		expect(result?.block).toBe(true);
		expect(result?.reason).toContain("use a copy");
	});

	it("treats a dismissed dialog as a denial", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95 }), answers: [undefined] });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
	});

	it("asks when Jev is unsure about a category, not only when it is certain", async () => {
		const h = harness({ jev: jevSays({ delete: 0.5 }), answers: [DENY] });
		expect((await h.bash("do-something"))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(1);
	});

	it("does not ask when a category is clearly unlikely", async () => {
		const h = harness({ jev: jevSays({ delete: 0.1, edit: 0.05 }) });
		expect(await h.bash("ls")).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
	});

	it("blocks without a prompt when nobody can answer, pointing at the way out", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95 }), hasUI: false });
		const result = await h.bash(`rm ${original("a.txt")}`);
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/approval/i);
		expect(result?.reason).toMatch(/--permission-mode/);
		expect(h.select).not.toHaveBeenCalled();
	});

	it("still lets a clearly safe call run without a UI", async () => {
		const h = harness({ jev: SAFE(), hasUI: false });
		expect(await h.bash("ls")).toBeUndefined();
	});

	it("asks one question at a time when calls overlap", async () => {
		const releases: ((answer: string) => void)[] = [];
		const h = harness({ jev: jevSays({ delete: 0.95 }) });
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
		await Promise.resolve();
		expect(h.dialogs).toHaveLength(1);
		releases[0]?.(ALLOW_ONCE);
		await vi.waitFor(() => expect(releases).toHaveLength(2));
		releases[1]?.(ALLOW_ONCE);
		await Promise.all([first, second]);
		expect(h.dialogs).toHaveLength(2);
	});
});

describe("when Jev cannot classify", () => {
	function failing(message: string): JevBackend {
		const backend = new MockBackend();
		backend.judge = async () => {
			throw new Error(message);
		};
		return backend;
	}

	it("asks the user rather than letting the call through, and says why", async () => {
		const h = harness({ jev: failing("network down"), answers: [DENY] });
		expect((await h.bash("ls"))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toMatch(/could not classify/i);
		expect(h.dialogs[0]?.title).toMatch(/network down/);
	});

	it("blocks headless with a reason naming Jev and the mode flag", async () => {
		const h = harness({ jev: failing("boom"), hasUI: false });
		const result = await h.bash("ls");
		expect(result?.block).toBe(true);
		expect(result?.reason).toMatch(/jev/i);
		expect(result?.reason).toMatch(/--permission-mode/);
	});

	it("lets the user allow the unclassifiable calls for the session, so a Jev outage is survivable", async () => {
		const h = harness({
			jev: failing("boom"),
			answers: [(_title, options) => options.find((option) => option.startsWith("Allow all"))],
		});
		expect(await h.bash("ls")).toBeUndefined();
		expect(await h.bash("pwd")).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
	});
});

describe("allowing for the session", () => {
	const allowSession = (_title: string, options: string[]) => options.find((option) => option.startsWith("Allow all"));

	it("remembers a category, so the next call in it passes while other categories still ask", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95 }), answers: [allowSession] });
		expect(await h.bash(`rm ${original("a.txt")}`)).toBeUndefined();
		expect(await h.bash(`rm ${original("b.txt")}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.options.join("\n")).toMatch(/Allow all delete/);
	});

	it("still asks Jev each time, because the category is what is remembered", async () => {
		const jev = spy(jevSays({ delete: 0.95 }));
		const h = harness({ jev: jev.backend, answers: [allowSession] });
		await h.bash(`rm ${original("a.txt")}`);
		await h.bash(`rm ${original("b.txt")}`);
		expect(jev.requests).toHaveLength(2);
	});

	it("does not carry over to a different category: a move still asks after delete was allowed", async () => {
		const jev = sequence(jevSays({ delete: 0.95 }), jevSays({ move: 0.95 }), jevSays({ delete: 0.95 }));
		const h = harness({ jev, answers: [allowSession, DENY] });
		expect(await h.bash(`rm ${original("a.txt")}`)).toBeUndefined();
		expect((await h.bash("mv a b"))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(2);
		expect(h.dialogs[1]?.title).toMatch(/\bmove\b/);
		// delete is still granted afterwards
		expect(await h.bash(`rm ${original("c.txt")}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(2);
	});

	it("needs every flagged category granted", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95, move: 0.95 }), answers: [allowSession, DENY] });
		await h.bash("mv a b && rm c");
		expect(h.dialogs[0]?.options.join("\n")).toMatch(/Allow all (delete\/move|move\/delete)/);

		expect(h.controller.isGranted("delete")).toBe(true);
		expect(h.controller.isGranted("move")).toBe(true);
		expect(h.controller.isGranted("edit")).toBe(false);
	});

	it("ends when the mode changes", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95 }), answers: [allowSession, DENY] });
		await h.bash(`rm ${original("a.txt")}`);
		h.controller.set("full");
		expect(await h.bash(`rm ${original("b.txt")}`)).toBeUndefined();
		h.controller.set("auto");
		expect((await h.bash(`rm ${original("c.txt")}`))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(2);
	});
});

describe("auto mode", () => {
	it("asks Jev the categories and its approval together, in one request", async () => {
		const jev = spy(jevSays({ delete: 0.97 }, { answer: "allow" }));
		const h = harness({ mode: "auto", jev: jev.backend });
		await h.bash(`rm ${original("a.txt")}`);
		expect(jev.requests).toHaveLength(1);
		expect(jev.requests[0]?.questions.map((question) => question.id)).toEqual([
			...PERMISSION_KINDS.map(permissionKindQuestionId),
			PERMISSION_GATE_QUESTION_ID,
		]);
	});

	it("lets Jev approve a flagged call without bothering the user, and says so", async () => {
		const h = harness({ mode: "auto", jev: jevSays({ delete: 0.97 }, { answer: "allow" }) });
		const a = original("a.txt");
		expect(await h.bash(`rm ${a}`)).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
		expect(h.notices.join("\n")).toMatch(/Jev approved/);
		expect(h.notices.join("\n")).toContain("delete");
		expect(h.notices.join("\n")).toContain("rm");
	});

	it("does not need Jev's approval, or say anything, for a call it finds safe", async () => {
		const h = harness({ mode: "auto", jev: SAFE() });
		expect(await h.bash("ls")).toBeUndefined();
		expect(h.select).not.toHaveBeenCalled();
		expect(h.notices).toEqual([]);
	});

	it("falls back to asking the user when Jev rejects, showing why", async () => {
		const h = harness({ mode: "auto", jev: jevSays({ delete: 0.97 }, { answer: "deny" }), answers: [ALLOW_ONCE] });
		expect(await h.bash(`rm ${original("a.txt")}`)).toBeUndefined();
		expect(h.dialogs).toHaveLength(1);
		expect(h.dialogs[0]?.title).toMatch(/Jev declined/);
		expect(h.dialogs[0]?.title).toMatch(/delete/);
	});

	it("blocks when Jev rejects and the user also denies", async () => {
		const h = harness({ mode: "auto", jev: jevSays({ delete: 0.97 }, { answer: "deny" }), answers: [DENY] });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
	});

	it("asks the user when Jev approves without enough confidence", async () => {
		const h = harness({
			mode: "auto",
			jev: jevSays({ delete: 0.97 }, { answer: "allow", confidence: 0.55 }),
			answers: [DENY],
		});
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		expect(h.dialogs).toHaveLength(1);
	});

	it("blocks without a UI when Jev does not approve, and passes when it does", async () => {
		const rejected = harness({ mode: "auto", hasUI: false, jev: jevSays({ delete: 0.97 }, { answer: "deny" }) });
		expect((await rejected.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		const approved = harness({ mode: "auto", hasUI: false, jev: jevSays({ delete: 0.97 }, { answer: "allow" }) });
		expect(await approved.bash(`rm ${original("b.txt")}`)).toBeUndefined();
	});
});

describe("trusted paths", () => {
	it("lets write and edit of a trusted file through without asking Jev at all", async () => {
		const config = original("config.json");
		const jev = spy(jevSays({ edit: 0.99, overwrite: 0.99 }));
		const h = harness({ jev: jev.backend, trustedPaths: [config] });
		expect(await h.call("edit", { path: config })).toBeUndefined();
		expect(await h.call("write", { path: config, content: "{}" })).toBeUndefined();
		expect(jev.requests).toHaveLength(0);
		// A different file is still classified.
		expect((await h.call("edit", { path: original("other.json") }))?.block).toBe(true);
	});
});

describe("what Jev is shown about the call", () => {
	it("shows a bash command in full", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		const script = `python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['/usr/bin/trash', '${root}/a.pdf'])\nEOF`;
		await h.bash(script);
		expect(jev.states()[0]).toContain(script);
		expect(jev.states()[0]).toContain(root);
	});

	it("tells Jev whether a write or edit target exists, without sending its content", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.call("write", { path: original("exists.md"), content: "NEW-CONTENT-MARKER" });
		await h.call("write", { path: join(root, "missing.md"), content: "NEW-CONTENT-MARKER" });
		await h.call("edit", {
			path: join(root, "exists.md"),
			edits: [{ oldText: "SECRET-OLD", newText: "SECRET-NEW" }],
		});
		const [existing, missing, edited] = jev.states().map(callPartOf);
		expect(existing).toMatch(/exists/i);
		expect(existing).not.toMatch(/does not exist/i);
		expect(missing).toMatch(/does not exist/i);
		expect(edited).toContain(join(root, "exists.md"));
		for (const state of jev.states()) {
			expect(state).not.toContain("NEW-CONTENT-MARKER");
			expect(state).not.toContain("SECRET-OLD");
			expect(state).not.toContain(FILE_MARKER);
		}
	});

	it("shows the earlier calls that worked, so Jev can tell what the agent created itself", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.bash("cp report.md report.edited.md");
		await h.call("write", { path: join(root, "notes.md"), content: "x" });
		await h.bash("sed -i 's/a/b/' report.edited.md");
		const third = jev.states()[2] ?? "";
		expect(third).toContain("cp report.md report.edited.md");
		expect(third).toContain(join(root, "notes.md"));
		// the call being judged is the proposed one, and it is not also listed as history
		expect(callPartOf(third).match(/sed -i/g)).toHaveLength(1);
	});

	it("tells Jev which absolute paths in a bash command exist right now, and which do not", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		const existing = original("target.txt");
		const missing = join(root, "copy.txt");
		await h.bash(`cp ${existing} ${missing}`);
		const state = jev.states()[0] ?? "";
		expect(state).toContain(`${existing}: exists (file)`);
		expect(state).toContain(`${missing}: does not exist`);
	});

	it("names a folder as a folder, and says so when the command names no path", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.bash(`ls ${root}`);
		await h.bash("echo hello");
		expect(callPartOf(jev.states()[0])).toContain(`${root}: exists (folder)`);
		expect(callPartOf(jev.states()[1])).toMatch(/no path in the call/i);
	});

	it("resolves ~ and relative paths against the working directory", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		original("rel.txt");
		await h.bash("cat rel.txt ./rel.txt missing-rel.txt");
		const state = jev.states()[0] ?? "";
		expect(state).toContain(`${join(root, "rel.txt")}: exists (file)`);
		expect(state).toContain(`${join(root, "missing-rel.txt")}: does not exist`);
	});

	it("marks a path the agent created earlier in this session, so a copy it made is not mistaken for an original", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		const source = original("report.md");
		const copy = join(root, "report.edited.md");
		await h.bash(`cp ${source} ${copy}`, () => writeFileSync(copy, "copy"));
		await h.bash(`sed -i.bak 's/old/new/' ${copy}`);
		const state = callPartOf(jev.states()[1]);
		expect(state).toContain(`${copy}: exists (file), created by the agent earlier in this session`);
		// the original it copied from is not the agent's
		await h.bash(`cat ${source}`);
		expect(callPartOf(jev.states()[2])).toMatch(new RegExp(`${source}: exists \\(file\\)$`, "m"));
		expect(callPartOf(jev.states()[2])).not.toContain(`${source}: exists (file), created`);
	});

	it("does not list option flags, sed scripts, or backup suffixes as paths", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		const target = original("report.md");
		await h.bash(`sed -i.bak 's/old/new/' ${target}`);
		const facts = callPartOf(jev.states()[0]).split("Paths named in the call")[1] ?? "";
		expect(facts).toContain(target);
		expect(facts).not.toContain("-i.bak");
		expect(facts).not.toContain("s/old/new");
		expect(facts).not.toMatch(/old\/new/);
	});

	it("lists a path the agent created earlier even when a later command only names it relatively", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.bash("mkdir sorted", () => mkdirSync(join(root, "sorted")));
		await h.bash("ls sorted");
		expect(callPartOf(jev.states()[1])).toContain(
			`${join(root, "sorted")}: exists (folder), created by the agent earlier in this session`,
		);
	});

	it("does not read any file, only checks that the paths exist", async () => {
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend });
		await h.bash(`cat ${original("secret.txt")}`);
		expect(jev.states()[0]).not.toContain(FILE_MARKER);
	});

	it("leaves out earlier calls that failed or were blocked", async () => {
		const jev = spy(jevSays({ delete: 0.95 }));
		const h = harness({ jev: jev.backend, answers: [DENY] });
		await h.bash("rm denied-file.txt");
		await h.bash("ls");
		expect(jev.states()[1]).not.toContain("denied-file.txt");
	});

	it("shows what the user said, every message, steers marked", async () => {
		const requests = new UserRequestLog();
		requests.record("organize my downloads folder");
		requests.record("not the 2025 ones", "steer");
		const jev = spy(SAFE());
		await harness({ jev: jev.backend, requests }).bash("ls");
		expect(jev.states()[0]).toContain("[1] organize my downloads folder");
		expect(jev.states()[0]).toContain("[2] (correction while the agent was working) not the 2025 ones");
	});

	it("does not cut a long request, and says so when there is nothing to show", async () => {
		const requests = new UserRequestLog();
		const long = `please tidy up ${"details ".repeat(1500)}THE-END-MARKER`;
		requests.record(long);
		const jev = spy(SAFE());
		await harness({ jev: jev.backend, requests }).bash("ls");
		await harness({ jev: jev.backend }).bash("ls");
		expect(jev.states()[0]).toContain(long);
		expect(jev.states()[1]).toContain("(no user message available)");
	});

	it("never sends secrets", async () => {
		const jev = spy(SAFE());
		const key = "sk-or-v1-0123456789abcdef0123456789abcdef0123456789abcdef";
		const requests = new UserRequestLog();
		requests.record(`my key is ${key}`);
		const h = harness({ jev: jev.backend, requests });
		await h.bash(`curl -H 'Authorization: Bearer ${key}' https://example.com`);
		await h.bash("ls");
		for (const state of jev.states()) expect(state).not.toContain(key);
		expect(jev.states()[0]).toContain("[redacted]");
	});

	it("keeps the whole state under the budget however long the command, history, and conversation are", async () => {
		const requests = new UserRequestLog();
		requests.record("OPENING-REQUEST organize everything");
		for (let i = 0; i < 300; i++) requests.record(`filler message ${i} ${"z".repeat(300)}`);
		requests.record("LATEST-STEER only the pdfs", "steer");
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend, requests, maxStateChars: 10_000 });
		for (let i = 0; i < 60; i++) await h.bash(`echo step ${i} ${"y".repeat(400)}`);
		jev.requests.length = 0;
		const longCommand = `echo START ${"word ".repeat(8000)} END-OF-COMMAND`;
		await h.bash(longCommand);
		const state = jev.states()[0] ?? "";
		expect(state.length).toBeLessThanOrEqual(10_000);
		expect(state).toContain("OPENING-REQUEST");
		expect(state).toContain("LATEST-STEER only the pdfs");
		expect(state).toContain("START");
		expect(state).toContain("END-OF-COMMAND");
	});

	it("by default stays under Jev's 30k-token ceiling and still answers fast on a huge unbroken blob", async () => {
		const requests = new UserRequestLog();
		requests.record(`delete the old export ${"A".repeat(300_000)} thanks`);
		const jev = spy(SAFE());
		const h = harness({ jev: jev.backend, requests });
		const started = performance.now();
		await h.bash(`echo ${"B".repeat(300_000)}`);
		expect(performance.now() - started).toBeLessThan(5000);
		const state = jev.states()[0] ?? "";
		expect(Math.ceil(state.length / 4)).toBeLessThan(30_000);
		expect(state).toContain("delete the old export");
		expect(state).toContain("thanks");
	}, 30_000);
});

describe("switching modes", () => {
	it("applies the new mode to the very next call", async () => {
		const h = harness({ jev: jevSays({ delete: 0.95 }), answers: [DENY] });
		expect((await h.bash(`rm ${original("a.txt")}`))?.block).toBe(true);
		h.controller.set("full");
		expect(await h.bash(`rm ${original("b.txt")}`)).toBeUndefined();
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
		const h = harness({ jev: jevSays({ delete: 0.95 }) });
		h.select.mockImplementationOnce(
			() => new Promise<string | undefined>((resolve) => (resolveDialog = resolve as (value: string) => void)),
		);
		await h.start();
		const pending = h.bash(`rm ${original("a.txt")}`);
		await vi.waitFor(() => expect(h.select).toHaveBeenCalled());
		expect(h.press("\u001b[Z")).toBe(true);
		expect(h.controller.mode).toBe("auto");
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
	it("tells the model to copy before editing in auto, and says nothing in full", async () => {
		const prompt = async (mode: PermissionMode) =>
			(
				await harness({ mode }).emit("before_agent_start", {
					messageType: "user",
					prompt: "hi",
					systemPrompt: "BASE",
				})
			)?.systemPrompt;
		const text = await prompt("auto");
		expect(text).toContain("BASE");
		expect(text).toMatch(/copy/i);
		// An explicit request must reach the approval prompt instead of being refused by the model.
		expect(text).toMatch(/explicitly asks/i);
		expect(text).toMatch(/do not refuse/i);
		expect(await prompt("full")).toBeUndefined();
	});

	it("in ask, says every file-capable call is approved by the user, and still does not refuse for them", async () => {
		const text = (
			await harness({ mode: "ask" }).emit("before_agent_start", {
				messageType: "user",
				prompt: "hi",
				systemPrompt: "BASE",
			})
		)?.systemPrompt;
		expect(text).toContain("BASE");
		expect(text).toMatch(/every/i);
		expect(text).toMatch(/do not refuse/i);
		expect(text).not.toMatch(/never need approval/i);
	});
});
