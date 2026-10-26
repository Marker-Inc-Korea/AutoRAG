import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import type { JevBackend } from "jev-use";
import type { BackendRequest, RawAnswer } from "jev-use/dist/backends/types.js";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { createPermissionGate, PermissionController, type PermissionMode } from "../../src/agent/permission-gate.ts";
import {
	PERMISSION_GATE_QUESTION_ID,
	PERMISSION_KINDS,
	type PermissionKind,
	permissionKindQuestionId,
} from "../../src/agent/permission-judgment.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { QUERY_ROUTE_QUESTION_ID } from "../../src/agent/query-routing.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-session-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** A promise and its resolver. tsconfig lib is ES2022, which predates Promise.withResolvers. */
function deferred(): { promise: Promise<void>; resolve: () => void } {
	let resolve: () => void = () => undefined;
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

/** The interactive UI surface the TUI binds, with no visible effect; only the dialogs are scripted. */
function ui(select: ExtensionUIContext["select"]): ExtensionUIContext {
	const noop = () => undefined;
	return {
		select,
		confirm: async () => false,
		input: async () => undefined,
		notify: noop,
		onTerminalInput: () => noop,
		setStatus: noop,
		setWorkingMessage: noop,
		setWorkingVisible: noop,
		setWorkingIndicator: noop,
		setHiddenThinkingLabel: noop,
		setWidget: noop,
		setFooter: noop,
		setHeader: noop,
		setTitle: noop,
		custom: async () => undefined as never,
		pasteToEditor: noop,
		setEditorText: noop,
		getEditorText: () => "",
		editor: async () => undefined,
		addAutocompleteProvider: noop,
		setEditorComponent: noop,
		getEditorComponent: noop,
		theme: { fg: (_color: string, text: string) => text } as never,
		getAllThemes: () => [],
		getTheme: noop,
		setTheme: () => ({ success: false, error: "UI not available" }),
		getToolsExpanded: () => false,
		setToolsExpanded: noop,
	} as unknown as ExtensionUIContext;
}

/** The model asks `bash` to run `command`, then answers `done`. */
function bashThenDone(command: string): FauxResponseStep[] {
	return [
		fauxAssistantMessage([fauxToolCall("bash", { command })], { stopReason: "toolUse" }),
		fauxAssistantMessage("done", { stopReason: "stop" }),
	];
}

/** Toolresult text of the first bash call in a finished session. */
function bashResult(messages: readonly { role: string; toolName?: string; content?: unknown }[]): string {
	const result = messages.find((message) => message.role === "toolResult" && message.toolName === "bash");
	return JSON.stringify(result?.content ?? "");
}

// ---------------------------------------------------------------------------
// A Jev double that classifies the way Jev would for the calls a test scripts.
// ---------------------------------------------------------------------------

interface ScriptedJev {
	readonly backend: JevBackend;
	/** The state of every permission request, in order. */
	readonly states: string[];
	/** How many permission requests Jev answered. */
	readonly count: () => number;
}

/**
 * Jev that answers a query-routing question `extra` and a permission request by looking up the proposed call's
 * text in `rules`: the first rule whose `match` the state contains decides the categories (default: safe) and
 * whether it approves.
 */
function scriptedJev(
	rules: readonly { match: string; kinds?: Partial<Record<PermissionKind, number>>; approve?: boolean }[] = [],
): ScriptedJev {
	const states: string[] = [];
	const answer = (request: BackendRequest, question: BackendRequest["questions"][number]): RawAnswer => {
		const state = typeof request.state === "string" ? request.state : JSON.stringify(request.state);
		const proposed = state.slice(state.lastIndexOf("Proposed tool call"));
		const rule = rules.find((candidate) => proposed.includes(candidate.match));
		if (question.id === PERMISSION_GATE_QUESTION_ID) {
			const allow = rule?.approve === true;
			return {
				answer: allow ? "allow" : "deny",
				distribution: allow ? { allow: 0.97, deny: 0.03 } : { allow: 0.03, deny: 0.97 },
				confidence: 0.97,
			};
		}
		if (question.id === QUERY_ROUTE_QUESTION_ID) {
			return { answer: "extra", distribution: { local: 0.05, organize: 0.05, extra: 0.9 }, confidence: 0.9 };
		}
		for (const kind of PERMISSION_KINDS) {
			if (question.id === permissionKindQuestionId(kind)) return { answer: rule?.kinds?.[kind] ?? 0.02 };
		}
		return { answer: 0.05 };
	};
	const backend: JevBackend = {
		name: "scripted",
		judge: async (request) => {
			if (request.questions.some((question) => question.id === permissionKindQuestionId("delete"))) {
				states.push(typeof request.state === "string" ? request.state : JSON.stringify(request.state));
			}
			return { answers: request.questions.map((question) => answer(request, question)) };
		},
	};
	return { backend, states, count: () => states.length };
}

describe("gate inside a real pi session (headless)", () => {
	async function run(mode: PermissionMode, command: string, jev?: ScriptedJev) {
		const registration = registerFauxProvider({ api: `faux-perm-${randomUUID()}`, models: [{ id: "perm" }] });
		registration.setResponses(bashThenDone(command));
		const controller = new PermissionController(mode);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			getSystemPrompt: () => "system",
			customTools: [],
			extensionFactories: [
				createPermissionGate({
					controller,
					...(jev === undefined
						? {}
						: {
								judge: createJevJudge({ backend: jev.backend }),
							}),
				}),
			],
		});
		try {
			await hosted.session.prompt("tidy up");
			return bashResult(hosted.session.messages as never);
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}

	it("blocks a flagged call and the file survives", async () => {
		const file = join(root, "keep.txt");
		writeFileSync(file, "precious");
		const result = await run("auto", `rm ${file}`, scriptedJev([{ match: "rm ", kinds: { delete: 0.97 } }]));
		expect(existsSync(file)).toBe(true);
		expect(result).toMatch(/approval/i);
	});

	it("blocks the python-subprocess trash that a pattern list could not see", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "precious");
		const script = `python3 - <<'EOF'\nimport subprocess\nsubprocess.run(['/usr/bin/trash', '${file}'])\nEOF`;
		const result = await run("auto", script, scriptedJev([{ match: "subprocess", kinds: { delete: 0.95 } }]));
		expect(existsSync(file)).toBe(true);
		expect(result).toMatch(/approval/i);
	});

	it("runs a call Jev finds safe", async () => {
		const file = join(root, "a.md");
		writeFileSync(file, "hello");
		const copy = join(root, "a.copy.md");
		await run("auto", `cp ${file} ${copy}`, scriptedJev());
		expect(readFileSync(copy, "utf8")).toBe("hello");
	});

	it("blocks everything gated when there is no Jev to classify, so it fails closed", async () => {
		const file = join(root, "keep.txt");
		writeFileSync(file, "precious");
		const result = await run("auto", `rm ${file}`);
		expect(existsSync(file)).toBe(true);
		expect(result).toMatch(/could not classify/i);
	});

	it("ask mode blocks even a harmless call when nobody can answer, and never asks Jev", async () => {
		const file = join(root, "a.md");
		writeFileSync(file, "hello");
		const copy = join(root, "a.copy.md");
		const jev = scriptedJev();
		const result = await run("ask", `cp ${file} ${copy}`, jev);
		expect(existsSync(copy)).toBe(false);
		expect(result).toMatch(/approval/i);
		expect(jev.count()).toBe(0);
	});

	it("deletes it in full mode without asking Jev", async () => {
		const file = join(root, "gone.txt");
		writeFileSync(file, "x");
		const jev = scriptedJev([{ match: "rm ", kinds: { delete: 0.97 } }]);
		await run("full", `rm ${file}`, jev);
		expect(existsSync(file)).toBe(false);
		expect(jev.count()).toBe(0);
	});
});

describe("gate inside the agent's interactive runtime", () => {
	async function tui(
		mode: PermissionMode,
		responses: FauxResponseStep[],
		jev: JevBackend | undefined,
		select: ExtensionUIContext["select"],
	) {
		const registration = registerFauxProvider({ api: `faux-perm-tui-${randomUUID()}`, models: [{ id: "perm-tui" }] });
		registration.setResponses(responses);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const model = registration.getModel();
		writeFileSync(
			join(agentDir, "settings.json"),
			JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
		);
		const agent = new AutoRAGAgent({
			model,
			apiKey: "test-key",
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
			permissionMode: mode,
			...(jev === undefined ? { jev: false as const } : { jev: { backend: jev } }),
		});
		const hosted = await agent.createPiInteractiveRuntime();
		await hosted.runtime.session.bindExtensions({ uiContext: ui(select) });
		return {
			agent,
			session: hosted.runtime.session,
			dispose: async () => {
				await hosted.dispose();
				registration.unregister();
			},
		};
	}

	it("asks the user, honours Deny, and the file survives", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const asked: string[] = [];
		const jev = scriptedJev([{ match: "mv ", kinds: { move: 0.95 } }]);
		const t = await tui(
			"auto",
			bashThenDone(`mv ${file} ${join(root, "elsewhere.pdf")}`),
			jev.backend,
			async (title) => {
				asked.push(title);
				return "Deny";
			},
		);
		try {
			await t.session.prompt("move the report", { source: "extension" });
			expect(asked).toHaveLength(1);
			expect(asked[0]).toContain(file);
			expect(asked[0]).toMatch(/\bmove\b/);
			expect(existsSync(file)).toBe(true);
			expect(bashResult(t.session.messages as never)).toMatch(/denied/i);
		} finally {
			await t.dispose();
		}
	});

	it("runs the operation when the user allows it", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const jev = scriptedJev([{ match: "rm ", kinds: { delete: 0.95 } }]);
		const t = await tui("auto", bashThenDone(`rm ${file}`), jev.backend, async () => "Allow once");
		try {
			await t.session.prompt("delete the report", { source: "extension" });
			expect(existsSync(file)).toBe(false);
		} finally {
			await t.dispose();
		}
	});

	it("switches mode live: the same call is blocked in ask and runs after the controller moves to full", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const jev = scriptedJev([{ match: "rm ", kinds: { delete: 0.95 } }]);
		const t = await tui(
			"auto",
			[...bashThenDone(`rm ${file}`), ...bashThenDone(`rm ${file}`)],
			jev.backend,
			async () => "Deny",
		);
		try {
			await t.session.prompt("delete it", { source: "extension" });
			expect(existsSync(file)).toBe(true);
			t.agent.permissionControl.set("full");
			await t.session.prompt("delete it again", { source: "extension" });
			expect(existsSync(file)).toBe(false);
		} finally {
			await t.dispose();
		}
	});

	it("auto mode: Jev approves without a dialog; a Jev rejection reaches the user", async () => {
		const first = join(root, "approved.txt");
		writeFileSync(first, "x");
		let dialogs = 0;
		const a = await tui(
			"auto",
			bashThenDone(`rm ${first}`),
			scriptedJev([{ match: "rm ", kinds: { delete: 0.97 }, approve: true }]).backend,
			async () => {
				dialogs += 1;
				return "Deny";
			},
		);
		try {
			await a.session.prompt("tidy", { source: "extension" });
			expect(existsSync(first)).toBe(false);
			expect(dialogs).toBe(0);
		} finally {
			await a.dispose();
		}

		const second = join(root, "rejected.txt");
		writeFileSync(second, "x");
		const titles: string[] = [];
		const b = await tui(
			"auto",
			bashThenDone(`rm ${second}`),
			scriptedJev([{ match: "rm ", kinds: { delete: 0.97 }, approve: false }]).backend,
			async (title) => {
				titles.push(title);
				return "Allow once";
			},
		);
		try {
			await b.session.prompt("tidy", { source: "extension" });
			expect(titles).toHaveLength(1);
			expect(titles[0]).toMatch(/jev/i);
			expect(existsSync(second)).toBe(false);
		} finally {
			await b.dispose();
		}
	});
});

describe("what Jev is told, through the agent's real input paths", () => {
	async function auto(responses: FauxResponseStep[], jev: ScriptedJev) {
		const registration = registerFauxProvider({ api: `faux-perm-state-${randomUUID()}`, models: [{ id: "state" }] });
		registration.setResponses(responses);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		const model = registration.getModel();
		writeFileSync(
			join(agentDir, "settings.json"),
			JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
		);
		const agent = new AutoRAGAgent({
			model,
			apiKey: "test-key",
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
			permissionMode: "auto",
			jev: { backend: jev.backend },
		});
		const hosted = await agent.createPiInteractiveRuntime();
		await hosted.runtime.session.bindExtensions({ uiContext: ui(async () => "Deny") });
		return {
			agent,
			session: hosted.runtime.session,
			runtime: hosted.runtime,
			dispose: async () => {
				await hosted.dispose();
				registration.unregister();
			},
		};
	}

	const flagged = (match: string) => scriptedJev([{ match, kinds: { delete: 0.97 }, approve: true }]);

	it("shows the first question and every later turn, never the phase prompts", async () => {
		const victim = join(root, "old-notes.txt");
		writeFileSync(victim, "x");
		const jev = flagged("rm ");
		const t = await auto(
			[
				fauxAssistantMessage("FIRST ANSWER", { stopReason: "stop" }),
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("done", { stopReason: "stop" }),
			],
			jev,
		);
		try {
			await t.session.prompt("Organize my notes folder", { source: "interactive" });
			await t.agent.waitForIdle();
			await t.session.prompt("Now delete old-notes.txt, it is obsolete", { source: "interactive" });
			await t.agent.waitForIdle();

			expect(jev.count()).toBe(1);
			const state = jev.states[0] ?? "";
			expect(state.indexOf("[1] Organize my notes folder")).toBeGreaterThan(-1);
			expect(state.indexOf("[2] Now delete old-notes.txt, it is obsolete")).toBeGreaterThan(
				state.indexOf("[1] Organize my notes folder"),
			);
			// AutoRAG's own prompts wrap the query; none of that is something the user said.
			expect(state).not.toContain("Answer this query directly");
			expect(state).not.toContain("It needs no search");
			expect(existsSync(victim)).toBe(false);
		} finally {
			await t.dispose();
		}
	});

	it("includes a steer typed while the agent was working", async () => {
		const victim = join(root, "report-2024.pdf");
		writeFileSync(victim, "x");
		const { promise: gate, resolve: release } = deferred();
		const { promise: started, resolve: entered } = deferred();
		const held: FauxResponseStep = async () => {
			entered();
			await gate;
			return fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" });
		};
		const jev = flagged("rm ");
		const t = await auto([held, fauxAssistantMessage("done", { stopReason: "stop" })], jev);
		try {
			await t.session.prompt("Clean up old reports", { source: "interactive" });
			await started;
			await t.session.prompt("Only the 2024 ones", { source: "interactive", streamingBehavior: "steer" });
			release();
			await t.agent.waitForIdle();

			expect(jev.count()).toBe(1);
			const state = jev.states[0] ?? "";
			expect(state).toContain("[1] Clean up old reports");
			expect(state).toContain("[2] (correction while the agent was working) Only the 2024 ones");
		} finally {
			await t.dispose();
		}
	});

	it("includes an Alt+Enter follow-up queued behind the running search", async () => {
		const victim = join(root, "scratch.txt");
		writeFileSync(victim, "x");
		const { promise: gate, resolve: release } = deferred();
		const { promise: started, resolve: entered } = deferred();
		const held: FauxResponseStep = async () => {
			entered();
			await gate;
			return fauxAssistantMessage("FIRST ANSWER", { stopReason: "stop" });
		};
		const jev = flagged("rm ");
		const t = await auto(
			[
				held,
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("done", { stopReason: "stop" }),
			],
			jev,
		);
		try {
			await t.session.prompt("What is in my scratch folder?", { source: "interactive" });
			await started;
			await t.session.prompt("After that, delete scratch.txt", {
				source: "interactive",
				streamingBehavior: "followUp",
			});
			release();
			await t.agent.waitForIdle();

			expect(jev.count()).toBe(1);
			expect(jev.states[0]).toContain("[1] What is in my scratch folder?");
			expect(jev.states[0]).toContain("[2] After that, delete scratch.txt");
		} finally {
			await t.dispose();
		}
	});

	it("shows the agent's own earlier calls, so Jev can tell what it created itself", async () => {
		const original = join(root, "report.md");
		writeFileSync(original, "hello");
		const copy = join(root, "report.edited.md");
		const jev = scriptedJev();
		const t = await auto(
			[
				fauxAssistantMessage([fauxToolCall("bash", { command: `cp ${original} ${copy}` })], {
					stopReason: "toolUse",
				}),
				fauxAssistantMessage([fauxToolCall("bash", { command: `sed -i.bak 's/hello/bye/' ${copy}` })], {
					stopReason: "toolUse",
				}),
				fauxAssistantMessage("done", { stopReason: "stop" }),
			],
			jev,
		);
		try {
			await t.session.prompt("Fix the typo in a copy of report.md", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(jev.count()).toBe(2);
			expect(jev.states[1]).toContain(`cp ${original} ${copy}`);
			expect(readFileSync(copy, "utf8")).toBe("bye");
			expect(readFileSync(original, "utf8")).toBe("hello");
		} finally {
			await t.dispose();
		}
	});

	it("brings the earlier conversation back on /resume and starts empty on /new", async () => {
		const victim = join(root, "later.txt");
		const kept = join(root, "kept.txt");
		writeFileSync(victim, "x");
		writeFileSync(kept, "x");
		const jev = flagged("rm ");
		const t = await auto(
			[
				fauxAssistantMessage("FIRST ANSWER", { stopReason: "stop" }),
				// After /new: a deletion with no earlier conversation.
				fauxAssistantMessage("FRESH ANSWER", { stopReason: "stop" }),
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${kept}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("fresh done", { stopReason: "stop" }),
				// After /resume: the earlier conversation is back.
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("resumed done", { stopReason: "stop" }),
			],
			jev,
		);
		try {
			await t.session.prompt("Remember: the archive folder is off limits", { source: "interactive" });
			await t.agent.waitForIdle();
			const file = t.session.sessionFile;
			if (file === undefined) throw new Error("expected a persisted session");

			await t.runtime.newSession();
			await t.runtime.session.bindExtensions({ uiContext: ui(async () => "Deny") });
			await t.runtime.session.prompt("Hello there", { source: "interactive" });
			await t.agent.waitForIdle();
			await t.runtime.session.prompt("Delete kept.txt", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(jev.count()).toBe(1);
			expect(jev.states[0]).toContain("[1] Hello there");
			expect(jev.states[0]).toContain("[2] Delete kept.txt");
			expect(jev.states[0]).not.toContain("archive folder is off limits");

			await t.runtime.switchSession(file);
			await t.runtime.session.bindExtensions({ uiContext: ui(async () => "Deny") });
			await t.runtime.session.prompt("Now delete later.txt", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(jev.count()).toBe(2);
			expect(jev.states[1]).toContain("[1] Remember: the archive folder is off limits");
			expect(jev.states[1]).toContain("[2] Now delete later.txt");
			expect(jev.states[1]).not.toContain("Hello there");
		} finally {
			await t.dispose();
		}
	});
});
