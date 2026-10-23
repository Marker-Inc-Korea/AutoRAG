import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import type { RawAnswer } from "jev-use/dist/backends/types.js";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	createPermissionGate,
	PERMISSION_GATE_QUESTION_ID,
	PermissionController,
	type PermissionMode,
} from "../../src/agent/permission-gate.ts";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";
import { QUERY_ROUTE_QUESTION_ID } from "../../src/agent/query-routing.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-permission-session-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

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

describe("gate inside a real pi session (headless)", () => {
	async function run(mode: PermissionMode, command: string) {
		const registration = registerFauxProvider({ api: `faux-perm-${randomUUID()}`, models: [{ id: "perm" }] });
		registration.setResponses(bashThenDone(command));
		const controller = new PermissionController(mode);
		const hosted = await createAutoRAGPiSession({
			cwd: root,
			agentDir: join(root, "agent"),
			model: registration.getModel(),
			getSystemPrompt: () => "system",
			customTools: [],
			extensionFactories: [createPermissionGate({ controller })],
		});
		try {
			await hosted.session.prompt("tidy up");
			return bashResult(hosted.session.messages as never);
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	}

	it("blocks a delete of an original in ask mode and the file survives", async () => {
		const file = join(root, "keep.txt");
		writeFileSync(file, "precious");
		const result = await run("ask", `rm ${file}`);
		expect(existsSync(file)).toBe(true);
		expect(result).toMatch(/approval/i);
	});

	it("deletes it in full mode", async () => {
		const file = join(root, "gone.txt");
		writeFileSync(file, "x");
		await run("full", `rm ${file}`);
		expect(existsSync(file)).toBe(false);
	});

	it("lets a copy-then-edit workflow run in ask mode without anyone to ask", async () => {
		const file = join(root, "a.md");
		writeFileSync(file, "hello");
		const copy = join(root, "a.copy.md");
		await run("ask", `cp ${file} ${copy} && sed -i.bak 's/hello/bye/' ${copy} && rm ${copy}.bak`);
		expect(readFileSync(copy, "utf8")).toBe("bye");
		expect(readFileSync(file, "utf8")).toBe("hello");
		expect(existsSync(`${copy}.bak`)).toBe(false);
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
		const t = await tui(
			"ask",
			bashThenDone(`mv ${file} ${join(root, "elsewhere.pdf")}`),
			undefined,
			async (title) => {
				asked.push(title);
				return "Deny";
			},
		);
		try {
			await t.session.prompt("move the report", { source: "extension" });
			expect(asked).toHaveLength(1);
			expect(asked[0]).toContain(file);
			expect(existsSync(file)).toBe(true);
			expect(bashResult(t.session.messages as never)).toMatch(/denied/i);
		} finally {
			await t.dispose();
		}
	});

	it("runs the operation when the user allows it", async () => {
		const file = join(root, "report.pdf");
		writeFileSync(file, "x");
		const t = await tui("ask", bashThenDone(`rm ${file}`), undefined, async () => "Allow once");
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
		const t = await tui(
			"ask",
			[...bashThenDone(`rm ${file}`), ...bashThenDone(`rm ${file}`)],
			undefined,
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
		const approve = new MockBackend({
			[PERMISSION_GATE_QUESTION_ID]: {
				answer: "allow",
				distribution: { allow: 0.97, deny: 0.03 },
				confidence: 0.97,
			},
		});
		const first = join(root, "approved.txt");
		writeFileSync(first, "x");
		let dialogs = 0;
		const a = await tui("auto", bashThenDone(`rm ${first}`), approve, async () => {
			dialogs += 1;
			return "Deny";
		});
		try {
			await a.session.prompt("tidy", { source: "extension" });
			expect(existsSync(first)).toBe(false);
			expect(dialogs).toBe(0);
		} finally {
			await a.dispose();
		}

		const reject = new MockBackend({
			[PERMISSION_GATE_QUESTION_ID]: { answer: "deny", distribution: { allow: 0.03, deny: 0.97 }, confidence: 0.97 },
		});
		const second = join(root, "rejected.txt");
		writeFileSync(second, "x");
		const titles: string[] = [];
		const b = await tui("auto", bashThenDone(`rm ${second}`), reject, async (title) => {
			titles.push(title);
			return "Allow once";
		});
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
	/** A Jev that routes every query `direct`, approves every gate question, and keeps each gate state. */
	function recordingJev(states: string[]): JevBackend {
		return {
			name: "recording",
			judge: async (request) => {
				const stateText = typeof request.state === "string" ? request.state : JSON.stringify(request.state);
				return {
					answers: request.questions.map((question): RawAnswer => {
						if (question.id === PERMISSION_GATE_QUESTION_ID) {
							states.push(stateText);
							return { answer: "allow", distribution: { allow: 0.99, deny: 0.01 }, confidence: 0.99 };
						}
						if (question.id === QUERY_ROUTE_QUESTION_ID) {
							return {
								answer: "direct",
								distribution: { local: 0.05, web: 0.05, direct: 0.9 },
								confidence: 0.9,
							};
						}
						return { answer: 0.05 };
					}),
				};
			},
		};
	}

	async function auto(responses: FauxResponseStep[], states: string[]) {
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
			jev: { backend: recordingJev(states) },
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

	it("shows the first question and every later turn, never the phase prompts", async () => {
		const states: string[] = [];
		const victim = join(root, "old-notes.txt");
		writeFileSync(victim, "x");
		const t = await auto(
			[
				fauxAssistantMessage("FIRST ANSWER", { stopReason: "stop" }),
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("done", { stopReason: "stop" }),
			],
			states,
		);
		try {
			await t.session.prompt("Organize my notes folder", { source: "interactive" });
			await t.agent.waitForIdle();
			await t.session.prompt("Now delete old-notes.txt, it is obsolete", { source: "interactive" });
			await t.agent.waitForIdle();

			expect(states).toHaveLength(1);
			const state = states[0] ?? "";
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
		const states: string[] = [];
		const victim = join(root, "report-2024.pdf");
		writeFileSync(victim, "x");
		let release: () => void = () => undefined;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let entered: () => void = () => undefined;
		const started = new Promise<void>((resolve) => {
			entered = resolve;
		});
		const held: FauxResponseStep = async () => {
			entered();
			await gate;
			return fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" });
		};
		const t = await auto([held, fauxAssistantMessage("done", { stopReason: "stop" })], states);
		try {
			await t.session.prompt("Clean up old reports", { source: "interactive" });
			await started;
			await t.session.prompt("Only the 2024 ones", { source: "interactive", streamingBehavior: "steer" });
			release();
			await t.agent.waitForIdle();

			expect(states).toHaveLength(1);
			const state = states[0] ?? "";
			expect(state).toContain("[1] Clean up old reports");
			expect(state).toContain("[2] (correction while the agent was working) Only the 2024 ones");
		} finally {
			await t.dispose();
		}
	});

	it("includes an Alt+Enter follow-up queued behind the running search", async () => {
		const states: string[] = [];
		const victim = join(root, "scratch.txt");
		writeFileSync(victim, "x");
		let release: () => void = () => undefined;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		let entered: () => void = () => undefined;
		const started = new Promise<void>((resolve) => {
			entered = resolve;
		});
		const held: FauxResponseStep = async () => {
			entered();
			await gate;
			return fauxAssistantMessage("FIRST ANSWER", { stopReason: "stop" });
		};
		const t = await auto(
			[
				held,
				fauxAssistantMessage([fauxToolCall("bash", { command: `rm ${victim}` })], { stopReason: "toolUse" }),
				fauxAssistantMessage("done", { stopReason: "stop" }),
			],
			states,
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

			expect(states).toHaveLength(1);
			expect(states[0]).toContain("[1] What is in my scratch folder?");
			expect(states[0]).toContain("[2] After that, delete scratch.txt");
		} finally {
			await t.dispose();
		}
	});

	it("brings the earlier conversation back on /resume and starts empty on /new", async () => {
		const states: string[] = [];
		const victim = join(root, "later.txt");
		const kept = join(root, "kept.txt");
		writeFileSync(victim, "x");
		writeFileSync(kept, "x");
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
			states,
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
			expect(states).toHaveLength(1);
			expect(states[0]).toContain("[1] Hello there");
			expect(states[0]).toContain("[2] Delete kept.txt");
			expect(states[0]).not.toContain("archive folder is off limits");

			await t.runtime.switchSession(file);
			await t.runtime.session.bindExtensions({ uiContext: ui(async () => "Deny") });
			await t.runtime.session.prompt("Now delete later.txt", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(states).toHaveLength(2);
			expect(states[1]).toContain("[1] Remember: the archive folder is off limits");
			expect(states[1]).toContain("[2] Now delete later.txt");
			expect(states[1]).not.toContain("Hello there");
		} finally {
			await t.dispose();
		}
	});
});
