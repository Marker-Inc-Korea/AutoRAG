import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import {
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "../../src/agent/pi-session.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
} from "../../src/agent/query-routing.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-pi-session-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function customTool(): AgentTool {
	return {
		name: "search_custom",
		label: "Search custom",
		description: "Custom AutoRAG test tool",
		parameters: Type.Object({ query: Type.String() }),
		async execute() {
			return { content: [{ type: "text", text: "custom" }], details: {} };
		},
	};
}

/** Jev routing every query to `route`; `followUp` is the post-fast-answer "verify?" probability. */
function jevRoute(route: QueryRoute, followUp = 0.9): JevBackend {
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: {
			answer: route,
			distribution: { local: 0.1, web: 0.1, direct: 0.1, [route]: 0.8 },
			confidence: 0.8,
		},
		[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
		[FOLLOW_UP_QUESTION_ID]: { answer: followUp },
	});
}

/** A TUI-hosted agent on the faux model, routed by `jev` (direct route by default). */
async function tuiAgent(
	responses: FauxResponseStep[],
	jev: JevBackend = jevRoute("direct"),
	extra: Pick<AutoRAGAgentOptions, "datasourceSkills"> = {},
) {
	const registration = registerFauxProvider({ api: `faux-tui-${randomUUID()}`, models: [{ id: "tui-model" }] });
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
		jev: { backend: jev },
		...extra,
	});
	const hosted = await agent.createPiInteractiveRuntime();
	return {
		agent,
		registration,
		session: hosted.runtime.session,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

/** Wraps a Jev backend so the first batch asking `questionId` stays in flight until released. */
function holdingJev(inner: JevBackend, questionId: string) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	let held = false;
	const backend: JevBackend = {
		name: inner.name,
		async judge(request) {
			if (!held && request.questions.some((question) => question.id === questionId)) {
				held = true;
				markEntered();
				await gate;
			}
			return inner.judge(request);
		},
	};
	return { backend, entered, release: () => release() };
}

function textOf(content: string | readonly { type: string; text?: string }[]): string {
	return typeof content === "string"
		? content
		: content.map((block) => (block.type === "text" ? (block.text ?? "") : "")).join("");
}

/** A faux step that records the latest user message the model saw, then answers. */
function seeingStep(answer: string, seen: string[]): FauxResponseStep {
	return (context) => {
		const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
		seen.push(lastUser === undefined ? "" : textOf(lastUser.content));
		return fauxAssistantMessage(answer, { stopReason: "stop" });
	};
}

/** A faux model step that stays in flight until the returned `release` is called. */
function heldStep(step: FauxResponseStep) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	const held: FauxResponseStep = async (...args) => {
		markEntered();
		await gate;
		return typeof step === "function" ? step(...args) : step;
	};
	return { held, entered, release: () => release() };
}

/** A datasource whose retrieval stays in flight until released or aborted through its signal. */
function slowDatasource(datasourceId: string) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	let aborted = false;
	const method: RetrievalMethod = {
		describe: () => ({
			name: `${datasourceId}.keyword`,
			type: "bm25",
			description: `${datasourceId} test method`,
			status: "active",
			capabilities: ["keyword"],
			datasourceId,
		}),
		async retrieve(_query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
			markEntered();
			const signal = options.signal;
			await Promise.race([
				gate,
				new Promise<void>((resolve) => {
					signal?.addEventListener("abort", () => resolve(), { once: true });
				}),
			]);
			if (signal?.aborted) aborted = true;
			return [];
		},
	};
	const skill: DatasourceSkill = {
		describe: () => ({
			name: datasourceId,
			type: "chat",
			description: `${datasourceId} test datasource`,
			capabilities: ["keyword"],
			status: "active",
			datasourceId,
			tags: [datasourceId],
		}),
		polling: () => ({ mode: "poll", intervalMs: 60_000 }),
		skillManifest: () => ({
			name: `datasource-${datasourceId}`,
			description: `Search ${datasourceId}.`,
			content: `# ${datasourceId}`,
		}),
		index: async () => ({
			ok: true,
			instanceId: "default",
			skill: datasourceId,
			chunkCount: 0,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [method],
		describeSources: () => [],
	};
	return { skill, entered, aborted: () => aborted, release: () => release() };
}

describe("AutoRAG pi coding-agent host", () => {
	it("registers AutoRAG tools with the pi session and persists resumable transcripts", async () => {
		const registration = registerFauxProvider({
			api: `faux-${Date.now()}`,
			models: [{ id: "pi-host-test" }],
		});
		registration.setResponses([fauxAssistantMessage("hosted response")]);
		try {
			const first = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				persistSession: true,
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [customTool()],
			});
			expect(first.session.getActiveToolNames()).toEqual(
				expect.arrayContaining([...PI_BUILTIN_TOOL_NAMES, "search_custom"]),
			);
			await first.session.prompt("hello");
			const sessionFile = first.session.sessionFile;
			expect(sessionFile).toBeTypeOf("string");
			first.session.dispose();

			if (sessionFile === undefined) throw new Error("expected a persisted pi session");
			const resumed = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				sessionPath: sessionFile,
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [customTool()],
			});
			expect(resumed.session.messages.some((message) => message.role === "user")).toBe(true);
			resumed.session.dispose();
		} finally {
			registration.unregister();
		}
	});

	it("creates a persistent interactive pi runtime", async () => {
		const registration = registerFauxProvider({
			api: `faux-runtime-${Date.now()}`,
			models: [{ id: "runtime-model" }],
		});
		try {
			const runtime = await createAutoRAGPiInteractiveRuntime({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "interactive prompt",
				customTools: [],
				onInput: () => ({ kind: "handled" }),
			});
			expect(runtime.runtime.session.sessionFile).toBeTypeOf("string");
			await runtime.dispose();
		} finally {
			registration.unregister();
		}
	});

	it("lets pi choose the interactive model when AutoRAG has no configured model", async () => {
		const runtime = await createAutoRAGPiInteractiveRuntime({
			cwd: root,
			agentDir: join(root, "agent"),
			sessionDir: join(root, "sessions"),
			getSystemPrompt: () => "interactive prompt",
			customTools: [],
			onInput: () => ({ kind: "handled" }),
		});
		try {
			// Undefined options leave model discovery and selection to pi.
			expect(runtime.runtime.session.model).toBeDefined();
		} finally {
			await runtime.dispose();
		}
	});

	it("activates an extension tool only when its name is allow-listed", async () => {
		const registration = registerFauxProvider({
			api: `faux-extension-${Date.now()}`,
			models: [{ id: "pi-extension-test" }],
		});
		const extensionTool: ExtensionFactory = (pi) =>
			pi.registerTool({
				name: "extension_probe",
				label: "Extension probe",
				description: "Probe tool registered by a pi extension",
				parameters: Type.Object({}),
				async execute() {
					return { content: [{ type: "text", text: "probe" }], details: {} };
				},
			});
		try {
			const listed = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
				extensionFactories: [extensionTool],
				extensionToolNames: ["extension_probe"],
			});
			expect(listed.session.getActiveToolNames()).toContain("extension_probe");
			listed.session.dispose();

			// pi's `tools` allow-list drops extension tools that are not named.
			const unlisted = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
				extensionFactories: [extensionTool],
			});
			expect(unlisted.session.getActiveToolNames()).not.toContain("extension_probe");
			unlisted.session.dispose();
		} finally {
			registration.unregister();
		}
	});

	it("injects an AutoRAG update notice into the interactive session on startup", async () => {
		const registration = registerFauxProvider({
			api: `faux-update-${Date.now()}`,
			models: [{ id: "update-model" }],
		});
		try {
			const runtime = await createAutoRAGPiInteractiveRuntime({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model: registration.getModel(),
				getSystemPrompt: () => "interactive prompt",
				customTools: [],
				onInput: () => ({ kind: "handled" }),
				updateNotice: async () => "AutoRAG v9.9.9 is available (you have v1.0.0).",
			});
			try {
				// The TUI binds extension UI context on init; session_start (and thus
				// the notice) fires then, so mirror that here.
				await runtime.runtime.session.bindExtensions({});
				await vi.waitFor(() => {
					const notice = runtime.runtime.session.messages.find(
						(message) => "customType" in message && message.customType === "autorag.update",
					);
					expect(notice).toBeDefined();
				});
			} finally {
				await runtime.dispose();
			}
		} finally {
			registration.unregister();
		}
	});

	it("records a TUI answer once in the session, with no harness copy of it", async () => {
		// The bound pi session already records and renders the model's own
		// assistant message. Re-sending the answer as a custom message would show
		// it twice and put it in the model's context twice (as a user message).
		// The direct route ends in one plain answer, so no extra turn runs either.
		const tui = await tuiAgent([fauxAssistantMessage("Paris.", { stopReason: "stop" })]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await tui.agent.waitForIdle();
			// One model call: the direct route ends in a single plain answer.
			expect(tui.registration.state.callCount).toBe(1);
			const copies = tui.session.messages.filter((message) => message.role === "custom");
			expect(copies).toEqual([]);
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" ? [textOf(message.content)] : [],
			);
			expect(answers).toEqual(["Paris."]);
		} finally {
			await tui.dispose();
		}
	});

	it("steers a running TUI search with a message typed during a model turn", async () => {
		// Enter while the search streams is a correction of the running search, not
		// a new query: the model sees it in the same run, before its next request,
		// and the steered direct answer goes on to verification.
		const seen: string[] = [];
		const first = heldStep(
			fauxAssistantMessage([fauxToolCall("search_custom", { query: "capital of france" })], {
				stopReason: "toolUse",
			}),
		);
		const tui = await tuiAgent([
			first.held,
			seeingStep("Lyon is not the capital.", seen),
			seeingStep("Verified: Paris, not Lyon.", seen),
		]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await first.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive", streamingBehavior: "steer" });
			first.release();
			await tui.agent.waitForIdle();

			// The very next model request carried the steer, inside the same run.
			expect(seen[0]).toContain("Check Lyon too.");
			// Verification restates the correction and the run ends on its answer.
			expect(seen[1]).toContain("Check Lyon too.");
			expect(tui.registration.state.callCount).toBe(3);
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" ? [textOf(message.content)] : [],
			);
			expect(answers.at(-1)).toBe("Verified: Paris, not Lyon.");
			expect(tui.session.sessionName).toBe("What is the capital of France?");
			const userTexts = tui.session.messages.flatMap((message) =>
				message.role === "user" ? [textOf(message.content)] : [],
			);
			expect(userTexts.some((text) => text.includes("busy"))).toBe(false);
		} finally {
			await tui.dispose();
		}
	});

	it("delivers a message typed between search phases to the next phase's model request", async () => {
		// While Jev routes the query no model turn streams, so pi hands Enter to the
		// input handler as an idle prompt. It still belongs to the running search:
		// the fast phase's own request must carry it, not a new search.
		const seen: string[] = [];
		const routing = holdingJev(jevRoute("local", 0.9), QUERY_ROUTE_QUESTION_ID);
		const tui = await tuiAgent(
			[seeingStep("Fast answer: Paris.", seen), seeingStep("Verified: Paris, not Lyon.", seen)],
			routing.backend,
		);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await routing.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive" });
			routing.release();
			await tui.agent.waitForIdle();

			expect(seen[0]).toContain("Check Lyon too.");
			expect(tui.registration.state.callCount).toBe(2);
			expect(tui.session.sessionName).toBe("What is the capital of France?");
		} finally {
			await tui.dispose();
		}
	});

	it("verifies a fast answer that Jev would accept once the user steered the search", async () => {
		// A steer says the search is going wrong: even a fast answer Jev would
		// accept as final goes on to verification, which restates the correction.
		const seen: string[] = [];
		const fast = heldStep(fauxAssistantMessage("Fast answer: Lyon.", { stopReason: "stop" }));
		const tui = await tuiAgent(
			[fast.held, seeingStep("Corrected: Paris.", seen), seeingStep("Verified: Paris.", seen)],
			jevRoute("local", 0.05),
		);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await fast.entered;
			await tui.session.prompt("Lyon is wrong.", { source: "interactive", streamingBehavior: "steer" });
			fast.release();
			await tui.agent.waitForIdle();

			expect(tui.registration.state.callCount).toBe(3);
			// The verification prompt, not the steer itself, and it restates the steer.
			expect(seen[1]).toContain("Original query: What is the capital of France?");
			expect(seen[1]).toContain("Lyon is wrong.");
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" ? [textOf(message.content)] : [],
			);
			expect(answers.at(-1)).toBe("Verified: Paris.");
		} finally {
			await tui.dispose();
		}
	});

	it("runs a follow-up typed during a search as the next query after it answers", async () => {
		// Alt+Enter queues a follow-up: a new question, answered after this one.
		const first = heldStep(fauxAssistantMessage("Paris.", { stopReason: "stop" }));
		const tui = await tuiAgent([first.held, fauxAssistantMessage("Leonardo da Vinci.", { stopReason: "stop" })]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await first.entered;
			await tui.session.prompt("Who painted the Mona Lisa?", {
				source: "interactive",
				streamingBehavior: "followUp",
			});
			first.release();
			await tui.agent.waitForIdle();

			expect(tui.registration.state.callCount).toBe(2);
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" ? [textOf(message.content)] : [],
			);
			expect(answers).toEqual(["Paris.", "Leonardo da Vinci."]);
			expect(tui.session.sessionName).toBe("Who painted the Mona Lisa?");
		} finally {
			await tui.dispose();
		}
	});

	it("cuts a running retrieval tool short when the user steers, so the steer lands promptly", async () => {
		// A slow datasource search would otherwise hold the steer until it returns.
		// Like an interruptible wait in pi, the steer stops the search and the
		// model gets a skipped-tool result telling it to retry if still needed.
		const datasource = slowDatasource("kakao");
		const toolResults: string[] = [];
		const seen: string[] = [];
		const tui = await tuiAgent(
			[
				fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "capital of france" })], {
					stopReason: "toolUse",
				}),
				(context) => {
					for (const message of context.messages) {
						if (message.role === "toolResult") toolResults.push(textOf(message.content));
					}
					const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
					seen.push(lastUser === undefined ? "" : textOf(lastUser.content));
					return fauxAssistantMessage("Checked Lyon: not the capital.", { stopReason: "stop" });
				},
				fauxAssistantMessage("Verified: Paris.", { stopReason: "stop" }),
			],
			jevRoute("direct"),
			{ datasourceSkills: [datasource.skill] },
		);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await datasource.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive", streamingBehavior: "steer" });
			// No release: only the steer may end the retrieval.
			await tui.agent.waitForIdle();

			expect(datasource.aborted()).toBe(true);
			expect(toolResults.some((text) => text.includes("Skipped due to the user's steering message"))).toBe(true);
			expect(seen[0]).toContain("Check Lyon too.");
		} finally {
			datasource.release();
			await tui.dispose();
		}
	});

	it("answers a steer that arrives after verification's final request within the same search", async () => {
		// The steer lands after the last phase's model request settled and before
		// the search ends: one more verification turn answers it in this search
		// instead of it running later as an unrelated query.
		const seen: string[] = [];
		const tui = await tuiAgent(
			[
				fauxAssistantMessage("Fast answer: Paris.", { stopReason: "stop" }),
				fauxAssistantMessage("Verified: Paris.", { stopReason: "stop" }),
				seeingStep("Paris, about 2.1 million people.", seen),
			],
			jevRoute("local", 0.9),
		);
		try {
			// Typed the moment verification's turn settles (the second settle of the run).
			let settles = 0;
			const unsubscribe = tui.session.subscribe((event) => {
				if (event.type !== "agent_settled" || ++settles < 2) return;
				unsubscribe();
				void tui.session.prompt("Also give the population.", { source: "interactive" });
			});
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await tui.agent.waitForIdle();

			expect(tui.registration.state.callCount).toBe(3);
			expect(seen[0]).toContain("Also give the population.");
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" ? [textOf(message.content)] : [],
			);
			expect(answers.at(-1)).toBe("Paris, about 2.1 million people.");
			// Answered within the search it corrected, not as a new query.
			expect(tui.session.sessionName).toBe("What is the capital of France?");
		} finally {
			await tui.dispose();
		}
	});

	it("stops the whole search when the user interrupts it, with no later phase running", async () => {
		// Esc aborts the streaming turn. The search must end there instead of
		// moving on to its next phase as if the phase had finished.
		const fast = heldStep(fauxAssistantMessage("Fast answer: Paris.", { stopReason: "stop" }));
		const tui = await tuiAgent(
			[fast.held, fauxAssistantMessage("Verification that must not run.", { stopReason: "stop" })],
			jevRoute("local", 0.9),
		);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await fast.entered;
			const aborting = tui.session.abort();
			fast.release();
			await aborting;
			await tui.agent.waitForIdle();

			expect(tui.registration.state.callCount).toBe(1);
		} finally {
			await tui.dispose();
		}
	});

	it("registers the hosted autorag provider on the session model runtime", async () => {
		const registration = registerFauxProvider({ api: `faux-${Date.now()}`, models: [{ id: "pi-provider-test" }] });
		try {
			const session = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			const provider = session.session.modelRuntime.getProvider("autorag");
			expect(provider?.name).toBe("AutoRAG");
			expect(provider?.baseUrl).toMatch(/\/v1$/);
			expect(provider?.auth.oauth?.name).toBe("AutoRAG");
			session.session.dispose();
		} finally {
			registration.unregister();
		}
	});
});
