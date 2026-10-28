import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionFactory, ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { type JevBackend, MockBackend } from "jev-use";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { CITATION_QUESTION_ID_PREFIX } from "../../src/agent/citation-relevance.ts";
import {
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "../../src/agent/pi-session.ts";
import {
	DECOMPOSE_QUESTION_ID,
	datasourceQuestionId,
	FILE_SEARCH_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
	WEB_DATASOURCE_ID,
} from "../../src/agent/query-routing.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";
import { citationsRelevant } from "../helpers/citations-relevant.ts";

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
	return citationsRelevant(
		new MockBackend({
			[QUERY_ROUTE_QUESTION_ID]: {
				answer: route,
				distribution: { local: 0.1, organize: 0.1, extra: 0.1, [route]: 0.8 },
				confidence: 0.8,
			},
			[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
			[FOLLOW_UP_QUESTION_ID]: { answer: followUp },
			// The verification phase also asks whether a file-name search helps; these runs must not run one.
			[FILE_SEARCH_QUESTION_ID]: { answer: 0 },
		}),
	);
}

/** A TUI-hosted agent on the faux model, routed by `jev` (the extra route by default). */
async function tuiAgent(
	responses: FauxResponseStep[],
	jev: JevBackend = jevRoute("extra"),
	extra: Pick<AutoRAGAgentOptions, "datasourceSkills"> = {},
	ui: Partial<ExtensionUIContext> = {},
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
	// What InteractiveMode does on init: bind a UI whose terminal-key listeners the test can drive.
	const keyListeners = new Set<(data: string) => { consume?: boolean } | undefined>();
	await hosted.runtime.session.bindExtensions({
		uiContext: {
			...noUi(),
			...ui,
			onTerminalInput: (handler) => {
				keyListeners.add(handler);
				return () => keyListeners.delete(handler);
			},
		},
	});
	return {
		agent,
		registration,
		session: hosted.runtime.session,
		runtime: hosted.runtime,
		/** Feeds one terminal key sequence through the bound listeners, as the TUI does before the editor sees it. */
		pressKey: (data: string): boolean => [...keyListeners].some((listener) => listener(data)?.consume === true),
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

/** The interactive UI surface the TUI binds, with no visible effect. */
function noUi(): ExtensionUIContext {
	return {
		select: async () => undefined,
		confirm: async () => false,
		input: async () => undefined,
		notify: () => undefined,
		onTerminalInput: () => () => undefined,
		setStatus: () => undefined,
		setWorkingMessage: () => undefined,
		setWorkingVisible: () => undefined,
		setWorkingIndicator: () => undefined,
		setHiddenThinkingLabel: () => undefined,
		setWidget: () => undefined,
		setFooter: () => undefined,
		setHeader: () => undefined,
		setTitle: () => undefined,
		custom: async () => undefined as never,
		pasteToEditor: () => undefined,
		setEditorText: () => undefined,
		getEditorText: () => "",
		editor: async () => undefined,
		addAutocompleteProvider: () => undefined,
		setEditorComponent: () => undefined,
		getEditorComponent: () => undefined,
		theme: undefined as never,
		getAllThemes: () => [],
		getTheme: () => undefined,
		setTheme: () => ({ success: false, error: "UI not available" }),
		getToolsExpanded: () => false,
		setToolsExpanded: () => undefined,
	} as ExtensionUIContext;
}

/** Wraps a Jev backend so the `nth` (1-based) batch asking `questionId` stays in flight until released. */
function holdingJev(inner: JevBackend, questionId: string, nth = 1) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	let seenCalls = 0;
	const backend: JevBackend = {
		name: inner.name,
		async judge(request) {
			if (request.questions.some((question) => question.id === questionId) && ++seenCalls === nth) {
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

/** A datasource skill whose only retrieval method is `method`. */
function datasourceSkill(datasourceId: string, method: RetrievalMethod): DatasourceSkill {
	return {
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
}

/** A datasource whose n-th search returns the n-th scripted batch of chunks (the last batch repeats). */
function scriptedDatasource(datasourceId: string, batches: readonly (readonly RetrievalChunk[])[]) {
	let searches = 0;
	const method: RetrievalMethod = {
		describe: () => ({
			name: `${datasourceId}.keyword`,
			type: "bm25",
			description: `${datasourceId} test method`,
			status: "active",
			capabilities: ["keyword"],
			datasourceId,
		}),
		async retrieve(): Promise<RetrievalResult[]> {
			const batch = batches[Math.min(searches++, batches.length - 1)] ?? [];
			return batch.map((chunk, index) => ({
				id: `${datasourceId}:${searches}:${index}`,
				source: chunk.source,
				content: chunk.content,
				score: 1,
				metadata: {},
			}));
		},
	};
	return datasourceSkill(datasourceId, method);
}

interface RetrievalChunk {
	readonly source: string;
	readonly content: string;
}

/** The user message the model is about to answer, exactly as it reached the model. */
function lastUserText(messages: readonly { role: string; content: unknown }[]): string {
	const last = [...messages].reverse().find((message) => message.role === "user");
	return last === undefined ? "" : textOf(last.content as string | readonly { type: string; text?: string }[]);
}

/**
 * A faux step answering `answer(ids)`, where `ids` are the evidence ids the tool results in the
 * transcript printed so far; the user message it answers is recorded in `seen`.
 */
function citingStep(answer: (ids: readonly string[]) => string, seen: string[] = []): FauxResponseStep {
	return (context) => {
		seen.push(lastUserText(context.messages));
		const ids = context.messages.flatMap((message) =>
			message.role === "toolResult"
				? [...textOf(message.content).matchAll(/\[(e\d+)\]/gu)].map((match) => match[1] as string)
				: [],
		);
		return fauxAssistantMessage(answer(ids), { stopReason: "stop" });
	};
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
	const skill = datasourceSkill(datasourceId, method);
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

	it("starts the interactive session on the configured model, not pi's own pick", async () => {
		const registration = registerFauxProvider({
			api: `faux-configured-${Date.now()}`,
			models: [{ id: "configured-model" }],
		});
		try {
			const model = registration.getModel();
			const runtime = await createAutoRAGPiInteractiveRuntime({
				cwd: root,
				agentDir: join(root, "agent"),
				sessionDir: join(root, "sessions"),
				model,
				getSystemPrompt: () => "interactive prompt",
				customTools: [],
				onInput: () => ({ kind: "handled" }),
			});
			try {
				expect(runtime.runtime.session.model?.provider).toBe(model.provider);
				expect(runtime.runtime.session.model?.id).toBe("configured-model");
			} finally {
				await runtime.dispose();
			}
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
		// The extra route ends in one plain answer, so no extra turn runs either.
		const tui = await tuiAgent([fauxAssistantMessage("Paris.", { stopReason: "stop" })]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await tui.agent.waitForIdle();
			// One model call: the extra route ends in a single plain answer.
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
		// a new query: the model sees it in the same run, before its next request.
		// The extra branch has no verification phase, so that next request's answer
		// is the run's final answer to the correction.
		const seen: string[] = [];
		const first = heldStep(
			fauxAssistantMessage([fauxToolCall("search_custom", { query: "capital of france" })], {
				stopReason: "toolUse",
			}),
		);
		const tui = await tuiAgent([first.held, seeingStep("Lyon is not the capital. Paris is.", seen)]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await first.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive", streamingBehavior: "steer" });
			first.release();
			await tui.agent.waitForIdle();

			// The very next model request carried the steer, inside the same run.
			expect(seen[0]).toContain("Check Lyon too.");
			// No verification phase followed: the steer's own turn answered it.
			expect(tui.registration.state.callCount).toBe(2);
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" && message.stopReason === "stop" ? [textOf(message.content)] : [],
			);
			expect(answers).toEqual(["Lyon is not the capital. Paris is."]);
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
			jevRoute("extra"),
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

	it("cuts the baseline prefetch short when the user steers before the first model request", async () => {
		// Jev selects the datasource, so the harness searches it before the fast
		// phase. A slow datasource there held the steer until it returned: the
		// steer now stops the prefetch and the fast phase starts with the steer.
		const datasource = slowDatasource("kakao");
		const seen: string[] = [];
		const routing = new MockBackend({
			[QUERY_ROUTE_QUESTION_ID]: {
				answer: "local",
				distribution: { local: 0.8, organize: 0.1, extra: 0.1 },
				confidence: 0.8,
			},
			[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
			[FOLLOW_UP_QUESTION_ID]: { answer: 0.9 },
			// The verification phase also asks whether a file-name search helps; these runs must not run one.
			[FILE_SEARCH_QUESTION_ID]: { answer: 0 },
			[datasourceQuestionId("kakao")]: { answer: 0.9 },
			// The gate also asks about the public web; this run must not search it.
			[datasourceQuestionId(WEB_DATASOURCE_ID)]: { answer: 0 },
		});
		const tui = await tuiAgent(
			[seeingStep("Fast answer: Paris.", seen), seeingStep("Verified: Paris.", seen)],
			routing,
			{ datasourceSkills: [datasource.skill] },
		);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await datasource.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive" });
			// No release: only the steer may end the prefetch.
			await tui.agent.waitForIdle();

			expect(datasource.aborted()).toBe(true);
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

	it("steers a follow-up turn like the first query: the steer reaches the model inside the same turn", async () => {
		const seen: string[] = [];
		const held = heldStep(
			fauxAssistantMessage([fauxToolCall("search_custom", { query: "mona lisa" })], { stopReason: "toolUse" }),
		);
		const tui = await tuiAgent([
			fauxAssistantMessage("FIRST QUERY ANSWER.", { stopReason: "stop" }),
			held.held,
			seeingStep("Leonardo.", seen),
		]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await tui.agent.waitForIdle();
			await tui.session.prompt("Who painted the Mona Lisa?", { source: "interactive" });
			await held.entered;
			await tui.session.prompt("Answer in one name.", { source: "interactive", streamingBehavior: "steer" });
			held.release();
			await tui.agent.waitForIdle();

			expect(seen[0]).toContain("Answer in one name.");
			const answers = tui.session.messages.flatMap((message) =>
				message.role === "assistant" && message.stopReason === "stop" ? [textOf(message.content)] : [],
			);
			expect(answers).toEqual(["FIRST QUERY ANSWER.", "Leonardo."]);
		} finally {
			await tui.dispose();
		}
	});

	it("keeps a slash-prefixed message typed during a search inside the search", async () => {
		// pi's input loop runs while the search is detached. A "/..." line that is
		// no command used to start pi's own prompt on the bound session between
		// phases, outside the harness, and the next phase then failed as busy.
		const seen: string[] = [];
		const routing = holdingJev(jevRoute("local", 0.9), QUERY_ROUTE_QUESTION_ID);
		const tui = await tuiAgent([seeingStep("Fast answer.", seen), seeingStep("Verified.", seen)], routing.backend);
		try {
			await tui.session.prompt("What does my hosts file say?", { source: "interactive" });
			await routing.entered;
			await tui.session.prompt("/etc/hosts is the file I mean.", { source: "interactive" });
			routing.release();
			await tui.agent.waitForIdle();

			const errors = tui.session.messages.filter(
				(message) => message.role === "custom" && message.customType === "autorag.error",
			);
			expect(errors).toEqual([]);
			expect(seen[0]).toContain("/etc/hosts is the file I mean.");
			expect(tui.registration.state.callCount).toBe(2);
		} finally {
			await tui.dispose();
		}
	});

	it("queues Alt+Enter between phases as the next question, not a steer", async () => {
		// With no turn streaming, pi submits Alt+Enter like Enter. AutoRAG reads
		// the key itself, so it still queues a new question rather than a correction.
		const seen: string[] = [];
		const routing = holdingJev(jevRoute("extra"), QUERY_ROUTE_QUESTION_ID);
		const tui = await tuiAgent([seeingStep("Paris.", seen), seeingStep("Leonardo da Vinci.", seen)], routing.backend);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await routing.entered;
			// The follow-up key (Alt+Enter; Ctrl+Q where pi defaults to Windows bindings)
			// reaches AutoRAG's listener, then pi submits the editor text.
			const windowsBindings =
				process.platform === "win32" ||
				(process.platform === "linux" && Boolean(process.env.WSL_DISTRO_NAME || process.env.WSL_INTEROP));
			tui.pressKey(windowsBindings ? "\x11" : "\x1b\r");
			await tui.session.prompt("Who painted the Mona Lisa?", { source: "interactive" });
			routing.release();
			await tui.agent.waitForIdle();

			expect(seen).toHaveLength(2);
			expect(seen[0]).not.toContain("Mona Lisa");
			expect(seen[1]).toContain("Who painted the Mona Lisa?");
			expect(seen[1]).not.toContain("interrupted the running search");
			expect(tui.session.sessionName).toBe("Who painted the Mona Lisa?");
		} finally {
			await tui.dispose();
		}
	});

	it("stops the search on Esc between phases, before its next phase runs", async () => {
		// pi's Esc only aborts a streaming turn. Between phases AutoRAG reads
		// the key itself and ends the search.
		const routing = holdingJev(jevRoute("local", 0.9), QUERY_ROUTE_QUESTION_ID);
		const tui = await tuiAgent([fauxAssistantMessage("Must not run.", { stopReason: "stop" })], routing.backend);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await routing.entered;
			expect(tui.pressKey("\x1b")).toBe(true);
			routing.release();
			await tui.agent.waitForIdle();

			expect(tui.registration.state.callCount).toBe(0);
		} finally {
			await tui.dispose();
		}
	});

	it("lists and restores a steer as the user's own words, and keeps queued steers on dequeue", async () => {
		// pi shows queued steering, puts it back in the editor on Esc, and Alt+Up
		// clears both queues. It must see what the user typed, not AutoRAG's framing,
		// while the model still receives the framed correction.
		const seen: string[] = [];
		const first = heldStep(
			fauxAssistantMessage([fauxToolCall("search_custom", { query: "capital" })], { stopReason: "toolUse" }),
		);
		const tui = await tuiAgent([first.held, seeingStep("Checked Lyon.", seen), seeingStep("Verified.", seen)]);
		try {
			await tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await first.entered;
			await tui.session.prompt("Check Lyon too.", { source: "interactive", streamingBehavior: "steer" });
			expect(tui.session.getSteeringMessages()).toEqual(["Check Lyon too."]);
			first.release();
			await tui.agent.waitForIdle();

			expect(seen[0]).toContain("The user interrupted the running search with this correction");
			expect(seen[0]).toContain("Check Lyon too.");
			expect(tui.session.getSteeringMessages()).toEqual([]);
		} finally {
			await tui.dispose();
		}
	});

	it("registers the hosted dazzi provider on the session model runtime", async () => {
		const registration = registerFauxProvider({ api: `faux-${Date.now()}`, models: [{ id: "pi-provider-test" }] });
		try {
			const session = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				model: registration.getModel(),
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			const provider = session.session.modelRuntime.getProvider("dazzi");
			expect(provider?.name).toBe("Dazzi Models");
			expect(provider?.baseUrl).toMatch(/\/v1$/);
			expect(provider?.auth.oauth?.name).toBe("Dazzi Models");
			session.session.dispose();
		} finally {
			registration.unregister();
		}
	});

	it("keeps the provider's own name when an embedded model is registered under it", async () => {
		const registration = registerFauxProvider({ api: `faux-${Date.now()}`, models: [{ id: "pi-provider-test" }] });
		try {
			const model = { ...registration.getModel(), provider: "dazzi", id: "not-in-catalog", name: "Some Model Name" };
			const session = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				model,
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [],
			});
			const provider = session.session.modelRuntime.getProvider("dazzi");
			expect(provider?.name).toBe("Dazzi Models");
			expect(provider?.auth.oauth).toBeDefined();
			session.session.dispose();
		} finally {
			registration.unregister();
		}
	});

	describe("follow-up turns in one conversation", () => {
		/** Counts Jev judge calls so a test can see whether a turn went through the gate. */
		function countingJev(inner: JevBackend): { backend: JevBackend; calls: () => number } {
			let calls = 0;
			return {
				backend: {
					name: inner.name,
					judge(request) {
						calls += 1;
						return inner.judge(request);
					},
				},
				calls: () => calls,
			};
		}

		it("answers the second query in the same session without the Jev gate or a second search pipeline", async () => {
			const seen: string[] = [];
			const jev = countingJev(jevRoute("extra"));
			const tui = await tuiAgent(
				[seeingStep("Paris.", seen), seeingStep("About 2.1 million people.", seen)],
				jev.backend,
			);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await tui.agent.waitForIdle();
				const afterFirst = jev.calls();
				expect(afterFirst).toBeGreaterThan(0);

				await tui.session.prompt("How many people live there?", { source: "interactive" });
				await tui.agent.waitForIdle();

				// No Jev routing, decomposition, datasource or follow-up check ran for turn two.
				expect(jev.calls()).toBe(afterFirst);
				// One model call: no baseline prefetch, fast answer, or verification pass.
				expect(tui.registration.state.callCount).toBe(2);
				// The follow-up reaches the model exactly as typed: no envelope around it.
				expect(seen[1]).toBe("How many people live there?");
				expect(seen[1]).not.toContain("Baseline retrieval evidence");
			} finally {
				await tui.dispose();
			}
		});

		it("lets the model search by tool call on a follow-up", async () => {
			const seen: string[] = [];
			const jev = countingJev(jevRoute("extra"));
			const tui = await tuiAgent(
				[
					seeingStep("Paris.", seen),
					fauxAssistantMessage([fauxToolCall("search_custom", { query: "population of paris" })], {
						stopReason: "toolUse",
					}),
					seeingStep("About 2.1 million.", seen),
				],
				jev.backend,
			);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await tui.agent.waitForIdle();
				const afterFirst = jev.calls();

				await tui.session.prompt("How many people live there?", { source: "interactive" });
				await tui.agent.waitForIdle();

				expect(jev.calls()).toBe(afterFirst);
				expect(tui.registration.state.callCount).toBe(3);
				expect(tui.session.messages.some((message) => message.role === "toolResult")).toBe(true);
				const answers = tui.session.messages.flatMap((message) =>
					message.role === "assistant" && message.stopReason === "stop" ? [textOf(message.content)] : [],
				);
				expect(answers).toEqual(["Paris.", "About 2.1 million."]);
			} finally {
				await tui.dispose();
			}
		});

		it("resolves evidence ids from any earlier turn, however many searches came in between", async () => {
			const seen: string[] = [];
			const kakao = scriptedDatasource("kakao", [
				[{ source: "/kakao/room/a.txt", content: "alpha fact" }],
				[{ source: "/kakao/room/b.txt", content: "beta fact" }],
			]);
			const search = fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
				stopReason: "toolUse",
			});
			const tui = await tuiAgent(
				[
					search,
					citingStep((ids) => `Alpha [${ids[0]}].`, seen),
					search,
					citingStep((ids) => `Beta [${ids[1]}].`, seen),
					citingStep((ids) => `Alpha [${ids[0]}] and beta [${ids[1]}].`, seen),
				],
				jevRoute("extra"),
				{ datasourceSkills: [kakao] },
			);
			try {
				await tui.agent.searchDocuments("what is alpha?");
				await tui.agent.searchDocuments("what is beta?");
				const third = await tui.agent.searchDocuments("how do they relate?");

				expect(seen.at(-1)).toBe("how do they relate?");
				expect(third.results.map((result) => result.title)).toEqual(["a.txt", "b.txt"]);
				expect(third.diagnostics?.some((diagnostic) => diagnostic.code === "citation-without-result")).toBe(false);
			} finally {
				await tui.dispose();
			}
		});

		it("keeps earlier ids resolvable and never reissues one when the conversation is resumed in a new process", async () => {
			const first = await tuiAgent(
				[
					fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
						stopReason: "toolUse",
					}),
					citingStep((ids) => `Alpha [${ids[0]}].`),
				],
				jevRoute("extra"),
				{
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/a.txt", content: "alpha fact" }]]),
					],
				},
			);
			let sessionFile: string | undefined;
			try {
				const answered = await first.agent.searchDocuments("what is alpha?");
				expect(answered.results.map((result) => result.title)).toEqual(["a.txt"]);
				sessionFile = first.session.sessionFile;
			} finally {
				await first.dispose();
			}
			if (sessionFile === undefined) throw new Error("expected a persisted pi session");

			// A fresh agent, as after a restart: its ledger starts empty and counts from e1.
			const resumed = await tuiAgent(
				[
					fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
						stopReason: "toolUse",
					}),
					citingStep((ids) => `Old [${ids[0]}], new [${ids[1]}].`),
				],
				jevRoute("extra"),
				{
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/b.txt", content: "beta fact" }]]),
					],
				},
			);
			try {
				await resumed.runtime.switchSession(sessionFile);
				// What InteractiveMode does after switching sessions: rebind, which starts the new session.
				await resumed.runtime.session.bindExtensions({ uiContext: noUi() });

				const response = await resumed.agent.searchDocuments("and what about beta?");

				expect(response.results.map((result) => result.title)).toEqual(["a.txt", "b.txt"]);
			} finally {
				await resumed.dispose();
			}
		});

		it("keeps a source's [n] across turns and numbers a new source with the next unused number", async () => {
			const kakao = scriptedDatasource("kakao", [
				[{ source: "/kakao/room/a.txt", content: "alpha fact" }],
				[{ source: "/kakao/room/b.txt", content: "beta fact" }],
			]);
			const search = fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
				stopReason: "toolUse",
			});
			const tui = await tuiAgent(
				[
					search,
					citingStep((ids) => `Alpha [${ids[0]}].`),
					search,
					citingStep((ids) => `Beta [${ids[1]}].`),
					citingStep((ids) => `Beta [${ids[1]}] and alpha [${ids[0]}].`),
				],
				jevRoute("extra"),
				{ datasourceSkills: [kakao] },
			);
			try {
				const first = await tui.agent.searchDocuments("what is alpha?");
				const second = await tui.agent.searchDocuments("what is beta?");
				const third = await tui.agent.searchDocuments("how do they relate?");

				expect(first.answer).toBe("Alpha [1].");
				expect(second.answer).toBe("Beta [2].");
				expect(second.results.map((result) => [result.number, result.title])).toEqual([[2, "b.txt"]]);
				expect(third.answer).toBe("Beta [2] and alpha [1].");
				expect(third.results.map((result) => [result.number, result.title])).toEqual([
					[2, "b.txt"],
					[1, "a.txt"],
				]);
			} finally {
				await tui.dispose();
			}
		});

		describe("what Jev reads as the user's messages when it judges citations", () => {
			/** Records the state of every citation-relevance batch and keeps every chunk. */
			function citationStates(route: JevBackend = jevRoute("extra")): { backend: JevBackend; states: string[] } {
				const states: string[] = [];
				const inner = route;
				const backend: JevBackend = {
					name: inner.name,
					async judge(request) {
						if (request.questions.some((question) => question.id?.startsWith(CITATION_QUESTION_ID_PREFIX))) {
							states.push(String(request.state));
						}
						return inner.judge(request);
					},
				};
				return { backend, states };
			}

			const search = fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
				stopReason: "toolUse",
			});

			it("holds every question of the conversation, as typed, oldest first", async () => {
				const { backend, states } = citationStates();
				const tui = await tuiAgent(
					[search, citingStep((ids) => `Alpha [${ids[0]}].`), search, citingStep((ids) => `Beta [${ids[1]}].`)],
					backend,
					{
						datasourceSkills: [
							scriptedDatasource("kakao", [
								[{ source: "/kakao/room/a.txt", content: "alpha fact" }],
								[{ source: "/kakao/room/b.txt", content: "beta fact" }],
							]),
						],
					},
				);
				try {
					await tui.agent.searchDocuments("what is alpha?");
					await tui.agent.searchDocuments("and what about beta?");

					expect(states).toHaveLength(2);
					expect(states[0]).toContain(JSON.stringify("what is alpha?"));
					expect(states[0]).not.toContain("and what about beta?");
					const second = states[1] ?? "";
					const alpha = second.indexOf(JSON.stringify("what is alpha?"));
					expect(alpha).toBeGreaterThan(-1);
					expect(second.indexOf(JSON.stringify("and what about beta?"))).toBeGreaterThan(alpha);
					// The harness prompt that wraps a question never stands in for what the user typed.
					expect(second).not.toContain("Baseline retrieval evidence");
					expect(second).not.toContain("Answer this original query");
				} finally {
					await tui.dispose();
				}
			});

			it("includes a steer the user typed while the search ran", async () => {
				const { backend, states } = citationStates(jevRoute("local", 0.05));
				const fast = heldStep(fauxAssistantMessage("Fast answer: Lyon.", { stopReason: "stop" }));
				const tui = await tuiAgent(
					[
						fast.held,
						// The steer lands after the fast turn: pi answers it within the fast phase, then verification runs.
						fauxAssistantMessage("Noted: checking Paris.", { stopReason: "stop" }),
						search,
						citingStep((ids) => `Paris [${ids[0]}].`),
					],
					backend,
					{
						datasourceSkills: [
							scriptedDatasource("kakao", [[{ source: "/kakao/room/a.txt", content: "paris fact" }]]),
						],
					},
				);
				try {
					const run = tui.agent.searchDocuments("capital of france?");
					await fast.entered;
					expect(tui.agent.steer("Lyon is wrong, check Paris.")).toBe(true);
					fast.release();
					await run;

					const last = states.at(-1) ?? "";
					expect(last).toContain(JSON.stringify("capital of france?"));
					expect(last).toContain(JSON.stringify("Lyon is wrong, check Paris."));
				} finally {
					await tui.dispose();
				}
			});

			it("keeps the earlier questions when the conversation is resumed in a new process", async () => {
				const first = await tuiAgent([search, citingStep((ids) => `Alpha [${ids[0]}].`)], jevRoute("extra"), {
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/a.txt", content: "alpha fact" }]]),
					],
				});
				let sessionFile: string | undefined;
				try {
					await first.agent.searchDocuments("what is alpha?");
					sessionFile = first.session.sessionFile;
				} finally {
					await first.dispose();
				}
				if (sessionFile === undefined) throw new Error("expected a persisted pi session");

				const { backend, states } = citationStates();
				const resumed = await tuiAgent([search, citingStep((ids) => `Beta [${ids[1]}].`)], backend, {
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/b.txt", content: "beta fact" }]]),
					],
				});
				try {
					await resumed.runtime.switchSession(sessionFile);
					await resumed.runtime.session.bindExtensions({ uiContext: noUi() });
					await resumed.agent.searchDocuments("and what about beta?");

					const state = states.at(-1) ?? "";
					expect(state).toContain(JSON.stringify("what is alpha?"));
					expect(state).toContain(JSON.stringify("and what about beta?"));
				} finally {
					await resumed.dispose();
				}
			});
		});

		it("keeps result numbers when the conversation is resumed in a new process", async () => {
			const first = await tuiAgent(
				[
					fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
						stopReason: "toolUse",
					}),
					citingStep((ids) => `Alpha [${ids[0]}].`),
				],
				jevRoute("extra"),
				{
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/a.txt", content: "alpha fact" }]]),
					],
				},
			);
			let sessionFile: string | undefined;
			try {
				const answered = await first.agent.searchDocuments("what is alpha?");
				expect(answered.answer).toBe("Alpha [1].");
				sessionFile = first.session.sessionFile;
			} finally {
				await first.dispose();
			}
			if (sessionFile === undefined) throw new Error("expected a persisted pi session");

			const resumed = await tuiAgent(
				[
					fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "q" })], {
						stopReason: "toolUse",
					}),
					citingStep((ids) => `New [${ids[1]}], old [${ids[0]}].`),
				],
				jevRoute("extra"),
				{
					datasourceSkills: [
						scriptedDatasource("kakao", [[{ source: "/kakao/room/b.txt", content: "beta fact" }]]),
					],
				},
			);
			try {
				await resumed.runtime.switchSession(sessionFile);
				await resumed.runtime.session.bindExtensions({ uiContext: noUi() });

				const response = await resumed.agent.searchDocuments("and what about beta?");

				expect(response.answer).toBe("New [2], old [1].");
				expect(response.results.map((result) => [result.number, result.title])).toEqual([
					[2, "b.txt"],
					[1, "a.txt"],
				]);
			} finally {
				await resumed.dispose();
			}
		});

		it("numbers the quick answer and the verification answer of one search from one sequence", async () => {
			const alpha = join(root, "alpha.txt");
			const beta = join(root, "beta.txt");
			writeFileSync(alpha, "alpha fact");
			writeFileSync(beta, "beta fact");
			const tui = await tuiAgent(
				[
					fauxAssistantMessage(`Beta first [file:${beta}]. Then alpha [file:${alpha}].`, { stopReason: "stop" }),
					fauxAssistantMessage(`Alpha first [file:${alpha}]. Then beta [file:${beta}].`, { stopReason: "stop" }),
				],
				jevRoute("local", 0.9),
			);
			try {
				const preliminaries: string[] = [];
				let finalAnswer = "";
				for await (const event of tui.agent.searchDocumentsStream("what are alpha and beta?")) {
					if (event.type === "preliminary") preliminaries.push(event.response.answer);
					if (event.type === "complete") finalAnswer = event.response.answer;
				}

				expect(preliminaries).toEqual(["Beta first [1]. Then alpha [2]."]);
				// The verification answer cites the same sources in the other order: each keeps the number the quick answer showed.
				expect(finalAnswer).toBe("Alpha first [2]. Then beta [1].");
			} finally {
				await tui.dispose();
			}
		});
	});

	describe("search progress widget", () => {
		const PROGRESS_KEY = "autorag-search-progress";
		interface WidgetComponent {
			render(width: number): string[];
			dispose?(): void;
		}
		type WidgetFactory = (
			tui: { requestRender(): void },
			theme: { fg(color: string, text: string): string },
		) => WidgetComponent;

		/** Records pi widget calls; `lines()` renders whatever progress widget is currently shown. */
		function widgetRecorder() {
			const calls: { key: string; shown: boolean }[] = [];
			let current: WidgetComponent | undefined;
			let disposed = 0;
			const setWidget = (key: string, content: unknown): void => {
				current?.dispose?.();
				if (current !== undefined) disposed += 1;
				current = undefined;
				calls.push({ key, shown: content !== undefined });
				if (typeof content === "function") {
					current = (content as WidgetFactory)({ requestRender: () => undefined }, { fg: (_c, text) => text });
				}
			};
			return {
				ui: { setWidget } as unknown as Partial<ExtensionUIContext>,
				calls,
				lines: (): string[] => current?.render(100) ?? [],
				disposed: () => disposed,
			};
		}

		function localRouting() {
			return new MockBackend({
				[QUERY_ROUTE_QUESTION_ID]: {
					answer: "local",
					distribution: { local: 0.8, organize: 0.1, extra: 0.1 },
					confidence: 0.8,
				},
				[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
				[FOLLOW_UP_QUESTION_ID]: { answer: 0.1 },
				// The verification phase also asks whether a file-name search helps; these runs must not run one.
				[FILE_SEARCH_QUESTION_ID]: { answer: 0 },
				[datasourceQuestionId("kakao")]: { answer: 0.9 },
				// The gate also asks about the public web; these runs must not search it.
				[datasourceQuestionId(WEB_DATASOURCE_ID)]: { answer: 0 },
			});
		}

		it("shows a live progress widget the moment a question is submitted, before any retrieval returns", async () => {
			const routing = holdingJev(localRouting(), QUERY_ROUTE_QUESTION_ID);
			const widgets = widgetRecorder();
			const tui = await tuiAgent(
				[fauxAssistantMessage("Fast answer.", { stopReason: "stop" })],
				routing.backend,
				{},
				widgets.ui,
			);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await routing.entered;

				expect(widgets.calls).toEqual([{ key: PROGRESS_KEY, shown: true }]);
				const lines = widgets.lines();
				expect(lines[0]).toContain("Searching");
				expect(lines.join("\n")).toContain("Routing the question");
				routing.release();
				await tui.agent.waitForIdle();
			} finally {
				await tui.dispose();
			}
		});

		it("names the source the baseline prefetch is still waiting on, and removes the widget when the search ends", async () => {
			const datasource = slowDatasource("kakao");
			const widgets = widgetRecorder();
			const tui = await tuiAgent(
				[
					fauxAssistantMessage("Fast answer.", { stopReason: "stop" }),
					fauxAssistantMessage("Verified.", { stopReason: "stop" }),
				],
				localRouting(),
				{ datasourceSkills: [datasource.skill] },
				widgets.ui,
			);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await datasource.entered;

				const waiting = widgets.lines().join("\n");
				expect(waiting).toMatch(/✓ Routing the question — local/u);
				expect(waiting).toMatch(/Datasource kakao · \d+s/u);
				expect(waiting).toContain("MinSync");

				datasource.release();
				await tui.agent.waitForIdle();

				expect(widgets.calls.at(-1)).toEqual({ key: PROGRESS_KEY, shown: false });
				expect(widgets.lines()).toEqual([]);
				expect(widgets.disposed()).toBe(1);
			} finally {
				datasource.release();
				await tui.dispose();
			}
		});

		it("removes the widget when the user stops the search with Esc", async () => {
			const routing = holdingJev(localRouting(), QUERY_ROUTE_QUESTION_ID);
			const widgets = widgetRecorder();
			const tui = await tuiAgent([], routing.backend, {}, widgets.ui);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await routing.entered;
				expect(tui.pressKey("\x1b")).toBe(true);
				routing.release();
				await tui.agent.waitForIdle();

				expect(widgets.calls.at(-1)).toEqual({ key: PROGRESS_KEY, shown: false });
			} finally {
				await tui.dispose();
			}
		});

		it("shows the progress of each follow-up question queued behind the first", async () => {
			const first = heldStep(fauxAssistantMessage("First answer.", { stopReason: "stop" }));
			const widgets = widgetRecorder();
			const tui = await tuiAgent(
				[first.held, fauxAssistantMessage("Second answer.", { stopReason: "stop" })],
				jevRoute("extra"),
				{},
				widgets.ui,
			);
			try {
				await tui.session.prompt("What is the capital of France?", { source: "interactive" });
				await first.entered;
				await tui.session.prompt("Who painted the Mona Lisa?", {
					source: "interactive",
					streamingBehavior: "followUp",
				});
				first.release();
				await tui.agent.waitForIdle();

				// One widget per question: shown, removed, shown, removed.
				expect(widgets.calls.map((call) => call.shown)).toEqual([true, false, true, false]);
			} finally {
				await tui.dispose();
			}
		});
	});
});
