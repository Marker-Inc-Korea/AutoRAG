import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type FauxResponseStep, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { MockBackend } from "jev-use";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "../../src/agent/pi-session.ts";
import { DECOMPOSE_QUESTION_ID, QUERY_ROUTE_QUESTION_ID } from "../../src/agent/query-routing.ts";

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

/** A TUI-hosted agent whose Jev router sends every query down the direct route. */
async function directRouteTui(responses: FauxResponseStep[]) {
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
		jev: {
			backend: new MockBackend({
				[QUERY_ROUTE_QUESTION_ID]: {
					answer: "direct",
					distribution: { local: 0.1, web: 0.1, direct: 0.8 },
					confidence: 0.8,
				},
				[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
			}),
		},
	});
	const hosted = await agent.createPiInteractiveRuntime();
	return {
		registration,
		session: hosted.runtime.session,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
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
				onQuery: async () => undefined,
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
			onQuery: async () => undefined,
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
				onQuery: async () => undefined,
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
		const registration = registerFauxProvider({ api: `faux-tui-${randomUUID()}`, models: [{ id: "tui-model" }] });
		registration.setResponses([fauxAssistantMessage("Paris.", { stopReason: "stop" })]);
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
			jev: {
				backend: new MockBackend({
					[QUERY_ROUTE_QUESTION_ID]: {
						answer: "direct",
						distribution: { local: 0.1, web: 0.1, direct: 0.8 },
						confidence: 0.8,
					},
					[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
				}),
			},
		});
		const hosted = await agent.createPiInteractiveRuntime();
		try {
			const session = hosted.runtime.session;
			await session.prompt("What is the capital of France?", { source: "interactive" });
			// One model call: the direct route ends in a single plain answer.
			expect(registration.state.callCount).toBe(1);
			const copies = session.messages.filter((message) => message.role === "custom");
			expect(copies).toEqual([]);
			const answers = session.messages.flatMap((message) =>
				message.role === "assistant"
					? message.content.flatMap((block) => (block.type === "text" ? [block.text] : []))
					: [],
			);
			expect(answers).toEqual(["Paris."]);
		} finally {
			await hosted.dispose();
			registration.unregister();
		}
	});

	it("rejects a query typed while one is running without feeding the rejection to the running model", async () => {
		// The rejection notice must be display-only: delivered by default while the
		// first query's turn streams, pi steers it into that turn as a user message
		// and the model answers the harness's own error instead of the user.
		const first = heldStep(fauxAssistantMessage("Paris.", { stopReason: "stop" }));
		const tui = await directRouteTui([
			first.held,
			...Array.from({ length: 5 }, () =>
				fauxAssistantMessage("Replying to the busy notice.", { stopReason: "stop" }),
			),
		]);
		try {
			const running = tui.session.prompt("What is the capital of France?", { source: "interactive" });
			await first.entered;
			// What pi's TUI does with Enter while the session streams.
			await tui.session.prompt("Who painted the Mona Lisa?", { source: "interactive", streamingBehavior: "steer" });
			first.release();
			await running;
			await tui.session.waitForIdle();

			expect(tui.registration.state.callCount).toBe(1);
			// The rejected query never named the session after itself.
			expect(tui.session.sessionName).toBe("What is the capital of France?");
			const userTexts = tui.session.messages.flatMap((message) =>
				message.role === "user"
					? (typeof message.content === "string" ? [message.content] : message.content).flatMap((block) =>
							typeof block === "string" ? [block] : block.type === "text" ? [block.text] : [],
						)
					: [],
			);
			expect(userTexts.some((text) => text.includes("busy"))).toBe(false);
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
