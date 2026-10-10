import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
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

	it("finishes a TUI query without feeding its own progress messages back into the model", async () => {
		// Progress/preliminary messages are display-only. Sent while the search
		// turn streams, pi's default steer delivery injects each one as a user
		// message, the model answers it, which emits more progress: the TUI
		// query never finishes (seen live on the Jev direct route).
		const registration = registerFauxProvider({ api: `faux-tui-${randomUUID()}`, models: [{ id: "tui-model" }] });
		registration.setResponses([
			fauxAssistantMessage([fauxToolCall("emit_fast_answer", { answer: "Paris.", results: [] })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			...Array.from({ length: 20 }, () =>
				fauxAssistantMessage("Replying to a progress note.", { stopReason: "stop" }),
			),
		]);
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
			// Two model calls: the direct fast answer and its tool-result turn.
			expect(registration.state.callCount).toBe(2);
			const complete = session.messages.filter(
				(message) => message.role === "custom" && message.customType === "autorag.complete",
			);
			expect(complete).toHaveLength(1);
		} finally {
			await hosted.dispose();
			registration.unregister();
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
