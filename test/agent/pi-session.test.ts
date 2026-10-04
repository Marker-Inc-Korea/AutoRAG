import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
	createAutoRAGPiInteractiveRuntime,
	createAutoRAGPiSession,
	PI_BUILTIN_TOOL_NAMES,
} from "../../src/agent/pi-session.ts";

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
});
