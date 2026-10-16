import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { type AssistantMessage, fauxAssistantMessage, fauxToolCall, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createAutoRAGPiSession } from "../../src/agent/pi-session.ts";

function userTexts(messages: readonly Message[]): string[] {
	return messages
		.filter((message) => message.role === "user")
		.map((message) =>
			typeof message.content === "string"
				? message.content
				: message.content.map((block) => (block.type === "text" ? block.text : "")).join(""),
		);
}

describe("context guard in a live pi session", () => {
	let root: string;

	beforeEach(() => {
		root = mkdtempSync(join(tmpdir(), "autorag-context-guard-"));
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	// Catalog models such as glm-5.2 (202k window, 131k max output) and
	// deepseek-v4-flash (1M window, 943k max output) leave no input budget once
	// maxTokens is reserved in full. The query must still reach the model.
	it.each([
		{ contextWindow: 202_752, maxTokens: 131_072 },
		{ contextWindow: 262_144, maxTokens: 235_929 },
	])(
		"keeps the user's query in the request for a $contextWindow window with $maxTokens max output",
		async (limits) => {
			const bulkyTool: AgentTool = {
				name: "dump_notes",
				label: "Dump notes",
				description: "Returns a large amount of text",
				parameters: Type.Object({}),
				async execute() {
					return { content: [{ type: "text", text: "memory note ".repeat(20_000) }], details: {} };
				},
			};
			const registration = registerFauxProvider({
				api: `faux-guard-${limits.contextWindow}`,
				models: [{ id: "guarded-model", ...limits }],
			});
			const requests: Message[][] = [];
			const record = (reply: AssistantMessage) => (context: { messages: Message[] }) => {
				requests.push(context.messages);
				return reply;
			};
			registration.setResponses([
				record(fauxAssistantMessage(fauxToolCall("dump_notes", {}), { stopReason: "toolUse" })),
				record(fauxAssistantMessage("done")),
			]);
			const session = await createAutoRAGPiSession({
				cwd: root,
				agentDir: join(root, "agent"),
				model: registration.getModel(),
				apiKey: "test-key",
				getSystemPrompt: () => "AutoRAG system prompt",
				customTools: [bulkyTool],
			});
			try {
				await session.session.prompt("장원영 집 잘사냐?");

				expect(requests).toHaveLength(2);
				for (const request of requests) expect(userTexts(request)).toContain("장원영 집 잘사냐?");
			} finally {
				session.session.dispose();
				registration.unregister();
			}
		},
	);
});
