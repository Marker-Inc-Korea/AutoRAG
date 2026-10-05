import type { AgentMessage } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { describe, expect, it } from "vitest";
import {
	createContextTokenGuardedTransform,
	estimateContextMessageTokens,
	guardAgentContextMessages,
} from "../../src/agent/context-token-guard.ts";

const model = {
	id: "guard-test",
	name: "Guard test",
	api: "test-api",
	provider: "test-provider",
	baseUrl: "http://localhost",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 1_200,
	maxTokens: 200,
} as Model<any>;

function retrievalMessage(contents: readonly string[]): AgentMessage {
	return {
		role: "toolResult",
		toolCallId: "call-1",
		toolName: "search_all_documents",
		content: [
			{
				type: "text",
				text: contents
					.map((content, index) => `[${index + 1}] /docs/${index}.md score=0.9\n${content}`)
					.join("\n\n"),
			},
		],
		isError: false,
		timestamp: Date.now(),
	};
}

function resultText(message: AgentMessage | undefined): string {
	if (message?.role !== "toolResult") return "";
	return message.content
		.filter((block): block is { type: "text"; text: string } => block.type === "text")
		.map((block) => block.text)
		.join("");
}

describe("context token guard", () => {
	it("keeps complete high-ranked candidates and drops lower-ranked candidates with a marker", () => {
		const contents = ["first candidate ".repeat(40), "second candidate ".repeat(40), "third candidate ".repeat(40)];
		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, [
			retrievalMessage(contents),
		]);
		const text = resultText(guarded[0]);

		expect(text).toContain(contents[0]);
		expect(text).not.toContain(contents[2]);
		expect(text).toContain("AutoRAG context guard");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});

	it("guards baseline retrieval embedded in a user prompt", () => {
		const messages: AgentMessage[] = [
			{
				role: "user",
				content: `Baseline retrieval evidence (already gathered for you):\n[1] /docs/first.md\n${"first ".repeat(120)}\n[2] /docs/second.md\n${"second ".repeat(120)}\n\nProduce the best answer now.`,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const message = guarded[0];
		const text = message?.role === "user" && typeof message.content === "string" ? message.content : "";

		expect(text).toContain("first ".repeat(120));
		expect(text).not.toContain("second ".repeat(120));
		expect(text).toContain("AutoRAG context guard");
	});

	it("returns the same array when the request already fits", () => {
		const messages = [retrievalMessage(["short evidence"])];
		expect(guardAgentContextMessages(model, messages)).toBe(messages);
	});

	it("bounds an oversized non-retrieval tool result with an explicit marker", () => {
		const messages: AgentMessage[] = [
			{
				role: "toolResult",
				toolCallId: "call-bash",
				toolName: "bash",
				content: [{ type: "text", text: "large command output ".repeat(1_000) }],
				isError: false,
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);

		expect(resultText(guarded[0])).toContain("AutoRAG context guard");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});

	it("bounds oversized image input with an explicit marker", () => {
		const messages: AgentMessage[] = [
			{
				role: "user",
				content: [{ type: "image", mimeType: "image/png", data: "large-image" }],
				timestamp: Date.now(),
			},
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const message = guarded[0];
		const text =
			message?.role === "user" && Array.isArray(message.content) && message.content[0]?.type === "text"
				? message.content[0].text
				: "";

		expect(text).toContain("AutoRAG context guard");
	});

	it("preserves tool declarations while reducing an oversized system prompt", () => {
		const messages: AgentMessage[] = [
			{
				role: "system",
				content: "system instruction ".repeat(1_000),
				toolsAdded: [
					{ name: "search_all_documents", description: "search", parameters: { type: "object" } as never },
				],
				timestamp: Date.now(),
			},
			{ role: "user", content: "answer this", timestamp: Date.now() },
		];

		const guarded = guardAgentContextMessages({ ...model, contextWindow: 500, maxTokens: 100 }, messages);
		const system = guarded[0];

		expect(system?.role).toBe("system");
		const systemContent = system?.role === "system" ? system.content : "";
		const systemText =
			typeof systemContent === "string"
				? systemContent
				: systemContent.map((block) => (block.type === "text" ? block.text : "")).join("");
		expect(systemText).toContain("AutoRAG context guard");
		expect(system?.role === "system" ? system.toolsAdded?.[0]?.name : undefined).toBe("search_all_documents");
	});

	it("composes with an existing context transform", async () => {
		const transform = createContextTokenGuardedTransform(
			{ ...model, contextWindow: 500, maxTokens: 100 },
			async (messages) => [{ role: "user", content: "memory hint", timestamp: Date.now() }, ...messages],
		);

		const guarded = await transform([retrievalMessage(["latest evidence ".repeat(200)])]);

		expect(guarded[0]?.role).toBe("user");
		expect(estimateContextMessageTokens(guarded)).toBeLessThanOrEqual(Math.floor(500 / 1.2) - 100);
	});
});
