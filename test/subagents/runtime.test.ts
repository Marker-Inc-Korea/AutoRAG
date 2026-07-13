import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { Model } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { describe, expect, it } from "vitest";
import { createMandatorySubagentSession } from "../../src/subagents/runtime.ts";

const model: Model<"openai-responses"> = {
	id: "gpt-5.6-sol",
	name: "GPT-5.6 Sol",
	api: "openai-responses",
	provider: "myproxy",
	baseUrl: "https://example.invalid/v1",
	reasoning: true,
	input: ["text", "image"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 400_000,
	maxTokens: 128_000,
};

const customTool: AgentTool = {
	name: "custom_search",
	label: "Custom search",
	description: "Test search tool",
	parameters: Type.Object({ query: Type.String() }),
	execute: async () => ({ content: [{ type: "text", text: "ok" }], details: {} }),
};

describe("mandatory pi-subagents runtime", () => {
	it("loads pi-subagents and exposes its tools beside AutoRAG tools", async () => {
		const runtime = await createMandatorySubagentSession({
			cwd: process.cwd(),
			model,
			systemPrompt: "test prompt",
			tools: [customTool],
		});
		try {
			expect(runtime.session.getActiveToolNames()).toEqual(
				expect.arrayContaining(["custom_search", "subagent", "wait"]),
			);
			expect(runtime.extensionPath).toContain("pi-subagents/src/extension/index.ts");
		} finally {
			runtime.session.dispose();
		}
	});

	it("fails closed when the mandatory extension path cannot load", async () => {
		await expect(
			createMandatorySubagentSession({
				cwd: process.cwd(),
				model,
				systemPrompt: "test prompt",
				tools: [customTool],
				extensionPath: "/definitely/missing/pi-subagents.ts",
			}),
		).rejects.toThrow(/mandatory pi-subagents extension/i);
	});
});
