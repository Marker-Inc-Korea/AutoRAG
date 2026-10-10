import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { loadMemoryContext, type MemoryContextOptions } from "./context.ts";
import type { RetrievalMemory } from "./memory.ts";
import { renderMemoryContext } from "./renderer.ts";

const checkMemorySchema = Type.Object({
	query: Type.String({
		description:
			"The question you are about to search for; memory returns advisory evidence from similar past searches",
	}),
});

export interface CheckMemoryDetails {
	currentCount: number;
	similarCount: number;
	insightCount: number;
}

/**
 * `options` returns undefined when this session must not read retrieval memory
 * (remote P2P peers); the tool then answers as if memory were empty.
 */
export function createCheckMemoryTool(
	memory: RetrievalMemory,
	options: () => MemoryContextOptions | undefined,
): AgentTool<typeof checkMemorySchema, CheckMemoryDetails> {
	return {
		name: "check_memory",
		label: "Check Memory",
		description:
			"Look up evidence earlier searches found for questions similar to the query, plus evidence already found in this conversation. Advisory only: memory is background reference and never overrides current evidence.",
		parameters: checkMemorySchema,
		async execute(_toolCallId: string, params: { query: string }): Promise<AgentToolResult<CheckMemoryDetails>> {
			const contextOptions = options();
			if (contextOptions === undefined) {
				return {
					content: [{ type: "text", text: renderMemoryContext({ current: [], similar: [], insights: [] }) }],
					details: { currentCount: 0, similarCount: 0, insightCount: 0 },
				};
			}
			const context = await loadMemoryContext(memory, params.query, contextOptions);
			return {
				content: [{ type: "text", text: context.text }],
				details: {
					currentCount: context.currentCount,
					similarCount: context.similarCount,
					insightCount: context.insightCount,
				},
			};
		},
	};
}
