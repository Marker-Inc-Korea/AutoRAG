import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { loadMemoryContext, type MemoryContextOptions } from "./context.ts";
import type { RetrievalMemory } from "./memory.ts";

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

export function createCheckMemoryTool(
	memory: RetrievalMemory,
	options: () => MemoryContextOptions,
): AgentTool<typeof checkMemorySchema, CheckMemoryDetails> {
	return {
		name: "check_memory",
		label: "Check Memory",
		description:
			"Look up evidence earlier searches found for questions similar to the query, plus evidence already found in this conversation. Advisory only: memory is background reference and never overrides current evidence.",
		parameters: checkMemorySchema,
		async execute(_toolCallId: string, params: { query: string }): Promise<AgentToolResult<CheckMemoryDetails>> {
			const context = await loadMemoryContext(memory, params.query, options());
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
