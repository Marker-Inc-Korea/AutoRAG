import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { RetrievalMemory } from "./memory.ts";
import { renderMemoryContext } from "./renderer.ts";

const checkMemorySchema = Type.Object({
	query: Type.String({
		description:
			"The query you plan to search for — memory will show advisory method hints from past result/evidence feedback",
	}),
});

export interface CheckMemoryDetails {
	signalCount: number;
	topMethod: string | null;
	insightCount: number;
}

export function createCheckMemoryTool(
	memory: RetrievalMemory,
): AgentTool<typeof checkMemorySchema, CheckMemoryDetails> {
	return {
		name: "check_memory",
		label: "Check Memory",
		description:
			"Check retrieval memory for advisory method hints from past result/evidence feedback. Hints are never method disable rules; broaden to other methods when results are insufficient.",
		parameters: checkMemorySchema,
		async execute(_toolCallId: string, params: { query: string }): Promise<AgentToolResult<CheckMemoryDetails>> {
			const hints = memory.getMethodHints(params.query);
			const insights = memory.getInsights(params.query);
			return {
				content: [{ type: "text", text: renderMemoryContext(hints, { insights }) }],
				details: {
					signalCount: memory.getSignalCount(),
					topMethod: hints[0]?.method ?? null,
					insightCount: insights.length,
				},
			};
		},
	};
}
