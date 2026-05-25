import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { RetrievalMemory } from "./memory.ts";
import { renderMemoryContext } from "./renderer.ts";

const checkMemorySchema = Type.Object({
	query: Type.String({
		description: "The query you plan to search for — memory will show similar past queries and which methods worked",
	}),
});

export interface CheckMemoryDetails {
	entryCount: number;
	topMethod: string | null;
}

export function createCheckMemoryTool(
	memory: RetrievalMemory,
): AgentTool<typeof checkMemorySchema, CheckMemoryDetails> {
	return {
		name: "check_memory",
		label: "Check Memory",
		description:
			"Check retrieval memory for past query outcomes. Returns which search methods were useful or not useful for similar queries. Call this before searching to pick the best method.",
		parameters: checkMemorySchema,
		async execute(_toolCallId: string, params: { query: string }): Promise<AgentToolResult<CheckMemoryDetails>> {
			const entries = memory.getEntries();
			const summary = renderMemoryContext(entries);
			const priority = memory.getMethodPriority(params.query);

			let recommendation = "";
			if (priority.length > 0) {
				recommendation =
					"\n\n## Recommended Methods\n" +
					priority
						.map((p, i) => `${i + 1}. **${p.method}** (usefulness: ${(p.score * 100).toFixed(0)}%)`)
						.join("\n");
			}

			return {
				content: [{ type: "text", text: summary + recommendation }],
				details: {
					entryCount: entries.length,
					topMethod: priority[0]?.method ?? null,
				},
			};
		},
	};
}
