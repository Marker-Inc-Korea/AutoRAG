import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { MinSyncVectorMethod } from "../minsync/method.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import { type SearchDocumentRetrievalTraceResult, toRetrievalTraceResults } from "./search-documents.ts";

export const SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME = "semantic_search_local_docs";

const searchMinSyncSchema = Type.Object({
	query: Type.String({
		description: "Semantic query to search parsed document mirrors with MinSync vector retrieval.",
	}),
	topK: Type.Optional(
		Type.Integer({ description: "Maximum number of MinSync semantic chunks to return. Defaults to 50." }),
	),
	scope: Type.Optional(Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/**." })),
});

export interface SearchMinSyncDocumentsDetails {
	readonly method: "semantic_search_local_docs";
	readonly resultCount: number;
	readonly sources: readonly string[];
	/** Top candidates in traceable shape (additive; used for the run's retrieval trace). */
	readonly results?: readonly SearchDocumentRetrievalTraceResult[];
}

/**
 * LLM-facing wrapper around the {@link MinSyncVectorMethod} vector
 * retrieval. The model can only supply `query`, `topK`, and an opaque `scope`.
 * MinSync is required, so a missing binary or a failed query is not turned
 * into an "unavailable" result: the error propagates with its own message and
 * the harness reports it as a failed tool call.
 */
export function createSearchMinSyncDocumentsTool(
	getMethod: () => MinSyncVectorMethod,
	resolveScope: (scope: string | undefined) => string | undefined = (scope) => scope,
): AgentTool<typeof searchMinSyncSchema, SearchMinSyncDocumentsDetails> {
	return {
		name: SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
		label: "Search MinSync Documents",
		description:
			"Search parsed document mirrors with MinSync semantic vector retrieval. Use for conceptual and meaning-based search.",
		parameters: searchMinSyncSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchMinSyncDocumentsDetails>> {
			if (params.query.trim().length === 0) {
				return {
					content: [{ type: "text", text: "MinSync query was empty; no documents searched." }],
					details: { method: "semantic_search_local_docs", resultCount: 0, sources: [] },
				};
			}
			const scope = resolveScope(params.scope);
			const results = await getMethod().retrieve(params.query, { topK: params.topK, scope });
			return {
				content: [{ type: "text", text: formatResults(results) }],
				details: {
					method: "semantic_search_local_docs",
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					results: toRetrievalTraceResults(results),
				},
			};
		},
	};
}

function formatResults(results: readonly RetrievalResult[]): string {
	if (results.length === 0) return "No MinSync results.";
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ").slice(0, 500);
		return `[${index + 1}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return `MinSync results:\n\n${rows.join("\n\n")}`;
}
