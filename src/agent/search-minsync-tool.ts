import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { MinSyncVectorMethod } from "../minsync/method.ts";
import type { RetrievalResult } from "../retrieval/types.ts";

export const SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME = "search_minsync_documents";

const searchMinSyncSchema = Type.Object({
	query: Type.String({
		description: "Semantic query to search parsed document mirrors with MinSync vector retrieval.",
	}),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of MinSync semantic chunks to return." })),
	scope: Type.Optional(Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/**." })),
});

export interface SearchMinSyncDocumentsDetails {
	readonly method: "search_minsync_documents";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly available: boolean;
}

/**
 * LLM-facing wrapper around the {@link MinSyncVectorMethod} vector
 * retrieval. The model can only supply `query`, `topK`, and an opaque `scope`.
 * When the method is missing, the binary is missing, or retrieval errors, the
 * tool returns a path-free unavailable message with `available: false` and a
 * zero result count — never a real binary or workspace path.
 */
export function createSearchMinSyncDocumentsTool(
	getMethod: () => MinSyncVectorMethod | undefined,
): AgentTool<typeof searchMinSyncSchema, SearchMinSyncDocumentsDetails> {
	return {
		name: SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
		label: "Search MinSync Documents",
		description:
			"Search parsed document mirrors with MinSync semantic vector retrieval. Use for conceptual and meaning-based search.",
		parameters: searchMinSyncSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchMinSyncDocumentsDetails>> {
			const method = getMethod();
			if (!method) {
				return unavailableResult("MinSync semantic search is not configured for this AutoRAG agent");
			}
			if (method.isBinaryMissing()) {
				return unavailableResult("MinSync semantic search is unavailable; the configured binary is missing.");
			}
			if (params.query.trim().length === 0) {
				return {
					content: [{ type: "text", text: "MinSync query was empty; no documents searched." }],
					details: { method: "search_minsync_documents", resultCount: 0, sources: [], available: true },
				};
			}
			try {
				const results = await method.retrieve(params.query, { topK: params.topK, scope: params.scope });
				return {
					content: [{ type: "text", text: formatResults(results) }],
					details: {
						method: "search_minsync_documents",
						resultCount: results.length,
						sources: [...new Set(results.map((result) => result.source))],
						available: true,
					},
				};
			} catch {
				return unavailableResult("MinSync semantic search is unavailable; the query could not be completed.");
			}
		},
	};
}

function unavailableResult(message: string): AgentToolResult<SearchMinSyncDocumentsDetails> {
	return {
		content: [{ type: "text", text: message }],
		details: {
			method: "search_minsync_documents",
			resultCount: 0,
			sources: [],
			available: false,
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
