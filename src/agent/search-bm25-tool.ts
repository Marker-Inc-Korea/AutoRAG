import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { BM25Method, BM25Status } from "../retrieval/methods/bm25.ts";
import { BM25UnavailableError } from "../retrieval/methods/bm25.ts";
import type { RetrievalResult } from "../retrieval/types.ts";

export const SEARCH_BM25_DOCUMENTS_TOOL_NAME = "search_bm25_documents";

const searchBM25Schema = Type.Object({
	query: Type.String({ description: "Lexical query to search in parsed document mirrors with BM25." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of BM25 chunks to return." })),
	scope: Type.Optional(Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/**." })),
});

export interface SearchBM25DocumentsDetails {
	readonly method: "bm25";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly readiness: BM25Status["readiness"];
	readonly engine: BM25Status["engine"];
}

export function createSearchBM25DocumentsTool(
	getMethod: () => BM25Method | undefined,
): AgentTool<typeof searchBM25Schema, SearchBM25DocumentsDetails> {
	return {
		name: SEARCH_BM25_DOCUMENTS_TOOL_NAME,
		label: "Search BM25 Documents",
		description:
			"Search parsed document mirrors with lexical BM25 ranking. Use for exact terms, repeated terms, headings, identifiers, and folder-scoped document search.",
		parameters: searchBM25Schema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchBM25DocumentsDetails>> {
			const method = getMethod();
			if (!method) {
				return unavailableResult("BM25 is not configured for this AutoRAG agent", {
					readiness: "disabled",
					engine: "none",
				});
			}
			const status = method.getStatus();
			if (params.query.trim().length === 0) {
				return {
					content: [{ type: "text", text: "BM25 query was empty; no documents searched." }],
					details: {
						method: "bm25",
						resultCount: 0,
						sources: [],
						readiness: status.readiness,
						engine: status.engine,
					},
				};
			}
			try {
				const results = await method.retrieve(params.query, { topK: params.topK, scope: params.scope });
				return {
					content: [{ type: "text", text: formatResults(results, method.getStatus()) }],
					details: {
						method: "bm25",
						resultCount: results.length,
						sources: [...new Set(results.map((result) => result.source))],
						readiness: method.getStatus().readiness,
						engine: method.getStatus().engine,
					},
				};
			} catch (error) {
				const nextStatus = method.getStatus();
				const message =
					error instanceof BM25UnavailableError ? error.message : `BM25 search failed: ${String(error)}`;
				return unavailableResult(message, nextStatus);
			}
		},
	};
}

function unavailableResult(message: string, status: BM25Status): AgentToolResult<SearchBM25DocumentsDetails> {
	return {
		content: [{ type: "text", text: `${message}\nreadiness=${status.readiness}; engine=${status.engine}` }],
		details: { method: "bm25", resultCount: 0, sources: [], readiness: status.readiness, engine: status.engine },
	};
}

function formatResults(results: readonly RetrievalResult[], status: BM25Status): string {
	if (results.length === 0) return `No BM25 results. readiness=${status.readiness}; engine=${status.engine}`;
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ").slice(0, 500);
		return `[${index + 1}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return `BM25 results (readiness=${status.readiness}; engine=${status.engine}):\n\n${rows.join("\n\n")}`;
}
