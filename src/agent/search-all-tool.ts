import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { RetrievalDiagnostic, RetrievalResult } from "../retrieval/types.ts";

export const SEARCH_ALL_DOCUMENTS_TOOL_NAME = "search_all_documents";

const searchAllSchema = Type.Object({
	query: Type.String({ description: "Query to search across all configured retrieval methods." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of merged results to return." })),
	scope: Type.Optional(Type.String({ description: "Optional opaque virtual-path scope, e.g. /docs or /docs/**." })),
});

export interface SearchAllDocumentsResult {
	readonly results: readonly RetrievalResult[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	readonly perMethodCounts?: Readonly<Record<string, number>>;
}

export interface SearchAllDocumentsProvider {
	searchAllDocuments(
		query: string,
		options?: { readonly topK?: number; readonly scope?: string },
	): Promise<SearchAllDocumentsResult>;
}

export interface SearchAllDocumentsDetails {
	readonly method: "search_all_documents";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	readonly perMethodCounts?: Readonly<Record<string, number>>;
}

/**
 * LLM-facing wrapper around multi-method merged retrieval. The
 * schema accepts only `{ query, topK?, scope? }`; datasource trust fields such
 * as `allowedTags`/`allowedScopes` are not part of the schema and are never
 * forwarded to the provider — only `query`, `topK`, and `scope` are passed
 * through, so model-provided extra properties cannot widen datasource access.
 */
export function createSearchAllDocumentsTool(
	provider: SearchAllDocumentsProvider,
): AgentTool<typeof searchAllSchema, SearchAllDocumentsDetails> {
	return {
		name: SEARCH_ALL_DOCUMENTS_TOOL_NAME,
		label: "Search All Documents",
		description:
			"Search across all configured retrieval methods (posix, BM25, MinSync, datasources) and return merged, deduplicated results. Authority is server-configured; tool arguments can only provide query, topK, and an optional narrowing scope.",
		parameters: searchAllSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchAllDocumentsDetails>> {
			const query = params.query.trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Search query was empty; no documents searched." }],
					details: { method: "search_all_documents", resultCount: 0, sources: [], diagnostics: [] },
				};
			}
			// Only forward query/topK/scope — never model-provided trust fields.
			const { results, diagnostics, perMethodCounts } = await provider.searchAllDocuments(query, {
				topK: params.topK,
				scope: params.scope,
			});
			return {
				content: [{ type: "text", text: formatResults(results, diagnostics) }],
				details: {
					method: "search_all_documents",
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					diagnostics,
					...(perMethodCounts ? { perMethodCounts } : {}),
				},
			};
		},
	};
}

function formatResults(results: readonly RetrievalResult[], diagnostics: readonly RetrievalDiagnostic[]): string {
	const diagnosticSummary =
		diagnostics.length > 0
			? `\n\nDiagnostics: ${diagnostics.map((d) => `${d.source ?? "all"}:${d.code}`).join(", ")}`
			: "";
	if (results.length === 0) return `No results.${diagnosticSummary}`;
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ").slice(0, 500);
		return `[${index + 1}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return `Merged results:\n\n${rows.join("\n\n")}${diagnosticSummary}`;
}
