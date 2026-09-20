import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { RetrievalDiagnostic, RetrievalResult } from "../retrieval/types.ts";
import { type SearchDocumentRetrievalTraceResult, toRetrievalTraceResults } from "./search-documents.ts";

export const SEARCH_ALL_DOCUMENTS_TOOL_NAME = "search_all_documents";

const searchAllSchema = Type.Object({
	query: Type.String({ description: "Query to search across all configured retrieval methods." }),
	topK: Type.Optional(
		Type.Integer({
			description:
				"Upper bound on returned evidence. Omit it to receive every distinct chunk the methods found; set it only when you deliberately want a shorter list.",
		}),
	),
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
	/** Top candidates in traceable shape (additive; used for the run's retrieval trace). */
	readonly results?: readonly SearchDocumentRetrievalTraceResult[];
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
			"Search across all configured retrieval methods (posix, MinSync, datasources) and return every distinct chunk they found, with only pure duplicates removed. This is the exhaustive retrieval surface: one call can return many candidates, including several passages of the same document, so judge the evidence yourself rather than assuming it was pre-filtered. Authority is server-configured; tool arguments can only provide query, topK, and an optional narrowing scope.",
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
					results: toRetrievalTraceResults(results),
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
