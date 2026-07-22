import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { RetrievalDiagnostic, RetrievalResult } from "../retrieval/types.ts";

export const SEARCH_DATASOURCE_DOCUMENTS_TOOL_NAME = "search_datasource_documents";

const searchDatasourceSchema = Type.Object({
	query: Type.String({ description: "Query to search across configured external datasource skills." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of datasource chunks to return." })),
	scope: Type.Optional(
		Type.String({ description: "Optional opaque datasource scope, e.g. /kakao/account or /kakao/account/**." }),
	),
});

export interface SearchDatasourceDocumentsDetails {
	readonly method: "datasource";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
}

export interface DatasourceSearchProvider {
	searchDatasourceDocuments(
		query: string,
		options?: { readonly topK?: number; readonly scope?: string },
	): Promise<{ readonly results: readonly RetrievalResult[]; readonly diagnostics: readonly RetrievalDiagnostic[] }>;
}

export function createSearchDatasourceDocumentsTool(
	provider: DatasourceSearchProvider,
): AgentTool<typeof searchDatasourceSchema, SearchDatasourceDocumentsDetails> {
	return {
		name: SEARCH_DATASOURCE_DOCUMENTS_TOOL_NAME,
		label: "Search Datasource Documents",
		description:
			"Search configured external datasource skills such as KakaoTalk chats. Authority is server-configured; tool arguments can only provide query, topK, and an optional narrowing scope.",
		parameters: searchDatasourceSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchDatasourceDocumentsDetails>> {
			const query = params.query.trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Datasource query was empty; no datasource documents searched." }],
					details: { method: "datasource", resultCount: 0, sources: [], diagnostics: [] },
				};
			}
			const { results, diagnostics } = await provider.searchDatasourceDocuments(query, {
				topK: params.topK,
				scope: params.scope,
			});
			return {
				content: [{ type: "text", text: formatResults(results, diagnostics) }],
				details: {
					method: "datasource",
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					diagnostics,
				},
			};
		},
	};
}

function formatResults(results: readonly RetrievalResult[], diagnostics: readonly RetrievalDiagnostic[]): string {
	const diagnosticSummary =
		diagnostics.length > 0
			? `\n\nDiagnostics: ${diagnostics.map((d) => `${d.source ?? "datasource"}:${d.code}`).join(", ")}`
			: "";
	if (results.length === 0) return `No datasource results.${diagnosticSummary}`;
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ").slice(0, 500);
		return `[${index + 1}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return `Datasource results:\n\n${rows.join("\n\n")}${diagnosticSummary}`;
}
