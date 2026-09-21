import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { datasourceSearchToolName } from "../datasource/tool-naming.ts";
import type { RetrievalDiagnostic, RetrievalResult } from "../retrieval/types.ts";
import { type SearchDocumentRetrievalTraceResult, toRetrievalTraceResults } from "./search-documents.ts";

/**
 * Per-datasource search tools.
 *
 * One tool is generated per authorized datasource connection (e.g.
 * `search_datasource_discord`, or `search_datasource_kakao_work` for a
 * `kakao-work` account alias). Its execution registers only that connection's
 * retrieval methods with the retriever, so no other datasource CLI is spawned
 * and the result set carries only that connection's hits.
 *
 * These generated tools are the only model-facing datasource retrieval
 * surface. Cross-datasource fan-out belongs to `search_all_documents`, which
 * already spans every configured retrieval method — including datasources — so
 * a datasource-only fan-out tool would be redundant. Every authorized
 * connection must therefore appear here; a datasource with no generated tool
 * would become unreachable except through the full fan-out.
 *
 * Tools are generated from the configured, access-authorized datasource skills
 * at agent construction, so a datasource that is disabled in config or denied
 * by the trusted access context never appears in the tool list.
 */

export const SEARCH_SINGLE_DATASOURCE_TOOL_PREFIX = "search_datasource_";

/** Model-visible tool name for one datasource connection. */
export const singleDatasourceToolName: (datasourceId: string) => string = datasourceSearchToolName;

/** Static per-connection descriptor used to generate one tool each. */
export interface SingleDatasourceToolSpec {
	/** Trusted datasource id (alias-aware, e.g. `kakao-work`). */
	readonly datasourceId: string;
	/** Operator/descriptor-authored context shown in the tool description. */
	readonly description: string;
	/** Authorized instance roots (e.g. `/kakao/personal`) for the description. */
	readonly instanceScopes: readonly string[];
}

export interface SingleDatasourceSearchProvider {
	searchSingleDatasourceDocuments(
		datasourceId: string,
		query: string,
		options?: { readonly topK?: number; readonly scope?: string },
	): Promise<{ readonly results: readonly RetrievalResult[]; readonly diagnostics: readonly RetrievalDiagnostic[] }>;
}

export interface SearchSingleDatasourceDetails {
	readonly method: "datasource";
	readonly datasource: string;
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	/** Top candidates in traceable shape (additive; used for the run's retrieval trace). */
	readonly results?: readonly SearchDocumentRetrievalTraceResult[];
}

export function createSingleDatasourceSearchTools(
	provider: SingleDatasourceSearchProvider,
	specs: readonly SingleDatasourceToolSpec[],
): AgentTool<typeof searchSingleDatasourceSchema, SearchSingleDatasourceDetails>[] {
	return specs.map((spec) => createTool(provider, spec));
}

const searchSingleDatasourceSchema = Type.Object({
	query: Type.String({ description: "Query to search within this datasource connection." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of chunks to return. Defaults to 50." })),
	scope: Type.Optional(
		Type.String({ description: "Optional scope narrowing inside this datasource, e.g. one instance root." }),
	),
});

function createTool(
	provider: SingleDatasourceSearchProvider,
	spec: SingleDatasourceToolSpec,
): AgentTool<typeof searchSingleDatasourceSchema, SearchSingleDatasourceDetails> {
	const scopeLine = spec.instanceScopes.length > 0 ? ` Authorized scopes: ${spec.instanceScopes.join(", ")}.` : "";
	return {
		name: singleDatasourceToolName(spec.datasourceId),
		label: `Search ${spec.datasourceId}`,
		description:
			`Search only the "${spec.datasourceId}" datasource connection: ${spec.description} ` +
			`This tool spawns only this connection's CLIs and returns only this connection's hits.${scopeLine} ` +
			`Authority is server-configured; tool arguments can only provide query, topK, and an optional narrowing scope.`,
		parameters: searchSingleDatasourceSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<SearchSingleDatasourceDetails>> {
			const query = params.query.trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Datasource query was empty; nothing searched." }],
					details: {
						method: "datasource",
						datasource: spec.datasourceId,
						resultCount: 0,
						sources: [],
						diagnostics: [],
					},
				};
			}
			const { results, diagnostics } = await provider.searchSingleDatasourceDocuments(spec.datasourceId, query, {
				topK: params.topK,
				scope: params.scope,
			});
			return {
				content: [{ type: "text", text: formatDatasourceResults(results, diagnostics) }],
				details: {
					method: "datasource",
					datasource: spec.datasourceId,
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					diagnostics,
					results: toRetrievalTraceResults(results),
				},
			};
		},
	};
}

/** Shared rendering for datasource search tool results (fan-out removed). */
export function formatDatasourceResults(
	results: readonly RetrievalResult[],
	diagnostics: readonly RetrievalDiagnostic[],
): string {
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
