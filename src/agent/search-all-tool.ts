import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { resultContextSuffix } from "../retrieval/chunk-location.ts";
import type { RetrievalDiagnostic, RetrievalResult, RetrievalWindowReport } from "../retrieval/types.ts";
import { EvidenceLedger } from "./evidence-ledger.ts";
import { type RecordCutToolOptions, registerAndRender } from "./result-render.ts";
import { type SearchDocumentRetrievalTraceResult, toRetrievalTraceResults } from "./search-documents.ts";
import { diagnosticSummary, parseToolWindow, timeWindowProperties, windowNotice } from "./time-window-tool.ts";

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
	...timeWindowProperties,
});

export interface SearchAllDocumentsResult {
	readonly results: readonly RetrievalResult[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	readonly perMethodCounts?: Readonly<Record<string, number>>;
	/** Present only for a windowed search: what the window did to the candidates. */
	readonly window?: RetrievalWindowReport;
}

export interface SearchAllDocumentsProvider {
	searchAllDocuments(
		query: string,
		options?: {
			readonly topK?: number;
			readonly scope?: string;
			readonly since?: number;
			readonly until?: number;
			readonly signal?: AbortSignal;
		},
	): Promise<SearchAllDocumentsResult>;
}

export interface SearchAllDocumentsDetails {
	readonly method: "search_all_documents";
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	readonly perMethodCounts?: Readonly<Record<string, number>>;
	/** Present only for a windowed search. */
	readonly window?: RetrievalWindowReport;
	/** Top candidates in traceable shape (additive; used for the run's retrieval trace). */
	readonly results?: readonly SearchDocumentRetrievalTraceResult[];
}

/**
 * LLM-facing wrapper around multi-method merged retrieval. The
 * schema accepts only `{ query, topK?, scope?, since?, until? }`; extra model-provided
 * properties are not part of the schema and are never forwarded to the
 * provider — only `query`, `topK`, `scope`, and the parsed time window are passed through.
 */
export function createSearchAllDocumentsTool(
	provider: SearchAllDocumentsProvider,
	ledger: EvidenceLedger = new EvidenceLedger(),
	options: RecordCutToolOptions = {},
): AgentTool<typeof searchAllSchema, SearchAllDocumentsDetails> {
	return {
		name: SEARCH_ALL_DOCUMENTS_TOOL_NAME,
		label: "Search All Documents",
		description:
			"Search across all configured retrieval methods (posix, MinSync, datasources) and return every distinct chunk they found, with only pure duplicates removed. This is the exhaustive retrieval surface: one call can return many candidates, including several passages of the same document, so judge the evidence yourself rather than assuming it was pre-filtered. Authority is server-configured; tool arguments can only provide query, topK, an optional narrowing scope, and an optional since/until time window. A time window only narrows: methods that cannot date a result exclude it and are reported, never passed through as in range.",
		parameters: searchAllSchema,
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<SearchAllDocumentsDetails>> {
			const query = params.query.trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Search query was empty; no documents searched." }],
					details: { method: "search_all_documents", resultCount: 0, sources: [], diagnostics: [] },
				};
			}
			// An invalid since/until throws its parse error verbatim before anything is searched.
			const window = parseToolWindow(params);
			// Only forward query/topK/scope/window — never model-provided trust fields.
			const {
				results,
				diagnostics,
				perMethodCounts,
				window: windowReport,
			} = await provider.searchAllDocuments(query, {
				topK: params.topK,
				scope: params.scope,
				...window,
				...(signal === undefined ? {} : { signal }),
			});
			const { text } = registerAndRender(
				ledger,
				SEARCH_ALL_DOCUMENTS_TOOL_NAME,
				results,
				(result) => `${result.source} score=${result.score.toFixed(4)}${resultContextSuffix(result.metadata)}`,
				options.recordCut,
			);
			return {
				content: [{ type: "text", text: withDiagnostics(text, results.length, diagnostics, windowReport) }],
				details: {
					method: "search_all_documents",
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					diagnostics,
					...(perMethodCounts ? { perMethodCounts } : {}),
					...(windowReport === undefined ? {} : { window: windowReport }),
					results: toRetrievalTraceResults(results),
				},
			};
		},
	};
}

function withDiagnostics(
	rows: string,
	resultCount: number,
	diagnostics: readonly RetrievalDiagnostic[],
	window: RetrievalWindowReport | undefined,
): string {
	const notice = windowNotice(window, resultCount);
	const diagnosticText = diagnosticSummary(diagnostics, "all");
	const noticeText = notice === undefined ? "" : `\n\n${notice}`;
	if (resultCount === 0) return `No results.${noticeText}${diagnosticText}`;
	return `Merged results (cite evidence by its [eN] id):\n\n${rows}${noticeText}${diagnosticText}`;
}
