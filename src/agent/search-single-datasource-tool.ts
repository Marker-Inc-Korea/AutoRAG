import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { datasourceSearchToolName } from "../datasource/tool-naming.ts";
import type { SearchFilterSpec } from "../datasource/types.ts";
import type { RetrievalDiagnostic, RetrievalResult, RetrievalWindowReport } from "../retrieval/types.ts";
import { EvidenceLedger } from "./evidence-ledger.ts";
import { type RecordCutToolOptions, registerAndRender } from "./result-render.ts";
import { type SearchDocumentRetrievalTraceResult, toRetrievalTraceResults } from "./search-documents.ts";
import { diagnosticSummary, parseToolWindow, timeWindowProperties, windowNotice } from "./time-window-tool.ts";

/**
 * Per-datasource search tools.
 *
 * One tool is generated per configured datasource connection (e.g.
 * `search_datasource_discord`, or `search_datasource_kakao_work` for a
 * `kakao-work` account alias). Its execution registers only that connection's
 * retrieval methods with the retriever, so no other datasource CLI is spawned
 * and the result set carries only that connection's hits.
 *
 * These generated tools are the only model-facing datasource retrieval
 * surface. Cross-datasource fan-out belongs to `search_all_documents`, which
 * already spans every configured retrieval method — including datasources — so
 * a datasource-only fan-out tool would be redundant. Every configured
 * connection must therefore appear here; a datasource with no generated tool
 * would become unreachable except through the full fan-out.
 *
 * Tools are generated from the configured datasource skills at agent
 * construction; a connection not present in the config never appears in the
 * tool list.
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
	/** Instance roots (e.g. `/kakao/personal`) for the description. */
	readonly instanceScopes: readonly string[];
	/**
	 * Names of the per-search filters this connection's skill offers (each becomes a string-list tool argument of that name).
	 * Empty or absent: the tool has no extra argument.
	 */
	readonly filterNames?: readonly string[];
	/**
	 * The filters and ids known when the session is built (read when the tool is declared to the model, so the ids appear in
	 * its description). Absent or empty: the description names none and the datasource validates a value itself.
	 */
	readonly searchFilters?: () => readonly SearchFilterSpec[];
}

export interface SingleDatasourceSearchProvider {
	searchSingleDatasourceDocuments(
		datasourceId: string,
		query: string,
		options?: {
			readonly topK?: number;
			readonly scope?: string;
			readonly since?: number;
			readonly until?: number;
			readonly filters?: Readonly<Record<string, readonly string[]>>;
			readonly signal?: AbortSignal;
		},
	): Promise<{
		readonly results: readonly RetrievalResult[];
		readonly diagnostics: readonly RetrievalDiagnostic[];
		/** Present only for a windowed search. */
		readonly window?: RetrievalWindowReport;
	}>;
	/** The filters this datasource offers now, with the exact ids each accepts. Empty when none, or when unreadable. */
	searchFilterSpecs?(datasourceId: string, signal?: AbortSignal): Promise<readonly SearchFilterSpec[]>;
}

export interface SearchSingleDatasourceDetails {
	readonly method: "datasource";
	readonly datasource: string;
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly diagnostics: readonly RetrievalDiagnostic[];
	/** Present only for a windowed search. */
	readonly window?: RetrievalWindowReport;
	/** Top candidates in traceable shape (additive; used for the run's retrieval trace). */
	readonly results?: readonly SearchDocumentRetrievalTraceResult[];
}

export function createSingleDatasourceSearchTools(
	provider: SingleDatasourceSearchProvider,
	specs: readonly SingleDatasourceToolSpec[],
	ledger: EvidenceLedger = new EvidenceLedger(),
	options: RecordCutToolOptions = {},
): AgentTool<typeof searchSingleDatasourceSchema, SearchSingleDatasourceDetails>[] {
	return specs.map((spec) => createTool(provider, spec, ledger, options));
}

const searchSingleDatasourceSchema = Type.Object({
	query: Type.String({ description: "Query to search within this datasource connection." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of chunks to return. Defaults to 20." })),
	scope: Type.Optional(
		Type.String({ description: "Optional scope narrowing inside this datasource, e.g. one instance root." }),
	),
	...timeWindowProperties,
});

/** The tool schema; each filter a connection declares is added to it as a string list. */
function schemaFor(spec: SingleDatasourceToolSpec): typeof searchSingleDatasourceSchema {
	const known = new Map((spec.searchFilters?.() ?? []).map((filter) => [filter.name, filter]));
	const filters = Object.fromEntries(
		(spec.filterNames ?? []).map((name) => {
			const filter = known.get(name);
			const ids = filter?.values.map((value) => value.id) ?? [];
			const description =
				`${filter?.description ?? `Narrow the search by ${name}.`}` +
				(ids.length > 0 ? ` Accepted values: ${ids.join(", ")}.` : "") +
				" Leave out to search all.";
			return [name, Type.Optional(Type.Array(Type.String(), { description }))];
		}),
	);
	return Type.Object({
		...searchSingleDatasourceSchema.properties,
		...filters,
	}) as typeof searchSingleDatasourceSchema;
}

function filterDescription(spec: SingleDatasourceToolSpec): string {
	const known = spec.searchFilters?.() ?? [];
	const lines = known
		.filter((filter) => spec.filterNames?.includes(filter.name) === true && filter.values.length > 0)
		.map(
			(filter) =>
				` Filter "${filter.name}": ${filter.description} Values: ` +
				`${filter.values.map((value) => (value.detail === undefined ? value.id : `${value.id} (${value.detail})`)).join(", ")}.`,
		);
	return lines.join("");
}

/** Lower-cased, trimmed, de-duplicated filter values in the order given; blanks dropped. */
function normalizeFilterValues(raw: readonly string[] | undefined): readonly string[] {
	if (raw === undefined) return [];
	return [...new Set(raw.map((value) => value.trim().toLowerCase()).filter((value) => value.length > 0))];
}

function createTool(
	provider: SingleDatasourceSearchProvider,
	spec: SingleDatasourceToolSpec,
	ledger: EvidenceLedger,
	options: RecordCutToolOptions,
): AgentTool<typeof searchSingleDatasourceSchema, SearchSingleDatasourceDetails> {
	const scopeLine = spec.instanceScopes.length > 0 ? ` Instance scopes: ${spec.instanceScopes.join(", ")}.` : "";
	return {
		name: singleDatasourceToolName(spec.datasourceId),
		label: `Search ${spec.datasourceId}`,
		// Read when the tool is declared, so the ids the datasource knows then are what the model sees.
		get description(): string {
			const names = spec.filterNames ?? [];
			const filterClause =
				names.length > 0 ? `, ${names.map((name) => `an optional ${name} filter`).join(", ")}` : "";
			return (
				`Search only the "${spec.datasourceId}" datasource connection: ${spec.description} ` +
				`This tool spawns only this connection's CLIs and returns only this connection's hits.${scopeLine}${filterDescription(spec)} ` +
				`Authority is server-configured; tool arguments can only provide query, topK, an optional narrowing scope${filterClause}, and an optional since/until time window.`
			);
		},
		get parameters(): typeof searchSingleDatasourceSchema {
			return schemaFor(spec);
		},
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<SearchSingleDatasourceDetails>> {
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
			// An invalid since/until throws its parse error verbatim before anything is searched.
			const window = parseToolWindow(params);
			const filters: Record<string, readonly string[]> = {};
			const asked = params as unknown as Record<string, readonly string[] | undefined>;
			for (const name of spec.filterNames ?? []) {
				const values = normalizeFilterValues(asked[name]);
				if (values.length > 0) filters[name] = values;
			}
			if (Object.keys(filters).length > 0) {
				// A value the datasource does not hold would only come back as an empty answer that looks like "nothing found".
				// Say so, with the real ids. When the ids cannot be listed now, the datasource itself decides.
				const live = (await provider.searchFilterSpecs?.(spec.datasourceId, signal)) ?? [];
				for (const [name, values] of Object.entries(filters)) {
					const accepted = live.find((filter) => filter.name === name)?.values.map((value) => value.id);
					if (accepted === undefined || accepted.length === 0) continue;
					const unknown = values.filter((value) => !accepted.includes(value));
					if (unknown.length === 0) continue;
					return {
						content: [
							{
								type: "text",
								text:
									`Nothing searched: ${unknown.map((value) => `"${value}"`).join(", ")} is not among the ${name} "${spec.datasourceId}" holds. ` +
									`Available: ${accepted.join(", ")}.`,
							},
						],
						isError: true,
						details: {
							method: "datasource",
							datasource: spec.datasourceId,
							resultCount: 0,
							sources: [],
							diagnostics: [],
						},
					};
				}
			}
			const {
				results,
				diagnostics,
				window: windowReport,
			} = await provider.searchSingleDatasourceDocuments(spec.datasourceId, query, {
				topK: params.topK,
				scope: params.scope,
				...window,
				...(Object.keys(filters).length > 0 ? { filters } : {}),
				...(signal === undefined ? {} : { signal }),
			});
			const { text } = registerAndRender(
				ledger,
				`search_datasource_${spec.datasourceId}`,
				query,
				results,
				(result) => `${result.source} score=${result.score.toFixed(4)}`,
				options.recordCut,
			);
			return {
				content: [{ type: "text", text: datasourceResultsText(text, results.length, diagnostics, windowReport) }],
				details: {
					method: "datasource",
					datasource: spec.datasourceId,
					resultCount: results.length,
					sources: [...new Set(results.map((result) => result.source))],
					diagnostics,
					...(windowReport === undefined ? {} : { window: windowReport }),
					results: toRetrievalTraceResults(results),
				},
			};
		},
	};
}

function datasourceResultsText(
	rows: string,
	resultCount: number,
	diagnostics: readonly RetrievalDiagnostic[],
	window?: RetrievalWindowReport,
): string {
	const notice = windowNotice(window, resultCount);
	const noticeText = notice === undefined ? "" : `\n\n${notice}`;
	const diagnosticText = diagnosticSummary(diagnostics, "datasource");
	if (resultCount === 0) return `No datasource results.${noticeText}${diagnosticText}`;
	return `Datasource results (cite evidence by its [eN] id):\n\n${rows}${noticeText}${diagnosticText}`;
}

/** Shared rendering for datasource search tool results (fan-out removed). */
export function formatDatasourceResults(
	results: readonly RetrievalResult[],
	diagnostics: readonly RetrievalDiagnostic[],
	evidenceIds: readonly string[],
): string {
	const rows = results.map((result, index) => {
		const line = result.content.replace(/\s+/gu, " ");
		return `[${evidenceIds[index]}] ${result.source} score=${result.score.toFixed(4)}\n${line}`;
	});
	return datasourceResultsText(rows.join("\n\n"), results.length, diagnostics);
}
