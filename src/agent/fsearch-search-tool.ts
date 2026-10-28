import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { FSearchSearchRequest, FSearchSearchResult } from "../fsearch/index.ts";
import {
	collectNameCandidates,
	mergeNameHits,
	renderNameHitLine,
	renderNameSearchProblems,
	searchNameCandidates,
} from "./name-candidates.ts";

export const FSEARCH_SEARCH_TOOL_NAME = "fsearch_search";

const MAX_RESULTS_CAP = 1000;
const DEFAULT_MAX_RESULTS = 100;

export interface FSearchSearchProvider {
	searchFsearch(request: FSearchSearchRequest): Promise<FSearchSearchResult>;
}

export interface FSearchSearchDetails {
	readonly method: typeof FSEARCH_SEARCH_TOOL_NAME;
	readonly resultCount: number;
	readonly sources: readonly string[];
	/** Retrieval-trace shape: each hit's absolute path plus its rendered line. */
	readonly results: readonly { readonly source: string; readonly excerpt: string }[];
}

const fsearchSearchSchema = Type.Object({
	query: Type.String({
		description:
			'FSearch query syntax over file and folder names: space = AND, OR keyword = OR (never the pipe |), ! = NOT, "quoted phrase", wildcards (*.pdf), ext:pdf;docx, size:>10mb, path:<fragment>, case:, regex:. When the slow-walk fallback answers (fsearch-cli not installed), only substring/regex name matching applies.',
	}),
	alternatives: Type.Optional(
		Type.Array(Type.String(), {
			description:
				"Other names to try in the same call, such as a translation, an abbreviation, or another spelling. Every name is searched at once with the same options and the hits are merged, each marked with the names that found it. Prefer this over repeated calls when the exact name is unknown.",
		}),
	),
	regex: Type.Optional(Type.Boolean({ description: "Treat query as a regular expression." })),
	matchCase: Type.Optional(Type.Boolean()),
	matchPath: Type.Optional(Type.Boolean({ description: "Match against the full path instead of the name." })),
	kind: Type.Optional(Type.Union([Type.Literal("files"), Type.Literal("folders")])),
	path: Type.Optional(Type.String({ description: "Absolute folder to search within." })),
	sort: Type.Optional(
		Type.Union([
			Type.Literal("name-ascending"),
			Type.Literal("name-descending"),
			Type.Literal("path-ascending"),
			Type.Literal("path-descending"),
			Type.Literal("size-ascending"),
			Type.Literal("size-descending"),
			Type.Literal("date-modified-ascending"),
			Type.Literal("date-modified-descending"),
		]),
	),
	offset: Type.Optional(Type.Integer({ minimum: 0, description: "Skip this many results (pagination)." })),
	maxResults: Type.Optional(
		Type.Integer({
			minimum: 1,
			description: `Maximum results (default ${DEFAULT_MAX_RESULTS}, cap ${MAX_RESULTS_CAP}).`,
		}),
	),
});

/**
 * macOS/Linux instant file/folder name search through the user's fsearch-cli
 * (FSearch) database that indexes the configured search folders. Degrades to
 * a bounded slow filesystem walk when fsearch-cli is not installed.
 */
export function createFSearchSearchTool(
	provider: FSearchSearchProvider,
): AgentTool<typeof fsearchSearchSchema, FSearchSearchDetails> {
	return {
		name: FSEARCH_SEARCH_TOOL_NAME,
		label: "FSearch Search",
		description:
			"Instantly find files and folders by name, extension, path, size, or modified date across the configured search folders using fsearch-cli/FSearch (macOS/Linux), falling back to a slow filesystem walk when fsearch-cli is not installed. Returns absolute paths; read file contents with bash.",
		parameters: fsearchSearchSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<FSearchSearchDetails>> {
			const { alternatives, ...options } = params;
			const { candidates, dropped } = collectNameCandidates(params.query, alternatives, {
				matchCase: params.matchCase,
			});
			if (candidates.length === 0) {
				return {
					content: [{ type: "text", text: "FSearch query was empty; nothing searched." }],
					details: { method: FSEARCH_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const maxResults = Math.min(params.maxResults ?? DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP);
			const searchOne = (candidate: string): Promise<FSearchSearchResult> =>
				provider.searchFsearch({ ...options, query: candidate, maxResults });
			const single = candidates.length === 1 && dropped === 0;
			const outcomes = await searchNameCandidates(candidates, searchOne);
			const failures = renderNameSearchProblems(outcomes, dropped);
			const succeeded = outcomes.flatMap((outcome) => (outcome.result.ok ? [outcome.result] : []));
			const first = outcomes[0]?.result;
			if (single && first !== undefined && !first.ok) {
				return {
					content: [{ type: "text", text: `FSearch ${first.reason}: ${first.message}` }],
					details: { method: FSEARCH_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			if (succeeded.length === 0) {
				return {
					content: [{ type: "text", text: `FSearch failed for every name:\n${failures.join("\n")}` }],
					details: { method: FSEARCH_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const merged = mergeNameHits(outcomes, maxResults);
			const lines = merged.map((hit, index) =>
				renderNameHitLine(index, hit.entry, single ? undefined : hit.candidates),
			);
			const walked = succeeded.some((result) => result.backend === "walk");
			const backendLabel = walked ? "FSearch slow filesystem walk (fsearch-cli unavailable)" : "FSearch";
			const limited = succeeded.find(
				(result) =>
					result.backend === "fsearch-cli" &&
					(result.total ?? 0) > result.results.length &&
					result.results.length === maxResults,
			);
			const limitNote =
				limited !== undefined
					? ` of ${limited.total} total (limit reached; narrow the query or page with offset)`
					: "";
			const notes = [...new Set(succeeded.flatMap((result) => (result.note !== undefined ? [result.note] : [])))];
			const noteSuffix = [...notes, ...failures].map((line) => `\n${line}`).join("");
			const names = single ? "" : ` (${candidates.length} names searched)`;
			const text =
				lines.length === 0
					? `${backendLabel} found no files or folders matching: ${candidates.join(" | ")}${noteSuffix}`
					: `${backendLabel} found ${lines.length} item(s)${names}${limitNote}:\n${lines.join("\n")}${noteSuffix}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: FSEARCH_SEARCH_TOOL_NAME,
					resultCount: merged.length,
					sources: merged.map((hit) => hit.entry.path),
					results: merged.map((hit, index) => ({ source: hit.entry.path, excerpt: lines[index]! })),
				},
			};
		},
	};
}
