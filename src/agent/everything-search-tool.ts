import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { EverythingSearchRequest, EverythingSearchResult } from "../everything/index.ts";
import {
	collectNameCandidates,
	mergeNameHits,
	renderNameHitLine,
	renderNameSearchProblems,
	searchNameCandidates,
} from "./name-candidates.ts";

export const EVERYTHING_SEARCH_TOOL_NAME = "everything_search";

const MAX_RESULTS_CAP = 1000;
const DEFAULT_MAX_RESULTS = 100;

export interface EverythingSearchProvider {
	searchEverything(request: EverythingSearchRequest): Promise<EverythingSearchResult>;
}

export interface EverythingSearchDetails {
	readonly method: typeof EVERYTHING_SEARCH_TOOL_NAME;
	readonly resultCount: number;
	readonly sources: readonly string[];
	/** Retrieval-trace shape: each hit's absolute path plus its rendered line. */
	readonly results: readonly { readonly source: string; readonly excerpt: string }[];
}

const everythingSearchSchema = Type.Object({
	query: Type.String({
		description:
			'Everything search syntax over file and folder names: space = AND, | = OR, ! = NOT, "quoted phrase", wildcards (*.pdf), ext:pdf;docx, dm:thisweek, size:>10mb, parent:<folder>, path fragments like \\reports\\ when matchPath is true.',
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
	wholeWord: Type.Optional(Type.Boolean()),
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
 * Windows-only instant file/folder name search through the bundled voidtools
 * Everything instance that indexes the configured search folders.
 */
export function createEverythingSearchTool(
	provider: EverythingSearchProvider,
): AgentTool<typeof everythingSearchSchema, EverythingSearchDetails> {
	return {
		name: EVERYTHING_SEARCH_TOOL_NAME,
		label: "Everything Search",
		description:
			"Instantly find files and folders by name, extension, path, size, or modified date across the configured search folders using voidtools Everything (Windows). Returns absolute paths; read file contents with bash.",
		parameters: everythingSearchSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<EverythingSearchDetails>> {
			const { alternatives, ...options } = params;
			const { candidates, dropped } = collectNameCandidates(params.query, alternatives, {
				matchCase: params.matchCase,
			});
			if (candidates.length === 0) {
				return {
					content: [{ type: "text", text: "Everything query was empty; nothing searched." }],
					details: { method: EVERYTHING_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const maxResults = Math.min(params.maxResults ?? DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP);
			const searchOne = (candidate: string): Promise<EverythingSearchResult> =>
				provider.searchEverything({ ...options, query: candidate, maxResults });
			const single = candidates.length === 1 && dropped === 0;
			const outcomes = await searchNameCandidates(candidates, searchOne);
			const failures = renderNameSearchProblems(outcomes, dropped);
			const first = outcomes[0]?.result;
			if (single && first !== undefined && !first.ok) {
				return {
					content: [{ type: "text", text: `Everything ${first.reason}: ${first.message}` }],
					details: { method: EVERYTHING_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			if (!outcomes.some((outcome) => outcome.result.ok)) {
				return {
					content: [{ type: "text", text: `Everything failed for every name:\n${failures.join("\n")}` }],
					details: { method: EVERYTHING_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const merged = mergeNameHits(outcomes, maxResults);
			const lines = merged.map((hit, index) =>
				renderNameHitLine(index, hit.entry, single ? undefined : hit.candidates),
			);
			const limitReached = outcomes.some(
				(outcome) => outcome.result.ok && outcome.result.results.length === maxResults,
			);
			const names = single ? "" : ` (${candidates.length} names searched)`;
			const problems = failures.map((line) => `\n${line}`).join("");
			const text =
				lines.length === 0
					? `Everything found no files or folders matching: ${candidates.join(" | ")}${problems}`
					: `Everything found ${lines.length} item(s)${names}${limitReached ? " (limit reached; narrow the query or page with offset)" : ""}:\n${lines.join("\n")}${problems}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: EVERYTHING_SEARCH_TOOL_NAME,
					resultCount: merged.length,
					sources: merged.map((hit) => hit.entry.path),
					results: merged.map((hit, index) => ({ source: hit.entry.path, excerpt: lines[index]! })),
				},
			};
		},
	};
}
