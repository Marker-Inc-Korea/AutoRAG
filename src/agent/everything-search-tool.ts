import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type { EverythingSearchRequest, EverythingSearchResult } from "../everything/index.ts";

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
			const query = params.query.trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Everything query was empty; nothing searched." }],
					details: { method: EVERYTHING_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const request: EverythingSearchRequest = {
				...params,
				query,
				maxResults: Math.min(params.maxResults ?? DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP),
			};
			const result = await provider.searchEverything(request);
			if (!result.ok) {
				return {
					content: [{ type: "text", text: `Everything ${result.reason}: ${result.message}` }],
					details: { method: EVERYTHING_SEARCH_TOOL_NAME, resultCount: 0, sources: [], results: [] },
				};
			}
			const lines = result.results.map((entry, index) => {
				const parts = [`[${index + 1}] ${entry.type} ${entry.path}`];
				if (entry.size !== undefined) parts.push(`size=${entry.size}`);
				if (entry.dateModified !== undefined) parts.push(`modified=${entry.dateModified}`);
				return parts.join(" ");
			});
			const text =
				lines.length === 0
					? `Everything found no files or folders matching: ${query}`
					: `Everything found ${lines.length} item(s)${lines.length === request.maxResults ? " (limit reached; narrow the query or page with offset)" : ""}:\n${lines.join("\n")}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: EVERYTHING_SEARCH_TOOL_NAME,
					resultCount: result.results.length,
					sources: result.results.map((entry) => entry.path),
					results: result.results.map((entry, index) => ({ source: entry.path, excerpt: lines[index]! })),
				},
			};
		},
	};
}
