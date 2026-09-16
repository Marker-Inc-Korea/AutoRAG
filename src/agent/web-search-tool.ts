/**
 * `web_search` agent tool — internet web search for the librarian agent.
 *
 * LLM-facing wrapper around the oh-my-pi-style provider chain in
 * `src/web/search/`: credential-free by default (DuckDuckGo/Startpage/…),
 * keyed providers activate via environment variables, and quota/auth/bot
 * failures automatically fall back down the chain. The model only supplies
 * `query` plus optional hints; provider credentials are never exposed
 * through tool arguments.
 */
import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { executeWebSearch } from "../web/search/index.ts";
import { WEB_SEARCH_TOOL_DESCRIPTION } from "../web/search/format.ts";
import { setExcludedSearchProviders, setSearchProviderOrder } from "../web/search/provider.ts";
import { isSearchProviderId, type SearchProviderId } from "../web/search/types.ts";

export const WEB_SEARCH_TOOL_NAME = "web_search";

const webSearchSchema = Type.Object({
	query: Type.String({
		description:
			'Search query. Supports Google-style directives: site:/-site:, after:/before: (YYYY-MM-DD), inurl:, intitle:, filetype:, "exact phrase", -term, OR.',
	}),
	recency: Type.Optional(
		Type.Union([Type.Literal("day"), Type.Literal("week"), Type.Literal("month"), Type.Literal("year")], {
			description: "Optional recency window for results (pure time filter).",
		}),
	),
	limit: Type.Optional(Type.Integer({ description: "Maximum number of results to return." })),
	num_search_results: Type.Optional(Type.Integer({ description: "Alias for limit (result count hint)." })),
	provider: Type.Optional(
		Type.String({
			description:
				"Optional explicit provider id (brave, tavily, exa, jina, kagi, kimi, searxng, startpage, duckduckgo, ecosia, google, mojeek, public). Default: auto chain with fallback.",
		}),
	),
});

export interface WebSearchToolOptions {
	/** Force one provider (default: walk the configured chain). */
	readonly provider?: SearchProviderId;
	/** Prioritize these providers; unlisted providers keep their built-in relative order. */
	readonly order?: readonly SearchProviderId[];
	/** Providers never used by web search, including fallbacks. */
	readonly exclude?: readonly SearchProviderId[];
	/** Per-provider transport hard timeout in seconds (default 60, max 300). */
	readonly timeoutSeconds?: number;
}

export interface WebSearchToolDetails {
	readonly method: "web_search";
	readonly provider: string;
	readonly resultCount: number;
	readonly sources: readonly string[];
	readonly available: boolean;
	readonly error?: string;
}

export function createWebSearchTool(
	options: WebSearchToolOptions = {},
): AgentTool<typeof webSearchSchema, WebSearchToolDetails> {
	if (options.order && options.order.length > 0) setSearchProviderOrder(options.order);
	if (options.exclude && options.exclude.length > 0) setExcludedSearchProviders(options.exclude);

	return {
		name: WEB_SEARCH_TOOL_NAME,
		label: "Web Search",
		description: WEB_SEARCH_TOOL_DESCRIPTION,
		parameters: webSearchSchema,
		async execute(_toolCallId, params, signal): Promise<AgentToolResult<WebSearchToolDetails>> {
			if (params.query.trim().length === 0) {
				return {
					content: [{ type: "text", text: "Web search query was empty; nothing searched." }],
					details: { method: WEB_SEARCH_TOOL_NAME, provider: "none", resultCount: 0, sources: [], available: true },
				};
			}
			const forcedProvider =
				options.provider ?? (params.provider && isSearchProviderId(params.provider) ? params.provider : undefined);
			const result = await executeWebSearch(
				{
					query: params.query,
					recency: params.recency,
					limit: params.limit,
					num_search_results: params.num_search_results,
					provider: forcedProvider,
				},
				{
					signal,
					timeoutMs: options.timeoutSeconds !== undefined ? options.timeoutSeconds * 1_000 : undefined,
				},
			);
			const response = result.details.response;
			const sources = [...new Set(response.sources.map((source) => source.url))];
			if (result.details.error !== undefined) {
				return {
					content: [
						{
							type: "text",
							text: `Web search is currently unavailable: ${result.details.error} Try rephrasing the query, selecting another provider, or answering from local documents instead.`,
						},
					],
					details: {
						method: WEB_SEARCH_TOOL_NAME,
						provider: response.provider,
						resultCount: 0,
						sources: [],
						available: false,
						error: result.details.error,
					},
				};
			}
			return {
				content: result.content,
				details: {
					method: WEB_SEARCH_TOOL_NAME,
					provider: response.provider,
					resultCount: response.sources.length,
					sources,
					available: true,
				},
			};
		},
	};
}
