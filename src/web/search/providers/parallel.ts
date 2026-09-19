/**
 * Parallel web search provider — keyless public MCP endpoint.
 *
 * Parallel exposes its web search as a public MCP server
 * (`https://search.parallel.ai/mcp`, JSON-RPC `tools/call` `web_search`)
 * that needs no key or signup, so this provider is always available.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/parallel.ts`. Dropped: oh-my-pi's keyed REST beta
 * path and AuthStorage wiring — AutoRAG ships the keyless MCP route only.
 */
import { formatQuery, parseSearchQuery, type QuerySyntax } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { dateToAgeSeconds } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const PARALLEL_MCP_URL = "https://search.parallel.ai/mcp";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 40;
const MAX_RESPONSE_BYTES = 4 * 1024 * 1024;
const MAX_ERROR_BYTES = 8 * 1024;

/** The public MCP accepts search operators in `search_queries` (natural-language objective). */
const PARALLEL_MCP_QUERY_SYNTAX: QuerySyntax = { phrases: true, negation: true, or: true, site: true, dateRange: true };

interface ParallelMcpResult {
	title?: string;
	url?: string;
	excerpts?: string[];
	publish_date?: string;
}

interface ParallelMcpToolResult {
	structuredContent?: { results?: ParallelMcpResult[] };
	content?: Array<{ type?: string; text?: string }>;
	isError?: boolean;
}

interface JsonRpcResponse {
	jsonrpc?: string;
	id?: number | string;
	result?: ParallelMcpToolResult;
	error?: { code?: number; message?: string };
}

function toSources(results: readonly ParallelMcpResult[]): SearchSource[] {
	const sources: SearchSource[] = [];
	for (const result of results) {
		if (typeof result.url !== "string" || result.url.length === 0) continue;
		sources.push({
			title: result.title ?? result.url,
			url: result.url,
			...(result.excerpts?.length ? { snippet: result.excerpts.join("\n") } : {}),
			...(result.publish_date !== undefined ? { publishedDate: result.publish_date } : {}),
			...(dateToAgeSeconds(result.publish_date) !== undefined
				? { ageSeconds: dateToAgeSeconds(result.publish_date) }
				: {}),
		});
	}
	return sources;
}

export class ParallelProvider extends SearchProvider {
	readonly id = "parallel" as const;
	readonly label = "Parallel";

	/** The public MCP endpoint needs no credential of any kind. */
	isAvailable(): boolean {
		return true;
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const numResults = Math.min(
			Math.max(params.numSearchResults ?? params.limit ?? DEFAULT_NUM_RESULTS, 1),
			MAX_NUM_RESULTS,
		);
		const objective = params.query;
		const queries = parsed.hasDirectives ? [formatQuery(parsed, PARALLEL_MCP_QUERY_SYNTAX)] : [params.query];

		const fetchImpl = params.fetch ?? fetch;
		const response = await fetchImpl(PARALLEL_MCP_URL, {
			method: "POST",
			headers: {
				"content-type": "application/json",
				accept: "application/json, text/event-stream",
			},
			body: JSON.stringify({
				jsonrpc: "2.0",
				id: 1,
				method: "tools/call",
				params: {
					name: "web_search",
					arguments: {
						objective,
						search_queries: queries,
						max_results: numResults,
					},
				},
			}),
			signal: withHardTimeout(params.signal, params.timeoutMs),
		});

		if (!response.ok) {
			const errorText = await readLimitedText(response, "parallel", MAX_ERROR_BYTES, true);
			const classified = classifyProviderHttpError("parallel", response.status, errorText);
			if (classified) throw classified;
			throw new SearchProviderError(
				"parallel",
				`Parallel MCP error (${response.status}): ${errorText}`,
				response.status,
			);
		}

		const raw = await readLimitedText(response, "parallel", MAX_RESPONSE_BYTES);
		// MCP over HTTP may answer with a bare JSON-RPC response or an SSE
		// stream carrying one; handle both.
		let message: JsonRpcResponse;
		try {
			const dataLine = raw
				.split("\n")
				.map((line) => line.trim())
				.find((line) => line.startsWith("data:"));
			message = JSON.parse(dataLine ? dataLine.slice(5).trim() : raw) as JsonRpcResponse;
		} catch {
			throw new SearchProviderError("parallel", "Parallel MCP returned an unreadable response", 500);
		}
		if (message.error) {
			throw new SearchProviderError(
				"parallel",
				`Parallel MCP error (${message.error.code ?? "unknown"}): ${message.error.message ?? "Unknown error"}`,
				500,
			);
		}

		const result = message.result;
		if (result?.isError) {
			const text = result.content?.find((part) => part.type === "text")?.text ?? "tool error";
			throw new SearchProviderError("parallel", `Parallel MCP tool error: ${text}`, 502);
		}

		let sources = toSources(result?.structuredContent?.results ?? []);
		if (sources.length === 0) {
			// Fallback: some MCP servers return JSON-encoded text content.
			const text = result?.content?.find((part) => part.type === "text")?.text;
			if (text) {
				try {
					const parsedText = JSON.parse(text) as { results?: ParallelMcpResult[] };
					sources = toSources(parsedText.results ?? []);
				} catch {
					// Plain-text content carries no structured results.
				}
			}
		}

		return {
			provider: "parallel",
			sources: sources.slice(0, numResults),
			authMode: "keyless",
		};
	}
}
