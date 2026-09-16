/**
 * Exa Web Search Provider
 *
 * High-quality neural search via the Exa Search REST API. Requests per-result
 * summaries via `contents.summary` and synthesizes them into a combined
 * `answer` string on the SearchResponse.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/exa.ts`.
 * Dropped: the keyless MCP fallback (`callExaMcpSearch`,
 * `normalizeExaMcpPayload`, `parseExaMcpTextPayload`, `buildExaMcpArgs`,
 * `EXA_MCP_URL`, `EXA_MCP_SOURCE`, `findApiKey`, `isSearchResponse`,
 * `readMcpJsonRpcResponse`, `settings`/`getDefault` integration) — AutoRAG
 * uses env-key mode only. Also dropped: the Exa rate-limit throttle
 * (`waitForExaSearchSlot`, `exaSearchThrottle`, `resetExaSearchThrottleForTest`)
 * — the caller's per-request timeout is the transport ceiling.
 */
import { envCredential } from "../credentials.ts";
import { formatQuery, parseSearchQuery, type StructuredQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { dateToAgeSeconds } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, withHardTimeout } from "./utils.ts";

const EXA_API_URL = "https://api.exa.ai/search";
const MAX_EXA_SNIPPET_CHARS = 500;
const MAX_ANSWER_SUMMARIES = 3;

type ExaSearchType = "neural" | "fast" | "auto" | "deep";

interface ExaSearchParams {
	query: string;
	num_results?: number;
	type?: ExaSearchType | "keyword";
	include_domains?: string[];
	exclude_domains?: string[];
	start_published_date?: string;
	end_published_date?: string;
	signal?: AbortSignal;
	timeoutMs?: number;
	fetch?: FetchImpl;
}

interface ExaSearchResult {
	title?: string | null;
	url?: string | null;
	author?: string | null;
	publishedDate?: string | null;
	text?: string | null;
	highlights?: string[] | null;
	summary?: string | null;
}

interface ExaSearchResponse {
	requestId?: string;
	resolvedSearchType?: string;
	results?: ExaSearchResult[];
	costDollars?: { total: number };
	searchTime?: number;
}

export function normalizeSearchType(type: ExaSearchParams["type"]): ExaSearchType {
	if (!type) return "auto";
	if (type === "keyword") return "fast";
	return type;
}

export function synthesizeAnswer(results: ExaSearchResult[]): string | undefined {
	const parts: string[] = [];
	for (const r of results) {
		if (parts.length >= MAX_ANSWER_SUMMARIES) break;
		const summary = r.summary?.trim();
		if (!summary) continue;
		const title = r.title?.trim() || r.url || "Untitled";
		parts.push(`**${title}**: ${summary}`);
	}
	return parts.length > 0 ? parts.join("\n\n") : undefined;
}

export function buildExaRequestBody(params: ExaSearchParams): Record<string, unknown> {
	const body: Record<string, unknown> = {
		query: params.query,
		numResults: params.num_results ?? 10,
		type: normalizeSearchType(params.type),
		contents: {
			summary: { query: params.query },
		},
	};
	if (params.include_domains?.length) body.includeDomains = params.include_domains;
	if (params.exclude_domains?.length) body.excludeDomains = params.exclude_domains;
	if (params.start_published_date) body.startPublishedDate = params.start_published_date;
	if (params.end_published_date) body.endPublishedDate = params.end_published_date;
	return body;
}

async function callExaSearch(apiKey: string, params: ExaSearchParams): Promise<ExaSearchResponse> {
	const body = buildExaRequestBody(params);
	const fetchImpl = params.fetch ?? fetch;
	const response = await fetchImpl(EXA_API_URL, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			"x-api-key": apiKey,
		},
		body: JSON.stringify(body),
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await response.text();
		const classified = classifyProviderHttpError("exa", response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError("exa", `Exa API error (${response.status}): ${errorText}`, response.status);
	}

	return (await response.json()) as ExaSearchResponse;
}

function directiveParams(
	parsed: StructuredQuery,
): Pick<
	ExaSearchParams,
	"query" | "include_domains" | "exclude_domains" | "start_published_date" | "end_published_date"
> {
	if (!parsed.hasDirectives) return { query: parsed.raw };
	const hosts = (sites: readonly string[]) => [...new Set(sites.map((site) => site.split("/", 1)[0]!))];
	return {
		query: formatQuery(parsed, { phrases: true }),
		include_domains: parsed.sites.length ? hosts(parsed.sites) : undefined,
		exclude_domains: parsed.excludedSites.length ? hosts(parsed.excludedSites) : undefined,
		start_published_date: parsed.after,
		end_published_date: parsed.before,
	};
}

export async function searchExa(params: ExaSearchParams): Promise<SearchResponse> {
	const apiKey = envCredential("EXA_API_KEY");
	if (!apiKey) {
		throw new SearchProviderError(
			"exa",
			'Exa credentials not found. Set EXA_API_KEY or configure an API key for provider "exa".',
		);
	}
	const response = await callExaSearch(apiKey, params);

	const sources: SearchSource[] = [];
	if (response.results) {
		for (const result of response.results) {
			if (!result.url) continue;
			sources.push({
				title: result.title ?? result.url,
				url: result.url,
				snippet: (result.summary || result.text || result.highlights?.join(" ") || undefined)?.slice(
					0,
					MAX_EXA_SNIPPET_CHARS,
				),
				publishedDate: result.publishedDate ?? undefined,
				ageSeconds: dateToAgeSeconds(result.publishedDate ?? undefined),
				author: result.author ?? undefined,
			});
		}
	}

	const limitedSources = params.num_results ? sources.slice(0, params.num_results) : sources;
	const answer = response.results ? synthesizeAnswer(response.results.filter((r) => !!r.url)) : undefined;

	return {
		provider: "exa",
		answer,
		sources: limitedSources,
		requestId: response.requestId,
	};
}

export class ExaProvider extends SearchProvider {
	readonly id = "exa" as const;
	readonly label = "Exa";

	isAvailable(): boolean {
		return !!envCredential("EXA_API_KEY");
	}

	search(params: SearchParams): Promise<SearchResponse> {
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		return searchExa({
			...directiveParams(parsed),
			num_results: params.numSearchResults ?? params.limit,
			signal: params.signal,
			timeoutMs: params.timeoutMs,
			fetch: params.fetch,
		});
	}
}
