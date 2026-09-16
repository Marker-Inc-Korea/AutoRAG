/**
 * Kimi Web Search Provider
 *
 * Uses the Kimi Code search API to retrieve web results. This is the Kimi Code
 * membership service, distinct from the Moonshot Open Platform — it requires a
 * Kimi Code Console credential (`KIMI_SEARCH_API_KEY` or
 * `MOONSHOT_SEARCH_API_KEY`), not `MOONSHOT_API_KEY`.
 * Endpoint: POST https://api.kimi.com/coding/v1/search
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/kimi.ts`.
 * Dropped: oh-my-pi AuthStorage/OAuth credential resolution and the
 * `kimi-code` stored-credential path. AutoRAG uses env-key mode only
 * (`KIMI_SEARCH_API_KEY` or `MOONSHOT_SEARCH_API_KEY`). Dropped: the
 * `MOONSHOT_SEARCH_BASE_URL`/`KIMI_SEARCH_BASE_URL` env overrides — the
 * endpoint is fixed to the Kimi Code service URL.
 */
import { envCredential } from "../credentials.ts";
import { formatQuery, parseSearchQuery, type QuerySyntax } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults, dateToAgeSeconds } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, withHardTimeout } from "./utils.ts";

const KIMI_SEARCH_URL = "https://api.kimi.com/coding/v1/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;
const DEFAULT_TIMEOUT_SECONDS = 30;

const KIMI_QUERY_SYNTAX: QuerySyntax = {
	phrases: true,
	negation: true,
	site: true,
	inTitle: true,
	inUrl: true,
	filetype: true,
};

interface KimiSearchResult {
	site_name?: string;
	title?: string;
	url?: string;
	snippet?: string;
	content?: string;
	date?: string;
	icon?: string;
	mime?: string;
}

interface KimiSearchResponse {
	search_results?: KimiSearchResult[];
}

function asTrimmed(value: string | undefined): string | undefined {
	if (!value) return undefined;
	const trimmed = value.trim();
	return trimmed.length > 0 ? trimmed : undefined;
}

async function callKimiSearch(
	apiKey: string,
	params: {
		query: string;
		limit: number;
		signal?: AbortSignal;
		timeoutMs?: number;
		fetch?: FetchImpl;
	},
): Promise<{ response: KimiSearchResponse; requestId?: string }> {
	const fetchImpl = params.fetch ?? fetch;
	const response = await fetchImpl(KIMI_SEARCH_URL, {
		method: "POST",
		headers: {
			Accept: "application/json",
			"Content-Type": "application/json",
			Authorization: `Bearer ${apiKey}`,
		},
		body: JSON.stringify({
			text_query: params.query,
			limit: params.limit,
			enable_page_crawling: false,
			timeout_seconds: DEFAULT_TIMEOUT_SECONDS,
		}),
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await response.text();
		const classified = classifyProviderHttpError("kimi", response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError(
			"kimi",
			`Kimi search API error (${response.status}): ${errorText}`,
			response.status,
		);
	}

	const data = (await response.json()) as KimiSearchResponse;
	const requestId = response.headers.get("x-request-id") ?? response.headers.get("x-msh-request-id") ?? undefined;
	return { response: data, requestId };
}

export class KimiProvider extends SearchProvider {
	readonly id = "kimi" as const;
	readonly label = "Kimi";

	isAvailable(): boolean {
		return !!envCredential("KIMI_SEARCH_API_KEY", "MOONSHOT_SEARCH_API_KEY");
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const apiKey = envCredential("KIMI_SEARCH_API_KEY", "MOONSHOT_SEARCH_API_KEY");
		if (!apiKey) {
			throw new SearchProviderError(
				"kimi",
				"Kimi search credentials not found. Set KIMI_SEARCH_API_KEY or MOONSHOT_SEARCH_API_KEY to a Kimi Code Console key.",
			);
		}
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const query = parsed.hasDirectives ? formatQuery(parsed, KIMI_QUERY_SYNTAX) : params.query;
		const limit = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
		const { response, requestId } = await callKimiSearch(apiKey, {
			query,
			limit,
			signal: params.signal,
			timeoutMs: params.timeoutMs,
			fetch: params.fetch,
		});

		const sources: SearchSource[] = [];
		for (const result of response.search_results ?? []) {
			if (!result.url) continue;
			const publishedDate = asTrimmed(result.date);
			const snippet = asTrimmed(result.snippet) ?? asTrimmed(result.content);
			sources.push({
				title: asTrimmed(result.title) ?? result.url,
				url: result.url,
				snippet,
				publishedDate,
				ageSeconds: dateToAgeSeconds(publishedDate),
				author: asTrimmed(result.site_name),
			});
		}

		return {
			provider: "kimi",
			sources: sources.slice(0, limit),
			requestId,
		};
	}
}
