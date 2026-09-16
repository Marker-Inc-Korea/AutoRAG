/**
 * Jina Reader Web Search Provider
 *
 * Uses the Jina Reader `s.jina.ai` endpoint to fetch search results with
 * cleaned content.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/jina.ts`.
 * Dropped: oh-my-pi AuthStorage/OAuth credential resolution — AutoRAG uses
 * `envCredential("JINA_API_KEY")` only.
 */
import { envCredential } from "../credentials.ts";
import { formatQuery, parseSearchQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, withHardTimeout } from "./utils.ts";

const JINA_SEARCH_URL = "https://s.jina.ai";
const DEFAULT_NUM_RESULTS = 5;
const MAX_NUM_RESULTS = 20;

interface JinaSearchResult {
	title?: string | null;
	url?: string | null;
	description?: string | null;
	content?: string | null;
}

interface JinaSearchEnvelope {
	code?: unknown;
	data?: unknown;
}

type JinaSearchResponse = JinaSearchResult[];

async function callJinaSearch(
	apiKey: string,
	query: string,
	numResults: number,
	site: string | undefined,
	signal: AbortSignal | undefined,
	fetchImpl: FetchImpl,
	timeoutMs?: number,
): Promise<JinaSearchResponse> {
	const requestUrl = new URL(`${JINA_SEARCH_URL}/${encodeURIComponent(query)}`);
	requestUrl.searchParams.set("count", String(numResults));

	const headers: Record<string, string> = {
		Accept: "application/json",
		Authorization: `Bearer ${apiKey}`,
		"X-Respond-With": "no-content",
		"X-Retain-Images": "none",
	};
	if (site) headers["X-Site"] = site;

	const response = await fetchImpl(requestUrl, {
		headers,
		signal: withHardTimeout(signal, timeoutMs),
	});

	if (!response.ok) {
		const errorText = await response.text();
		const classified = classifyProviderHttpError("jina", response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError("jina", `Jina API error (${response.status}): ${errorText}`, response.status);
	}

	const payload = (await response.json()) as JinaSearchEnvelope | JinaSearchResponse | null;
	if (Array.isArray(payload)) return payload;
	if (!payload || typeof payload !== "object") {
		throw new SearchProviderError("jina", "Jina API returned invalid response: expected an object or array");
	}
	if (typeof payload.code === "number" && payload.code !== 200) {
		throw new SearchProviderError("jina", `Jina API response reported failure (${payload.code})`, payload.code);
	}
	if (!Array.isArray(payload.data)) {
		throw new SearchProviderError("jina", "Jina API returned invalid response: expected data array");
	}
	return payload.data as JinaSearchResponse;
}

export class JinaProvider extends SearchProvider {
	readonly id = "jina" as const;
	readonly label = "Jina";

	isAvailable(): boolean {
		return !!envCredential("JINA_API_KEY");
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const apiKey = envCredential("JINA_API_KEY");
		if (!apiKey) {
			throw new SearchProviderError(
				"jina",
				'Jina credentials not found. Set JINA_API_KEY or configure an API key for provider "jina".',
			);
		}
		const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		let query = params.query;
		let site: string | undefined;
		if (parsed.hasDirectives) {
			if (parsed.sites.length === 1) site = parsed.sites[0]!.split("/")[0];
			query = formatQuery(parsed, {
				phrases: true,
				negation: true,
				site: !site,
				inTitle: true,
				inUrl: true,
				filetype: true,
			});
		}

		const response = await callJinaSearch(
			apiKey,
			query,
			numResults,
			site,
			params.signal,
			params.fetch ?? fetch,
			params.timeoutMs,
		);

		const sources: SearchSource[] = [];
		for (const result of response) {
			if (!result?.url) continue;
			sources.push({
				title: result.title ?? result.url,
				url: result.url,
				snippet: result.description?.trim() || result.content?.trim() || undefined,
			});
		}

		return {
			provider: "jina",
			sources: sources.slice(0, numResults),
		};
	}
}
