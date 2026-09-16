/**
 * Tavily Web Search Provider
 *
 * Uses Tavily's agent-focused search API to return structured results with an
 * optional synthesized answer.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/tavily.ts`.
 * Dropped: oh-my-pi AuthStorage/OAuth credential resolution — AutoRAG uses
 * `envCredential("TAVILY_API_KEY")` only.
 */
import { envCredential } from "../credentials.ts";
import { formatQuery, parseSearchQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults, dateToAgeSeconds } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, withHardTimeout } from "./utils.ts";

const TAVILY_SEARCH_URL = "https://api.tavily.com/search";
const DEFAULT_NUM_RESULTS = 5;
const MAX_NUM_RESULTS = 20;

interface TavilySearchParams {
	query: string;
	num_results?: number;
	recency?: "day" | "week" | "month" | "year";
	include_domains?: string[];
	exclude_domains?: string[];
	start_date?: string;
	end_date?: string;
	signal?: AbortSignal;
	timeoutMs?: number;
	fetch?: FetchImpl;
}

interface TavilyResult {
	title?: string;
	url?: string;
	content?: string;
	published_date?: string;
}

interface TavilySearchResponse {
	answer?: unknown;
	results?: unknown[];
	request_id?: string;
}

function asRecord(value: unknown): Record<string, unknown> | null {
	if (typeof value !== "object" || value === null) return null;
	return value as Record<string, unknown>;
}

function getErrorMessage(value: unknown): string | null {
	if (typeof value === "string") {
		const trimmed = value.trim();
		return trimmed.length > 0 ? trimmed : null;
	}
	const record = asRecord(value);
	if (!record) return null;
	for (const key of ["detail", "error", "message"]) {
		const message = getErrorMessage(record[key]);
		if (message) return message;
	}
	return null;
}

export function buildRequestBody(params: TavilySearchParams): Record<string, unknown> {
	const numResults = clampNumResults(params.num_results, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const body: Record<string, unknown> = {
		query: params.query,
		search_depth: "basic",
		max_results: numResults,
		include_answer: "advanced",
		include_raw_content: false,
	};
	if (params.include_domains?.length) body.include_domains = params.include_domains;
	if (params.exclude_domains?.length) body.exclude_domains = params.exclude_domains;
	if (params.start_date) body.start_date = params.start_date;
	if (params.end_date) body.end_date = params.end_date;
	if (params.recency && !params.start_date && !params.end_date) body.time_range = params.recency;
	return body;
}

async function callTavilySearch(apiKey: string, params: TavilySearchParams): Promise<TavilySearchResponse> {
	const response = await (params.fetch ?? fetch)(TAVILY_SEARCH_URL, {
		method: "POST",
		headers: {
			"Content-Type": "application/json",
			Authorization: `Bearer ${apiKey}`,
		},
		body: JSON.stringify(buildRequestBody(params)),
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await response.text();
		const classified = classifyProviderHttpError("tavily", response.status, errorText);
		if (classified) throw classified;
		let message = errorText.trim();
		if (message.length === 0) {
			message = response.statusText;
		} else {
			try {
				message = getErrorMessage(JSON.parse(errorText)) ?? message;
			} catch {
				// keep raw text fallback
			}
		}
		throw new SearchProviderError("tavily", `Tavily API error (${response.status}): ${message}`, response.status);
	}

	const payload: unknown = await response.json();
	return asRecord(payload) ?? {};
}

function toSearchResponse(response: TavilySearchResponse, numResults: number): SearchResponse {
	const sources: SearchSource[] = [];
	if (Array.isArray(response.results)) {
		for (const value of response.results) {
			const result = asRecord(value) as TavilyResult | null;
			if (!result || typeof result.url !== "string" || !result.url) continue;
			const title = typeof result.title === "string" && result.title ? result.title : result.url;
			const snippet = typeof result.content === "string" ? result.content : undefined;
			const publishedDate = typeof result.published_date === "string" ? result.published_date : undefined;
			sources.push({ title, url: result.url, snippet, publishedDate, ageSeconds: dateToAgeSeconds(publishedDate) });
		}
	}
	const answer = typeof response.answer === "string" ? response.answer.trim() || undefined : undefined;
	return {
		provider: "tavily",
		answer,
		sources: sources.slice(0, numResults),
		requestId: typeof response.request_id === "string" ? response.request_id : undefined,
		authMode: "api_key",
	};
}

function hasRenderableResponse(response: SearchResponse): boolean {
	if (response.answer?.trim()) return true;
	return response.sources.length > 0;
}

function siteHosts(sites: readonly string[]): string[] {
	const hosts = new Set<string>();
	for (const site of sites) {
		const host = site.split("/", 1)[0];
		if (host) hosts.add(host);
	}
	return [...hosts];
}

export async function searchTavily(params: SearchParams): Promise<SearchResponse> {
	const apiKey = envCredential("TAVILY_API_KEY");
	if (!apiKey) {
		throw new SearchProviderError(
			"tavily",
			'Tavily credentials not found. Set TAVILY_API_KEY or configure an API key for provider "tavily".',
		);
	}
	const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
	const tavilyParams: TavilySearchParams = {
		query: params.query,
		num_results: params.numSearchResults ?? params.limit,
		recency: params.recency,
		signal: params.signal,
		timeoutMs: params.timeoutMs,
		fetch: params.fetch,
	};
	if (parsed.hasDirectives) {
		tavilyParams.query = formatQuery(parsed, { phrases: true, negation: true });
		const include = siteHosts(parsed.sites);
		const exclude = siteHosts(parsed.excludedSites);
		if (include.length > 0) tavilyParams.include_domains = include;
		if (exclude.length > 0) tavilyParams.exclude_domains = exclude;
		if (parsed.after) tavilyParams.start_date = parsed.after;
		if (parsed.before) tavilyParams.end_date = parsed.before;
	}

	const numResults = clampNumResults(tavilyParams.num_results, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const callWithAuth = (searchParams: TavilySearchParams) => callTavilySearch(apiKey, searchParams);

	const response = toSearchResponse(await callWithAuth(tavilyParams), numResults);
	const hasTimeFilter = Boolean(tavilyParams.recency || tavilyParams.start_date || tavilyParams.end_date);
	if (!hasTimeFilter || hasRenderableResponse(response)) return response;

	return toSearchResponse(
		await callWithAuth({ ...tavilyParams, recency: undefined, start_date: undefined, end_date: undefined }),
		numResults,
	);
}

export class TavilyProvider extends SearchProvider {
	readonly id = "tavily" as const;
	readonly label = "Tavily";

	isAvailable(): boolean {
		return !!envCredential("TAVILY_API_KEY");
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchTavily(params);
	}
}
