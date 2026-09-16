/**
 * Brave Web Search Provider
 *
 * Calls Brave's web search REST API and maps results into the unified
 * SearchResponse shape used by the web search tool.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/brave.ts`.
 * Dropped: oh-my-pi AuthStorage/OAuth credential resolution — AutoRAG uses
 * `envCredential("BRAVE_API_KEY")` only.
 */
import { envCredential } from "../credentials.ts";
import {
	formatQuery,
	GOOGLE_QUERY_SYNTAX,
	parseSearchQuery,
	type QuerySyntax,
	type StructuredQuery,
} from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults, dateToAgeSeconds } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, readLimitedText, withHardTimeout } from "./utils.ts";

const BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;
const MAX_QUERY_CHARACTERS = 500;
const MAX_RESPONSE_BYTES = 2 * 1024 * 1024;
const MAX_ERROR_BYTES = 8 * 1024;

const RECENCY_MAP: Record<"day" | "week" | "month" | "year", "pd" | "pw" | "pm" | "py"> = {
	day: "pd",
	week: "pw",
	month: "pm",
	year: "py",
};

const BRAVE_QUERY_SYNTAX: QuerySyntax = { ...GOOGLE_QUERY_SYNTAX, dateRange: false };

function braveFreshness(parsed: StructuredQuery, recency?: "day" | "week" | "month" | "year"): string | undefined {
	if (parsed.after || parsed.before) {
		const start = parsed.after ?? "1970-01-01";
		const end = parsed.before ?? new Date().toISOString().slice(0, 10);
		return `${start}to${end}`;
	}
	return recency ? RECENCY_MAP[recency] : undefined;
}

function normalizeText(value: unknown, maxLength: number): string | undefined {
	if (typeof value !== "string") return undefined;
	const text = value
		.replace(/<[^>]*>/g, " ")
		.replace(/\s+/g, " ")
		.trim();
	if (!text) return undefined;
	return text.length <= maxLength ? text : `${text.slice(0, maxLength - 1)}…`;
}

function normalizeUrl(value: unknown): string | undefined {
	if (typeof value !== "string" || value.length > 2048) return undefined;
	try {
		const url = new URL(value);
		if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
		return url.toString();
	} catch {
		return undefined;
	}
}

interface BraveResult {
	url?: unknown;
	title?: unknown;
	description?: unknown;
	extra_snippets?: unknown;
	age?: unknown;
}

interface BraveWebResponse {
	web?: { results?: unknown[] };
}

function webResults(response: BraveWebResponse): readonly unknown[] {
	if (typeof response.web !== "object" || response.web === null || !("results" in response.web)) return [];
	return Array.isArray(response.web.results) ? response.web.results : [];
}

function buildSnippet(result: BraveResult): string | undefined {
	const snippets = new Set<string>();
	const description = normalizeText(result.description, 8_000);
	if (description) snippets.add(description);

	const extras = result.extra_snippets;
	if (Array.isArray(extras)) {
		for (const value of extras) {
			const snippet = normalizeText(value, 8_000);
			if (snippet) snippets.add(snippet);
		}
	}

	const combined = [...snippets].join("\n");
	return combined ? (combined.length <= 8_000 ? combined : `${combined.slice(0, 7_999)}…`) : undefined;
}

async function callBraveSearch(
	apiKey: string,
	params: {
		query: string;
		numResults: number;
		recency?: "day" | "week" | "month" | "year";
		parsed: StructuredQuery;
		signal?: AbortSignal;
		timeoutMs?: number;
		fetch?: SearchParams["fetch"];
	},
): Promise<{ response: BraveWebResponse; requestId?: string }> {
	const query = params.parsed.hasDirectives ? formatQuery(params.parsed, BRAVE_QUERY_SYNTAX) : params.query;
	if (query.length > MAX_QUERY_CHARACTERS) {
		throw new SearchProviderError(
			"brave",
			`Brave search queries cannot exceed ${MAX_QUERY_CHARACTERS} characters`,
			400,
		);
	}
	const url = new URL(BRAVE_SEARCH_URL);
	url.searchParams.set("q", query);
	url.searchParams.set("count", String(params.numResults));
	url.searchParams.set("extra_snippets", "true");
	url.searchParams.set("text_decorations", "false");
	url.searchParams.set("safesearch", "moderate");
	const freshness = braveFreshness(params.parsed, params.recency);
	if (freshness) url.searchParams.set("freshness", freshness);

	const fetchImpl = params.fetch ?? fetch;
	const response = await fetchImpl(url, {
		headers: {
			Accept: "application/json",
			"X-Subscription-Token": apiKey,
		},
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await readLimitedText(response, "brave", MAX_ERROR_BYTES, true);
		const classified = classifyProviderHttpError("brave", response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError("brave", `Brave API error (${response.status}): ${errorText}`, response.status);
	}

	const raw = await readLimitedText(response, "brave", MAX_RESPONSE_BYTES);
	let data: BraveWebResponse;
	try {
		data = JSON.parse(raw) as BraveWebResponse;
	} catch {
		throw new SearchProviderError("brave", "Brave API returned invalid JSON", 500);
	}
	const requestId = response.headers.get("x-request-id") ?? response.headers.get("request-id") ?? undefined;
	return { response: data, requestId };
}

export class BraveProvider extends SearchProvider {
	readonly id = "brave" as const;
	readonly label = "Brave";

	isAvailable(): boolean {
		return !!envCredential("BRAVE_API_KEY");
	}

	async search(params: SearchParams): Promise<SearchResponse> {
		const apiKey = envCredential("BRAVE_API_KEY");
		if (!apiKey) {
			throw new SearchProviderError(
				"brave",
				'Brave credentials not found. Set BRAVE_API_KEY or configure an API key for provider "brave".',
			);
		}
		const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
		const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
		const { response, requestId } = await callBraveSearch(apiKey, {
			query: params.query,
			numResults,
			recency: params.recency,
			parsed,
			signal: params.signal,
			timeoutMs: params.timeoutMs,
			fetch: params.fetch,
		});

		const sources: SearchSource[] = [];
		for (const raw of webResults(response)) {
			if (typeof raw !== "object" || raw === null) continue;
			const result = raw as BraveResult;
			const url = normalizeUrl(result.url);
			if (!url) continue;
			const publishedDate = normalizeText(result.age, 100);
			sources.push({
				title: normalizeText(result.title, 300) ?? url,
				url,
				snippet: buildSnippet(result),
				publishedDate,
				ageSeconds: dateToAgeSeconds(publishedDate),
			});
		}

		return {
			provider: "brave",
			sources: sources.slice(0, numResults),
			requestId,
			authMode: "api_key",
		};
	}
}
