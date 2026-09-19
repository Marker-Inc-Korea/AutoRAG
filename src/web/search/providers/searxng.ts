/**
 * SearXNG Web Search Provider
 *
 * Calls a SearXNG instance's JSON search API and maps results into the
 * unified SearchResponse shape. Supports bearer token and RFC 7617 Basic
 * authentication. Configuration via environment variables:
 *   SEARXNG_ENDPOINT        — Base URL (required)
 *   SEARXNG_TOKEN           — Optional bearer token
 *   SEARXNG_BASIC_USERNAME  — Optional Basic auth username
 *   SEARXNG_BASIC_PASSWORD  — Optional Basic auth password
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/searxng.ts`.
 * Dropped: the settings-backed configuration paths (`settings.get(...)`
 * overrides for endpoint, token, basic auth, categories, safesearch, engines,
 * language) — AutoRAG uses environment variables only. Engine shortcut
 * resolution via `/config` is kept since it is endpoint-driven, not
 * settings-driven.
 */
import { envCredential } from "../credentials.ts";
import { formatScraperQuery, parseSearchQuery, type StructuredQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse, type SearchSource } from "../types.ts";
import { clampNumResults, dateToAgeSeconds } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, withHardTimeout } from "./utils.ts";

const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;

const RECENCY_MAP: Record<"day" | "week" | "month" | "year", string> = {
	day: "day",
	week: "month",
	month: "month",
	year: "year",
};

interface SearXNGResult {
	title?: string;
	url?: string;
	content?: string;
	snippet?: string;
	engine?: string;
	publishedDate?: string;
	published_date?: string;
	score?: number;
}

interface SearXNGResponse {
	query?: string;
	number_of_results?: number;
	results?: SearXNGResult[];
	suggestions?: string[];
	corrections?: string[];
	unresponsive_engines?: Array<[string, string]>;
	answers?: unknown[];
}

interface SearXNGAuth {
	type: "basic" | "bearer";
	value: string;
}

interface SearXNGConfig {
	engines?: Array<{ name?: string; shortcut?: string }>;
}

function findEndpoint(): string | undefined {
	return envCredential("SEARXNG_ENDPOINT");
}

function findEngines(): string | undefined {
	return envCredential("SEARXNG_ENGINES");
}

function findCategories(): string | undefined {
	return envCredential("SEARXNG_CATEGORIES");
}

function findSafesearch(): 0 | 1 | 2 | undefined {
	const raw = envCredential("SEARXNG_SAFESARCH");
	if (raw === undefined) return undefined;
	const value = Number(raw);
	if (value !== 0 && value !== 1 && value !== 2) {
		throw new SearchProviderError("searxng", "SEARXNG_SAFESARCH must be 0 (off), 1 (moderate), or 2 (strict).");
	}
	return value;
}

function findToken(): string | undefined {
	return envCredential("SEARXNG_TOKEN");
}

function findBasicUsername(): string | undefined {
	return envCredential("SEARXNG_BASIC_USERNAME");
}

function findBasicPassword(): string | undefined {
	return envCredential("SEARXNG_BASIC_PASSWORD");
}

function hasControlCharacters(value: string): boolean {
	return /[\u0000-\u001F\u007F-\u009F]/u.test(value);
}

function findAuth(): SearXNGAuth | null {
	const basicUsername = findBasicUsername();
	const basicPassword = findBasicPassword();
	if (basicUsername !== undefined || basicPassword !== undefined) {
		if (basicUsername === undefined || basicPassword === undefined) {
			throw new SearchProviderError(
				"searxng",
				"SearXNG Basic auth requires both SEARXNG_BASIC_USERNAME and SEARXNG_BASIC_PASSWORD.",
			);
		}
		if (basicUsername.includes(":")) {
			throw new SearchProviderError(
				"searxng",
				"SearXNG Basic auth username cannot contain ':' because RFC 7617 uses it as the separator.",
			);
		}
		if (hasControlCharacters(basicUsername) || hasControlCharacters(basicPassword)) {
			throw new SearchProviderError(
				"searxng",
				"SearXNG Basic auth credentials must not contain RFC 7617 control characters.",
			);
		}
		return { type: "basic", value: Buffer.from(`${basicUsername}:${basicPassword}`, "utf-8").toString("base64") };
	}
	const token = findToken();
	return token ? { type: "bearer", value: token } : null;
}

function buildHeaders(auth: SearXNGAuth | null): Record<string, string> {
	const headers: Record<string, string> = { Accept: "application/json" };
	if (auth?.type === "basic") headers.Authorization = `Basic ${auth.value}`;
	else if (auth?.type === "bearer") headers.Authorization = `Bearer ${auth.value}`;
	return headers;
}

const engineNameMapCache = new Map<string, Promise<Map<string, string> | null>>();

async function fetchEngineNameMap(
	base: string,
	auth: SearXNGAuth | null,
	fetchImpl: FetchImpl | undefined,
	signal: AbortSignal | undefined,
	timeoutMs?: number,
): Promise<Map<string, string> | null> {
	try {
		const response = await (fetchImpl ?? fetch)(`${base}/config`, {
			headers: buildHeaders(auth),
			signal: withHardTimeout(signal, timeoutMs),
		});
		if (!response.ok) return null;
		const config = (await response.json()) as SearXNGConfig;
		const map = new Map<string, string>();
		for (const engine of config.engines ?? []) {
			if (!engine.name) continue;
			map.set(engine.name.toLowerCase(), engine.name);
			if (engine.shortcut) map.set(engine.shortcut.toLowerCase(), engine.name);
		}
		return map.size ? map : null;
	} catch {
		return null;
	}
}

function getEngineNameMap(
	endpoint: string,
	auth: SearXNGAuth | null,
	fetchImpl: FetchImpl | undefined,
	signal: AbortSignal | undefined,
	timeoutMs?: number,
): Promise<Map<string, string> | null> {
	const base = endpoint.replace(/\/+$/, "");
	let cached = engineNameMapCache.get(base);
	if (!cached) {
		cached = fetchEngineNameMap(base, auth, fetchImpl, signal, timeoutMs).then((map) => {
			if (!map) engineNameMapCache.delete(base);
			return map;
		});
		engineNameMapCache.set(base, cached);
	}
	return cached;
}

async function resolveEngineNames(
	raw: string,
	endpoint: string,
	auth: SearXNGAuth | null,
	fetchImpl: FetchImpl | undefined,
	signal: AbortSignal | undefined,
	timeoutMs?: number,
): Promise<string | undefined> {
	const entries = raw
		.split(",")
		.map((entry) => entry.trim())
		.filter(Boolean);
	if (!entries.length) return undefined;
	const map = await getEngineNameMap(endpoint, auth, fetchImpl, signal, timeoutMs);
	if (!map) return entries.join(",");
	return entries.map((entry) => map.get(entry.toLowerCase()) ?? entry).join(",");
}

function stripExternalBangs(query: string): string {
	return query
		.split(/\s+/)
		.filter((part) => !part.startsWith("!!"))
		.join(" ");
}

function extractAnswerText(answer: unknown): string | undefined {
	if (typeof answer === "string") return answer.trim() || undefined;
	if (!answer || typeof answer !== "object") return undefined;
	const record = answer as Record<string, unknown>;
	if (typeof record.answer === "string") return record.answer.trim() || undefined;

	if (Array.isArray(record.translations)) {
		const translations: string[] = [];
		for (const item of record.translations) {
			if (!item || typeof item !== "object") continue;
			const text = (item as Record<string, unknown>).text;
			if (typeof text === "string" && text.trim()) translations.push(text.trim());
			if (translations.length === 3) break;
		}
		if (translations.length) return translations.join("\n");
	}

	if (record.current && typeof record.current === "object") {
		const current = record.current as Record<string, unknown>;
		if (typeof current.summary === "string" && current.summary.trim()) return current.summary.trim();
		const location =
			current.location && typeof current.location === "object"
				? (current.location as Record<string, unknown>).name
				: undefined;
		const temperature =
			current.temperature && typeof current.temperature === "object"
				? (current.temperature as Record<string, unknown>)
				: undefined;
		const temperatureText =
			temperature && (typeof temperature.val === "string" || typeof temperature.val === "number")
				? `${temperature.val}${typeof temperature.unit === "string" ? temperature.unit : ""}`
				: undefined;
		const condition = typeof current.condition === "string" ? current.condition : undefined;
		const parts = [location, temperatureText, condition].filter(
			(part): part is string => typeof part === "string" && part.trim().length > 0,
		);
		if (parts.length) return parts.join(": ");
	}

	return undefined;
}

function formatAnswers(answers: unknown[] | undefined): string | undefined {
	const texts: string[] = [];
	for (const answer of answers ?? []) {
		const text = extractAnswerText(answer);
		if (text) texts.push(text);
		if (texts.length === 3) break;
	}
	return texts.length ? texts.join("\n\n") : undefined;
}

function buildRequest(
	endpoint: string,
	params: {
		query: string;
		num_results?: number;
		recency?: "day" | "week" | "month" | "year";
		categories?: string;
		engines?: string;
		language?: string;
		safesearch?: 0 | 1 | 2;
		signal?: AbortSignal;
	},
	auth: SearXNGAuth | null,
): { url: URL; headers: Record<string, string> } {
	const base = endpoint.replace(/\/+$/, "");
	const url = new URL(`${base}/search`);
	url.searchParams.set("q", params.query);
	url.searchParams.set("format", "json");
	if (params.num_results) url.searchParams.set("pageno", "1");
	if (params.recency) url.searchParams.set("time_range", RECENCY_MAP[params.recency]);
	if (params.categories) url.searchParams.set("categories", params.categories);
	if (params.engines) url.searchParams.set("engines", params.engines);
	if (params.safesearch !== undefined) url.searchParams.set("safesearch", String(params.safesearch));
	if (params.language) url.searchParams.set("language", params.language);
	return { url, headers: buildHeaders(auth) };
}

async function callSearXNGSearch(
	endpoint: string,
	params: {
		query: string;
		num_results?: number;
		recency?: "day" | "week" | "month" | "year";
		categories?: string;
		engines?: string;
		language?: string;
		safesearch?: 0 | 1 | 2;
		signal?: AbortSignal;
		timeoutMs?: number;
		fetch?: FetchImpl;
	},
	auth: SearXNGAuth | null,
): Promise<SearXNGResponse> {
	const { url, headers } = buildRequest(endpoint, params, auth);
	const response = await (params.fetch ?? fetch)(url, {
		headers,
		signal: withHardTimeout(params.signal, params.timeoutMs),
	});

	if (!response.ok) {
		const errorText = await response.text();
		const classified = classifyProviderHttpError("searxng", response.status, errorText);
		if (classified) throw classified;
		throw new SearchProviderError("searxng", `SearXNG API error (${response.status}): ${errorText}`, response.status);
	}

	return (await response.json()) as SearXNGResponse;
}

export async function searchSearXNG(params: {
	query: string;
	parsedQuery?: StructuredQuery;
	num_results?: number;
	recency?: "day" | "week" | "month" | "year";
	signal?: AbortSignal;
	timeoutMs?: number;
	fetch?: FetchImpl;
}): Promise<SearchResponse> {
	const numResults = clampNumResults(params.num_results, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);

	const endpoint = findEndpoint();
	if (!endpoint) {
		throw new SearchProviderError("searxng", "SearXNG endpoint not configured. Set SEARXNG_ENDPOINT in environment.");
	}

	const auth = findAuth();
	const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
	const query = formatScraperQuery(params.query, parsed);
	const language = parsed.lang;
	const categories = findCategories();
	const safesearch = findSafesearch();
	const configuredEngines = findEngines();
	const engines = configuredEngines
		? await resolveEngineNames(configuredEngines, endpoint, auth, params.fetch, params.signal, params.timeoutMs)
		: undefined;

	const response = await callSearXNGSearch(
		endpoint,
		{
			...params,
			query: stripExternalBangs(query),
			categories,
			engines,
			language,
			safesearch,
			fetch: params.fetch,
		},
		auth,
	);

	const sources: SearchSource[] = [];
	for (const result of response.results ?? []) {
		if (!result.url) continue;
		const publishedDate = result.publishedDate ?? result.published_date;
		sources.push({
			title: result.title ?? result.url,
			url: result.url,
			snippet: (result.content ?? result.snippet)?.trim() || undefined,
			publishedDate: publishedDate ?? undefined,
			ageSeconds: dateToAgeSeconds(publishedDate),
		});
	}

	const limitedSources = sources.slice(0, numResults);
	if (limitedSources.length === 0 && response.unresponsive_engines?.length) {
		const upstreamFailures = response.unresponsive_engines
			.map(([engine, reason]) => `${engine}: ${reason}`)
			.join("; ");
		throw new SearchProviderError(
			"searxng",
			`SearXNG returned no usable results; upstream engines failed: ${upstreamFailures}`,
			503,
		);
	}

	return {
		provider: "searxng",
		answer: formatAnswers(response.answers),
		sources: limitedSources,
		relatedQuestions: response.suggestions?.length ? response.suggestions : undefined,
	};
}

export class SearXNGProvider extends SearchProvider {
	readonly id = "searxng" as const;
	readonly label = "SearXNG";

	isAvailable(): boolean {
		return !!findEndpoint();
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchSearXNG({
			parsedQuery: params.parsedQuery,
			query: params.query,
			num_results: params.numSearchResults ?? params.limit,
			recency: params.recency,
			signal: params.signal,
			timeoutMs: params.timeoutMs,
			fetch: params.fetch,
		});
	}
}
