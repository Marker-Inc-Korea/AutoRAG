/**
 * Kagi Web Search Provider
 *
 * Calls the Kagi V1 Search API (POST /api/v1/search) and maps categorized
 * result buckets into the unified SearchResponse shape.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/kagi.ts` and
 * `packages/coding-agent/src/web/kagi.ts`.
 * Dropped: oh-my-pi AuthStorage/OAuth credential resolution — AutoRAG uses
 * `envCredential("KAGI_API_KEY")` only.
 */
import { envCredential } from "../credentials.ts";
import { formatQuery, GOOGLE_QUERY_SYNTAX, parseSearchQuery, type StructuredQuery } from "../query.ts";
import { SearchProviderError, type SearchResponse } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, toSearchSources, withHardTimeout } from "./utils.ts";

interface KagiSearchSource {
	title: string;
	url: string;
	snippet?: string;
	publishedDate?: string;
}

const KAGI_SEARCH_URL = "https://kagi.com/api/v1/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 40;

interface KagiSearchRequest {
	query: string;
	workflow?: string;
	limit?: number;
	lens?: string;
	filters?: { after?: string; before?: string };
}

interface KagiSearchResultItem {
	url: string;
	title: string;
	snippet?: string;
	time?: string;
	image?: { url: string; height?: number; width?: number };
	props?: Record<string, unknown>;
}

interface KagiSearchData {
	search?: KagiSearchResultItem[];
	video?: KagiSearchResultItem[];
	news?: KagiSearchResultItem[];
	infobox?: KagiSearchResultItem[];
	adjacent_question?: KagiSearchResultItem[];
	related_search?: KagiSearchResultItem[];
	direct_answer?: KagiSearchResultItem[];
}

interface KagiErrorEntry {
	code?: number;
	url?: string;
	message?: string;
	msg?: string;
	location?: string;
}

interface KagiSearchResponse {
	meta?: { trace?: string; id?: string; ms?: number };
	data?: KagiSearchData;
	error?: KagiErrorEntry[];
}

interface KagiErrorResponse {
	meta?: Record<string, unknown>;
	error?: string | KagiErrorEntry[];
	message?: string;
	detail?: string;
}

export class KagiApiError extends Error {
	readonly statusCode?: number;

	constructor(message: string, statusCode?: number) {
		super(message);
		this.name = "KagiApiError";
		this.statusCode = statusCode;
	}
}

function extractKagiErrorMessage(payload: unknown): string | null {
	if (!payload || typeof payload !== "object" || Array.isArray(payload)) return null;
	const record = payload as Record<string, unknown>;

	for (const value of [record.message, record.detail]) {
		if (typeof value === "string" && value.trim().length > 0) return value.trim();
	}

	for (const errors of [record.error, record.errors]) {
		if (typeof errors === "string" && errors.trim().length > 0) return errors.trim();
		if (!Array.isArray(errors)) continue;
		for (const entry of errors) {
			if (!entry || typeof entry !== "object") continue;
			const e = entry as Record<string, unknown>;
			for (const value of [e.message, e.msg, e.code]) {
				if (
					(typeof value === "string" && value.trim().length > 0) ||
					(typeof value === "number" && Number.isFinite(value))
				) {
					return String(value).trim();
				}
			}
		}
	}

	return null;
}

function createKagiApiError(statusCode: number, detail?: string): KagiApiError {
	return new KagiApiError(
		detail ? `Kagi API error (${statusCode}): ${detail}` : `Kagi API error (${statusCode})`,
		statusCode,
	);
}

function parseKagiErrorResponse(statusCode: number, responseText: string): KagiApiError {
	const trimmed = responseText.trim();
	if (trimmed.length === 0) return createKagiApiError(statusCode);

	try {
		const payload = JSON.parse(trimmed) as KagiErrorResponse;
		return createKagiApiError(statusCode, extractKagiErrorMessage(payload) ?? trimmed);
	} catch {
		return createKagiApiError(statusCode, trimmed);
	}
}

function parseKagiSuccessResponse(statusCode: number, responseText: string): KagiSearchResponse {
	let payload: unknown;
	try {
		payload = JSON.parse(responseText);
	} catch {
		throw new KagiApiError("Kagi API returned an invalid response: invalid JSON", statusCode);
	}
	if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
		throw new KagiApiError("Kagi API returned an invalid response: expected an object envelope", statusCode);
	}

	const record = payload as Record<string, unknown>;
	const errorMessage = extractKagiErrorMessage(payload);
	if (errorMessage && (record.error !== undefined || record.errors !== undefined)) {
		const errors = Array.isArray(record.error) ? record.error : Array.isArray(record.errors) ? record.errors : [];
		const first = errors[0];
		const code =
			first && typeof first === "object" && typeof (first as Record<string, unknown>).code === "number"
				? ((first as Record<string, unknown>).code as number)
				: statusCode;
		throw createKagiApiError(code, errorMessage);
	}
	if (record.data !== undefined && (!record.data || typeof record.data !== "object" || Array.isArray(record.data))) {
		throw new KagiApiError("Kagi API returned an invalid response: expected data to be an object", statusCode);
	}
	return payload as KagiSearchResponse;
}

function recencyToDate(recency: "day" | "week" | "month" | "year"): string {
	const d = new Date();
	switch (recency) {
		case "day":
			d.setUTCDate(d.getUTCDate() - 1);
			break;
		case "week":
			d.setUTCDate(d.getUTCDate() - 7);
			break;
		case "month":
			d.setUTCMonth(d.getUTCMonth() - 1);
			break;
		case "year":
			d.setUTCFullYear(d.getUTCFullYear() - 1);
			break;
	}
	const yyyy = d.getUTCFullYear();
	const mm = String(d.getUTCMonth() + 1).padStart(2, "0");
	const dd = String(d.getUTCDate()).padStart(2, "0");
	return `${yyyy}-${mm}-${dd}`;
}

function buildRequestBody(
	query: string,
	limit: number,
	recency?: "day" | "week" | "month" | "year",
): KagiSearchRequest {
	const req: KagiSearchRequest = { query, workflow: "search", limit };
	if (recency) req.filters = { after: recencyToDate(recency) };
	return req;
}

function firstNonEmptyString(...values: unknown[]): string | undefined {
	for (const value of values) {
		if (typeof value === "string" && value.trim().length > 0) return value.trim();
	}
	return undefined;
}

function collectSources(sources: KagiSearchSource[], items: unknown, tag?: string): void {
	if (!Array.isArray(items)) return;
	for (const value of items) {
		if (!value || typeof value !== "object" || Array.isArray(value)) continue;
		const item = value as Record<string, unknown>;
		const url = firstNonEmptyString(item.url, item.href, item.link);
		if (!url) continue;
		const title = firstNonEmptyString(item.title, item.name) ?? url;
		sources.push({
			title: tag ? `${tag} ${title}` : title,
			url,
			snippet: firstNonEmptyString(item.snippet, item.description, item.summary),
			publishedDate: firstNonEmptyString(item.time),
		});
	}
}

function questionOf(value: unknown): string | undefined {
	if (!value || typeof value !== "object" || Array.isArray(value)) return undefined;
	const item = value as Record<string, unknown>;
	const props =
		item.props && typeof item.props === "object" && !Array.isArray(item.props)
			? (item.props as Record<string, unknown>)
			: undefined;
	return firstNonEmptyString(props?.question, props?.query, item.title);
}

export interface KagiSearchResult {
	requestId: string;
	sources: KagiSearchSource[];
	relatedQuestions: string[];
	answer?: string;
}

async function searchWithKagi(
	query: string,
	options: {
		limit: number;
		recency?: "day" | "week" | "month" | "year";
		signal?: AbortSignal;
		timeoutMs?: number;
		fetch?: FetchImpl;
	},
	apiKey: string,
): Promise<KagiSearchResult> {
	const fetchImpl = options.fetch ?? fetch;
	const body = JSON.stringify(buildRequestBody(query, options.limit, options.recency));

	const response = await fetchImpl(KAGI_SEARCH_URL, {
		method: "POST",
		headers: {
			Authorization: `Bearer ${apiKey}`,
			"Content-Type": "application/json",
			Accept: "application/json",
		},
		body,
		signal: withHardTimeout(options.signal, options.timeoutMs),
	});

	if (!response.ok) {
		throw parseKagiErrorResponse(response.status, await response.text());
	}

	const payload = parseKagiSuccessResponse(response.status, await response.text());
	const data = payload.data;
	const sources: KagiSearchSource[] = [];
	const relatedQuestions: string[] = [];

	collectSources(sources, data?.search);
	collectSources(sources, data?.video, "[Video]");
	collectSources(sources, data?.news, "[News]");
	collectSources(sources, data?.infobox, "[Info]");

	const adjacentQuestions: unknown = data?.adjacent_question;
	if (Array.isArray(adjacentQuestions)) {
		for (const item of adjacentQuestions) {
			const question = questionOf(item);
			if (question) relatedQuestions.push(question);
		}
	}
	const relatedSearches: unknown = data?.related_search;
	if (Array.isArray(relatedSearches)) {
		for (const item of relatedSearches) {
			const question = questionOf(item);
			if (question) relatedQuestions.push(question);
		}
	}

	const directAnswers: unknown = data?.direct_answer;
	const directAnswer = Array.isArray(directAnswers) ? directAnswers[0] : undefined;
	const answer =
		directAnswer && typeof directAnswer === "object" && !Array.isArray(directAnswer)
			? firstNonEmptyString(
					(directAnswer as Record<string, unknown>).snippet,
					(directAnswer as Record<string, unknown>).title,
				)
			: undefined;

	return {
		requestId: payload.meta?.trace ?? payload.meta?.id ?? "",
		sources,
		relatedQuestions,
		answer,
	};
}

async function searchKagi(params: {
	query: string;
	num_results?: number;
	recency?: "day" | "week" | "month" | "year";
	parsedQuery?: StructuredQuery;
	signal?: AbortSignal;
	timeoutMs?: number;
	fetch?: FetchImpl;
}): Promise<SearchResponse> {
	const apiKey = envCredential("KAGI_API_KEY");
	if (!apiKey) {
		throw new SearchProviderError(
			"kagi",
			'Kagi credentials not found. Set KAGI_API_KEY or configure an API key for provider "kagi".',
		);
	}
	const numResults = clampNumResults(params.num_results, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const parsed = params.parsedQuery ?? parseSearchQuery(params.query);
	const query = parsed.hasDirectives ? formatQuery(parsed, GOOGLE_QUERY_SYNTAX) : params.query;

	try {
		const result = await searchWithKagi(
			query,
			{
				limit: numResults,
				recency: params.recency,
				signal: params.signal,
				timeoutMs: params.timeoutMs,
				fetch: params.fetch,
			},
			apiKey,
		);

		return {
			provider: "kagi",
			sources: toSearchSources(result.sources, numResults),
			relatedQuestions: result.relatedQuestions.length > 0 ? result.relatedQuestions : undefined,
			requestId: result.requestId,
			answer: result.answer,
		};
	} catch (err) {
		if (err instanceof KagiApiError) {
			if (typeof err.statusCode === "number") {
				const classified = classifyProviderHttpError("kagi", err.statusCode, err.message);
				if (classified) throw classified;
			}
			throw new SearchProviderError("kagi", err.message, err.statusCode);
		}
		throw err;
	}
}

export class KagiProvider extends SearchProvider {
	readonly id = "kagi" as const;
	readonly label = "Kagi";

	isAvailable(): boolean {
		return !!envCredential("KAGI_API_KEY");
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchKagi({
			query: params.query,
			parsedQuery: params.parsedQuery,
			num_results: params.numSearchResults ?? params.limit,
			recency: params.recency,
			signal: params.signal,
			timeoutMs: params.timeoutMs,
			fetch: params.fetch,
		});
	}
}
