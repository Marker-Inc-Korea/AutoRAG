/**
 * Google's server-rendered search results page (the `udm=14` "Web" tab
 * variant), scraped via browser-profiled fetch with a headless-browser
 * fallback. Google blocks automated requests aggressively, so the provider
 * detects traffic challenges and JavaScript challenges and surfaces them as
 * provider-tagged 429s so the chain advances to the next engine.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/google.ts`.
 */

import { decodeHtmlEntities } from "../../entities.ts";
import type { LoadedHtmlPage } from "../browser-page.ts";
import { browserFetch } from "../browser-page.ts";
import { formatScraperQuery } from "../query.ts";
import type { SearchProviderId, SearchResponse, SearchSource } from "../types.ts";
import { SearchProviderError } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { normalizeSearchText, withHardTimeout } from "./utils.ts";

const GOOGLE_HOME_URL = "https://www.google.com/";
const GOOGLE_SEARCH_URL = "https://www.google.com/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;
const RESULT_RENDER_TIMEOUT_MS = 10_000;

const RECENCY_TO_GOOGLE_TBS: Record<NonNullable<SearchParams["recency"]>, string> = {
	day: "qdr:d",
	week: "qdr:w",
	month: "qdr:m",
	year: "qdr:y",
};

const GOOGLE_SNIPPET_SELECTORS: readonly string[] = [
	"[data-sncf='1'] .VwiC3b",
	".VwiC3b",
	".IsZvec",
	".BNeawe.s3v9rd",
	"[data-sncf='1']",
];

interface ParsedResult {
	title: string;
	url: string;
	snippet?: string;
}

function normalizeText(value: string | null | undefined): string {
	return normalizeSearchText(value) ?? "";
}

function decodeHtmlText(value: string): string {
	return decodeHtmlEntities(value.replace(/<[^>]*>/g, " "))
		.replace(/\s+/g, " ")
		.trim();
}

function unwrapResultUrl(href: string): string | undefined {
	let url: URL;
	try {
		url = new URL(href, GOOGLE_HOME_URL);
	} catch {
		return undefined;
	}

	if ((url.hostname === "google.com" || url.hostname === "www.google.com") && url.pathname === "/url") {
		const target = url.searchParams.get("q") || url.searchParams.get("url");
		if (!target) return undefined;
		try {
			url = new URL(target);
		} catch {
			return undefined;
		}
	}

	if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
	if (url.hostname === "google.com" || url.hostname === "www.google.com") return undefined;
	return url.href;
}

function findSnippet(blockHtml: string): string | undefined {
	for (const selector of GOOGLE_SNIPPET_SELECTORS) {
		let text = "";
		if (selector === "[data-sncf='1'] .VwiC3b") {
			const sncfMatch = /<div\b[^>]*\bdata-sncf="1"[^>]*>([\s\S]*?)<\/div>/i.exec(blockHtml);
			if (sncfMatch) {
				const inner = sncfMatch[1];
				const vwiMatch = /<div\b[^>]*\bclass="[^"]*\bVwiC3b\b[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(inner);
				if (vwiMatch) text = decodeHtmlText(vwiMatch[1]);
			}
		} else if (selector === ".VwiC3b") {
			const match = /<div\b[^>]*\bclass="[^"]*\bVwiC3b\b[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(blockHtml);
			if (match) text = decodeHtmlText(match[1]);
		} else if (selector === ".IsZvec") {
			const match = /<div\b[^>]*\bclass="[^"]*\bIsZvec\b[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(blockHtml);
			if (match) text = decodeHtmlText(match[1]);
		} else if (selector === ".BNeawe.s3v9rd") {
			const s3Match = /<div\b[^>]*\bclass="[^"]*\bBNeawe\b[^"]*\bs3v9rd\b[^"]*"[^>]*>([\s\S]*?)<\/div>/i.exec(
				blockHtml,
			);
			if (s3Match) text = decodeHtmlText(s3Match[1]);
		} else if (selector === "[data-sncf='1']") {
			const sncfMatch = /<div\b[^>]*\bdata-sncf="1"[^>]*>([\s\S]*?)<\/div>/i.exec(blockHtml);
			if (sncfMatch) text = decodeHtmlText(sncfMatch[1]);
		}
		text = text.replace(/\s*Read more$/i, "");
		if (text) return text;
	}
	return undefined;
}

function parseHtmlResults(html: string): ParsedResult[] {
	const results: ParsedResult[] = [];
	const headingRe = /<h3\b[^>]*>([\s\S]*?)<\/h3>/gi;
	for (const headingMatch of html.matchAll(headingRe)) {
		const headingHtml = headingMatch[1];
		const headingEnd = headingMatch.index + headingMatch[0].length;
		// Find the enclosing anchor by looking backwards from the h3
		const beforeHtml = html.slice(0, headingMatch.index);
		const anchorMatch = /<a\b[^>]*\bhref="([^"]+)"[^>]*>\s*$/i.exec(beforeHtml.slice(beforeHtml.lastIndexOf("<a ")));
		if (!anchorMatch) continue;
		const url = unwrapResultUrl(anchorMatch[1]);
		if (!url) continue;
		const title = normalizeText(decodeHtmlText(headingHtml));
		if (!title) continue;
		// Find the enclosing result container for snippet extraction
		const afterHtml = html.slice(headingEnd);
		const containerMatch =
			/([\s\S]*?)(?=<h3\b|<\/div>\s*<div\b[^>]*\bclass="[^"]*\b(MjjYud|tF2Cxc|Gx5Zad)\b|$)/i.exec(afterHtml);
		const blockHtml = containerMatch ? headingHtml + containerMatch[1] : headingHtml;
		const snippet = findSnippet(blockHtml);
		results.push({ title, url, snippet });
	}
	return results;
}

function buildSearchUrl(params: SearchParams, numResults: number): string {
	const url = new URL(GOOGLE_SEARCH_URL);
	url.searchParams.set("q", formatScraperQuery(params.query, params.parsedQuery));
	url.searchParams.set("num", String(numResults));
	url.searchParams.set("hl", "en");
	url.searchParams.set("gl", "us");
	url.searchParams.set("udm", "14");
	url.searchParams.set("pws", "0");
	const tbs = params.recency ? RECENCY_TO_GOOGLE_TBS[params.recency] : undefined;
	if (tbs) url.searchParams.set("tbs", tbs);
	return url.href;
}

function blockReason(page: LoadedHtmlPage): "javascript" | "traffic" | undefined {
	if (page.html.includes("/httpservice/retry/enablejs") && !/<h3\b/i.test(page.html)) return "javascript";
	if (
		page.status === 403 ||
		page.status === 429 ||
		page.url.includes("/sorry/") ||
		/unusual traffic|detected unusual traffic|g-recaptcha/i.test(page.html)
	) {
		return "traffic";
	}
	return undefined;
}

async function callGoogleHtml(params: SearchParams, numResults: number): Promise<string> {
	const signal = withHardTimeout(params.signal, params.timeoutMs);
	const url = buildSearchUrl(params, numResults);
	let page: LoadedHtmlPage;
	try {
		page = await browserFetch(url, {
			fetch: params.fetch,
			signal,
			timeoutMs: params.timeoutMs,
			referer: GOOGLE_HOME_URL,
			browser: {
				homeUrl: GOOGLE_HOME_URL,
				ready: { selector: "a h3", timeoutMs: RESULT_RENDER_TIMEOUT_MS },
				shouldFallback: (candidate) => blockReason(candidate) !== undefined,
			},
		});
	} catch (error) {
		if (error instanceof SearchProviderError || params.signal?.aborted) throw error;
		if (signal.aborted) {
			throw new SearchProviderError("google", "Google browser search timed out.", 504);
		}
		const message = error instanceof Error ? error.message : String(error);
		throw new SearchProviderError("google", `Google browser search failed: ${message}`, 503);
	}

	const blocked = blockReason(page);
	if (blocked === "traffic") {
		throw new SearchProviderError(
			"google",
			"Google blocked the browser search with an automated-traffic challenge. Try another web search provider or retry later.",
			429,
		);
	}
	if (page.status < 200 || page.status >= 300) {
		throw new SearchProviderError("google", `Google HTML error (${page.status})`, page.status);
	}
	if (blocked === "javascript") {
		throw new SearchProviderError(
			"google",
			"Google returned its JavaScript challenge instead of rendered search results.",
			429,
		);
	}
	return page.html;
}

/** Execute a Google web search with fetch-first loading and a headless-browser fallback. */
export async function searchGoogle(params: SearchParams): Promise<SearchResponse> {
	const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const html = await callGoogleHtml(params, numResults);
	const parsed = parseHtmlResults(html);

	const sources: SearchSource[] = [];
	const seen = new Set<string>();
	for (const result of parsed) {
		if (seen.has(result.url)) continue;
		seen.add(result.url);
		sources.push({ title: result.title, url: result.url, snippet: result.snippet });
		if (sources.length >= numResults) break;
	}

	return { provider: "google", sources };
}

/** Fetch-first Google Search provider with a headless-browser fallback; no API key is required. */
export class GoogleProvider extends SearchProvider {
	readonly id = "google" as SearchProviderId;
	readonly label = "Google";

	isAvailable(): boolean {
		return true;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchGoogle(params);
	}
}
