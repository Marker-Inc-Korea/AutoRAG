/**
 * Mojeek's independent index served as a server-rendered HTML results page.
 * Each organic result renders as `ul.results-standard > li` with the title
 * in `h2 > a.title` (href is the direct target URL) and the preview text in
 * `p.s`. The site fronts with an ALTCHA proof-of-work captcha; the
 * headless-browser fallback solves it when available, otherwise the fetch-only
 * path surfaces the wall as a provider-tagged 429.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/mojeek.ts`.
 */

import type { LoadedHtmlPage } from "../browser-page.ts";
import { browserFetch } from "../browser-page.ts";
import { formatScraperQuery, type QuerySyntax } from "../query.ts";
import type { SearchProviderId, SearchResponse, SearchSource } from "../types.ts";
import { SearchProviderError } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, normalizeSearchText, withHardTimeout } from "./utils.ts";

const MOJEEK_ORIGIN = "https://www.mojeek.de";
const MOJEEK_HOME_URL = `${MOJEEK_ORIGIN}/?arc=none&lang=en&lb=en&theme=dark`;
const MOJEEK_SEARCH_URL = `${MOJEEK_ORIGIN}/search`;
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;

interface ParsedResult {
	title: string;
	url: string;
	snippet?: string;
}

function decodeHtmlText(value: string): string {
	return value
		.replace(/<[^>]*>/g, " ")
		.replace(/&nbsp;/g, " ")
		.replace(/&amp;/g, "&")
		.replace(/&lt;/g, "<")
		.replace(/&gt;/g, ">")
		.replace(/&quot;/g, '"')
		.replace(/&#39;|&apos;/g, "'")
		.replace(/\s+/g, " ")
		.trim();
}

function normalizeResultUrl(href: string): string | undefined {
	let url: URL;
	try {
		url = new URL(href, MOJEEK_HOME_URL);
	} catch {
		return undefined;
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
	if (
		url.hostname === "mojeek.com" ||
		url.hostname.endsWith(".mojeek.com") ||
		url.hostname === "mojeek.co.uk" ||
		url.hostname.endsWith(".mojeek.co.uk") ||
		url.hostname === "mojeek.fr" ||
		url.hostname.endsWith(".mojeek.fr") ||
		url.hostname === "mojeek.de" ||
		url.hostname.endsWith(".mojeek.de")
	) {
		return undefined;
	}
	return url.href;
}

function parseHtmlResults(html: string): ParsedResult[] {
	const results: ParsedResult[] = [];
	const listRe = /<ul\b[^>]*\bclass="[^"]*\bresults-standard\b[^"]*"[^>]*>([\s\S]*?)<\/ul>/gi;
	for (const listMatch of html.matchAll(listRe)) {
		const listHtml = listMatch[1];
		const itemRe = /<li\b[^>]*>([\s\S]*?)<\/li>/gi;
		for (const itemMatch of listHtml.matchAll(itemRe)) {
			const item = itemMatch[1];
			const headingMatch = /<a\b[^>]*\bclass="[^"]*\btitle\b[^"]*"[^>]*\bhref="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i.exec(
				item,
			);
			if (!headingMatch) continue;
			const url = normalizeResultUrl(headingMatch[1]);
			if (!url) continue;
			const title = normalizeSearchText(decodeHtmlText(headingMatch[2])) ?? "";
			if (!title) continue;
			const snippetMatch = /<p\b[^>]*\bclass="[^"]*\bs\b[^"]*"[^>]*>([\s\S]*?)<\/p>/i.exec(item);
			const snippet = snippetMatch ? normalizeSearchText(decodeHtmlText(snippetMatch[1])) : undefined;
			results.push({ title, url, snippet });
		}
	}
	return results;
}

const MOJEEK_QUERY_SYNTAX: QuerySyntax = { phrases: true, negation: true, site: true };

function buildSearchUrl(params: SearchParams, numResults: number): string {
	const url = new URL(MOJEEK_SEARCH_URL);
	url.searchParams.set("q", formatScraperQuery(params.query, params.parsedQuery, MOJEEK_QUERY_SYNTAX));
	url.searchParams.set("t", String(numResults));
	url.searchParams.set("arc", "none");
	url.searchParams.set("lang", "en");
	url.searchParams.set("lb", "en");
	url.searchParams.set("theme", "dark");
	if (params.recency) url.searchParams.set("since", params.recency);
	return url.href;
}

function isRobotPage(page: LoadedHtmlPage): boolean {
	return (
		(page.html.includes("altcha-widget") ||
			page.html.includes("captcha-wrap") ||
			/sending automated queries/i.test(page.html)) &&
		!page.html.includes("results-standard")
	);
}

async function callMojeekHtml(params: SearchParams, numResults: number): Promise<string> {
	const signal = withHardTimeout(params.signal, params.timeoutMs);
	const url = buildSearchUrl(params, numResults);
	let page: LoadedHtmlPage;
	try {
		page = await browserFetch(url, {
			fetch: params.fetch,
			signal,
			timeoutMs: params.timeoutMs,
			randomizeHeaders: false,
			referer: MOJEEK_HOME_URL,
			browser: {
				homeUrl: MOJEEK_HOME_URL,
				shouldFallback: isRobotPage,
				attempts: 2,
				retryDelayMs: 1_000,
			},
		});
	} catch (error) {
		if (error instanceof SearchProviderError || params.signal?.aborted) throw error;
		if (signal.aborted) {
			throw new SearchProviderError("mojeek", "Mojeek search timed out.", 504);
		}
		const message = error instanceof Error ? error.message : String(error);
		throw new SearchProviderError("mojeek", `Mojeek search failed: ${message}`, 503);
	}

	if (isRobotPage(page)) {
		throw new SearchProviderError(
			"mojeek",
			"Mojeek blocked the request with its automated-queries wall. Mojeek rate-limits scripted searches from datacenter/shared-egress IPs; retry later or configure another provider such as Brave, Tavily, Exa, or Kagi.",
			429,
		);
	}
	if (page.status < 200 || page.status >= 300) {
		const classified = classifyProviderHttpError("mojeek", page.status, page.html);
		if (classified) throw classified;
		throw new SearchProviderError("mojeek", `Mojeek HTML error (${page.status})`, page.status);
	}
	return page.html;
}

/** Execute a Mojeek web search against the standard HTML results page. */
export async function searchMojeek(params: SearchParams): Promise<SearchResponse> {
	const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const html = await callMojeekHtml(params, numResults);
	const parsed = parseHtmlResults(html);

	const sources: SearchSource[] = [];
	const seen = new Set<string>();
	for (const result of parsed) {
		if (seen.has(result.url)) continue;
		seen.add(result.url);
		sources.push({ title: result.title, url: result.url, snippet: result.snippet });
		if (sources.length >= numResults) break;
	}

	return { provider: "mojeek", sources };
}

/** Search provider for Mojeek (independent index, no API key required). */
export class MojeekProvider extends SearchProvider {
	readonly id = "mojeek" as SearchProviderId;
	readonly label = "Mojeek";

	isAvailable(): boolean {
		return true;
	}

	override isExplicitlyAvailable(): boolean {
		return true;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchMojeek(params);
	}
}
