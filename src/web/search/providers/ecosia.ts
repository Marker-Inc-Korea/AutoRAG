/**
 * Ecosia serves a server-rendered Vue/Nuxt results page (no `__NUXT_DATA__`
 * JSON island — probed 2026-07), so both load paths parse the same markup:
 * `<article data-test-id="organic-result">` blocks whose title anchor carries
 * the final target URL directly (no redirect wrapper). The site fronts search
 * with Cloudflare. Requests start with a browser-profiled fetch and escalate
 * to the shared stealth browser only when the response is blocked or fails.
 *
 * Recency is ignored: Ecosia's web results expose no date filter in the UI
 * and the legacy Bing-era `freshness` param is a server-side no-op (verified
 * live), so per the {@link SearchParams.recency} contract the field must not
 * be approximated.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/ecosia.ts`.
 */

import type { LoadedHtmlPage } from "../browser-page.ts";
import { browserFetch } from "../browser-page.ts";
import { formatScraperQuery } from "../query.ts";
import type { SearchProviderId, SearchResponse, SearchSource } from "../types.ts";
import { SearchProviderError } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, normalizeSearchText, withHardTimeout } from "./utils.ts";

const ECOSIA_HOME_URL = "https://www.ecosia.org/";
const ECOSIA_SEARCH_URL = "https://www.ecosia.org/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;
const RESULT_RENDER_TIMEOUT_MS = 10_000;

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

function resolveResultUrl(href: string): string | undefined {
	let url: URL;
	try {
		url = new URL(href, ECOSIA_HOME_URL);
	} catch {
		return undefined;
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
	if (url.hostname === "ecosia.org" || url.hostname === "www.ecosia.org") return undefined;
	return url.href;
}

function parseHtmlResults(html: string): ParsedResult[] {
	const results: ParsedResult[] = [];
	const articleRe = /<article\b[^>]*\bdata-test-id="organic-result"[^>]*>([\s\S]*?)<\/article>/gi;
	for (const articleMatch of html.matchAll(articleRe)) {
		const block = articleMatch[1];
		const headingMatch = /<h2\b[^>]*\bdata-test-id="result-title"[^>]*>([\s\S]*?)<\/h2>/i.exec(block);
		if (!headingMatch) continue;
		// Find the enclosing <a> for the heading
		const beforeHeading = block.slice(0, headingMatch.index);
		const anchorMatch = /<a\b[^>]*\bhref="([^"]+)"[^>]*>\s*$/i.exec(
			beforeHeading.slice(beforeHeading.lastIndexOf("<a ")),
		);
		if (!anchorMatch) continue;
		const url = resolveResultUrl(anchorMatch[1]);
		if (!url) continue;
		const title = normalizeSearchText(decodeHtmlText(headingMatch[1])) ?? "";
		if (!title) continue;
		const descMatch =
			/<p\b[^>]*\bdata-test-id="web-result-description"[^>]*>([\s\S]*?)<\/p>/i.exec(block) ??
			/<[a-z]+\b[^>]*\bdata-test-id="result-description"[^>]*>([\s\S]*?)<\/[a-z]+>/i.exec(block);
		const snippet = descMatch ? normalizeSearchText(decodeHtmlText(descMatch[1])) : undefined;
		results.push({ title, url, snippet });
	}
	return results;
}

function isBlockedPage(page: LoadedHtmlPage): boolean {
	return (
		page.status === 403 ||
		page.status === 429 ||
		page.html.includes("Ecosia Firewall") ||
		page.html.includes("_cf_chl_opt") ||
		page.html.includes("/cdn-cgi/challenge-platform/") ||
		/confirm you.{0,3}re not a robot/i.test(page.html)
	);
}

async function callEcosiaHtml(params: SearchParams): Promise<string> {
	const signal = withHardTimeout(params.signal, params.timeoutMs);
	const url = new URL(ECOSIA_SEARCH_URL);
	url.searchParams.set("q", formatScraperQuery(params.query, params.parsedQuery));

	let page: LoadedHtmlPage;
	try {
		page = await browserFetch(url.href, {
			fetch: params.fetch,
			signal,
			timeoutMs: params.timeoutMs,
			referer: ECOSIA_HOME_URL,
			browser: {
				homeUrl: ECOSIA_HOME_URL,
				ready: { selector: 'article[data-test-id="organic-result"]', timeoutMs: RESULT_RENDER_TIMEOUT_MS },
				shouldFallback: isBlockedPage,
			},
		});
	} catch (error) {
		if (error instanceof SearchProviderError || params.signal?.aborted) throw error;
		if (signal.aborted) {
			throw new SearchProviderError("ecosia", "Ecosia search timed out.", 504);
		}
		const message = error instanceof Error ? error.message : String(error);
		throw new SearchProviderError("ecosia", `Ecosia search failed: ${message}`, 503);
	}

	if (isBlockedPage(page)) {
		throw new SearchProviderError(
			"ecosia",
			"Ecosia blocked the request with a Cloudflare bot challenge. Ecosia's firewall throttles automated searches from datacenter/shared-egress IPs; try another web search provider such as DuckDuckGo, Brave, or Tavily.",
			429,
		);
	}
	if (page.status < 200 || page.status >= 300) {
		const classified = classifyProviderHttpError("ecosia", page.status, page.html);
		if (classified) throw classified;
		throw new SearchProviderError("ecosia", `Ecosia HTML error (${page.status})`, page.status);
	}
	return page.html;
}

/** Execute an Ecosia web search and parse the server-rendered result page. */
export async function searchEcosia(params: SearchParams): Promise<SearchResponse> {
	const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const html = await callEcosiaHtml(params);
	const parsed = parseHtmlResults(html);

	const sources: SearchSource[] = [];
	const seen = new Set<string>();
	for (const result of parsed) {
		if (seen.has(result.url)) continue;
		seen.add(result.url);
		sources.push({ title: result.title, url: result.url, snippet: result.snippet });
		if (sources.length >= numResults) break;
	}

	return { provider: "ecosia", sources };
}

/** Search provider for Ecosia (no API key required). */
export class EcosiaProvider extends SearchProvider {
	readonly id = "ecosia" as SearchProviderId;
	readonly label = "Ecosia";

	isAvailable(): boolean {
		return true;
	}

	override isExplicitlyAvailable(): boolean {
		return true;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchEcosia(params);
	}
}
