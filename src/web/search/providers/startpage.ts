/**
 * Startpage proxies Google's index behind a privacy frontend and serves fully
 * server-rendered result pages — no JS challenge on the happy path. Its bot
 * defense keys on requests that skip the homepage handshake: the search form
 * carries a session token (`sc`) plus sibling hidden inputs, and posting the
 * form with a stale/absent token 302s to the `/en/errors/` CAPTCHA shell.
 * The robust flow is therefore the same dance a real browser performs: GET
 * the homepage, lift the form's hidden inputs, POST them back with the query.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `packages/coding-agent/src/web/search/providers/startpage.ts`.
 */

import type { LoadedHtmlPage } from "../browser-page.ts";
import { browserFetch } from "../browser-page.ts";
import { formatScraperQuery } from "../query.ts";
import type { SearchProviderId, SearchResponse, SearchSource } from "../types.ts";
import { SearchProviderError } from "../types.ts";
import { clampNumResults } from "../utils.ts";
import type { FetchImpl, SearchParams } from "./base.ts";
import { SearchProvider } from "./base.ts";
import { classifyProviderHttpError, normalizeSearchText, withHardTimeout } from "./utils.ts";

const STARTPAGE_HOME_URL = "https://www.startpage.com/";
const STARTPAGE_SEARCH_URL = "https://www.startpage.com/sp/search";
const DEFAULT_NUM_RESULTS = 10;
const MAX_NUM_RESULTS = 20;

const RECENCY_TO_STARTPAGE_WITH_DATE: Record<NonNullable<SearchParams["recency"]>, string> = {
	day: "d",
	week: "w",
	month: "m",
	year: "y",
};

interface ParsedResult {
	title: string;
	url: string;
	snippet?: string;
}

function normalizeText(value: string | null | undefined): string {
	return normalizeSearchText(value) ?? "";
}

function isChallengeResponse(page: LoadedHtmlPage): boolean {
	if (/\/(?:errors|captcha)\//.test(page.url) || page.url.includes("/sp/captcha")) return true;
	return page.html.includes("component---src-pages-captcha") || page.html.includes("/sp/captcha");
}

/**
 * Lift the hidden inputs from the homepage's `/sp/search` form. Returns
 * `undefined` when the form or its `sc` anti-bot token cannot be found so the
 * caller can degrade to a tokenless GET instead of posting a doomed form.
 */
function parseSearchFormInputs(html: string): Record<string, string> | undefined {
	const formMatch = /<form\b[^>]*\baction="\/sp\/search"[^>]*>([\s\S]*?)<\/form>/i.exec(html);
	if (!formMatch) return undefined;
	const inputs: Record<string, string> = {};
	for (const inputMatch of formMatch[1].matchAll(/<input\b[^>]*>/gi)) {
		const input = inputMatch[0];
		const type = /\btype\s*=\s*(["'])(.*?)\1/i.exec(input)?.[2];
		if (type && type.toLowerCase() !== "hidden") continue;
		const name = /\bname\s*=\s*(["'])(.*?)\1/i.exec(input)?.[2];
		const value = /\bvalue\s*=\s*(["'])(.*?)\1/i.exec(input)?.[2];
		if (name) inputs[name] = value ?? "";
	}
	return inputs.sc ? inputs : undefined;
}

/** Accept only http(s) result targets that point away from Startpage itself. */
function sanitizeResultUrl(href: string | null | undefined): string | undefined {
	if (!href) return undefined;
	let url: URL;
	try {
		url = new URL(href, STARTPAGE_HOME_URL);
	} catch {
		return undefined;
	}
	if (url.protocol !== "http:" && url.protocol !== "https:") return undefined;
	if (url.hostname === "startpage.com" || url.hostname.endsWith(".startpage.com")) return undefined;
	return url.href;
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

/**
 * Walk the server-rendered results page in document order.
 *
 * Each organic hit lives in a `div.result` container holding the title
 * anchor `a.result-link` (with an `h2.wgl-title` heading) and an optional
 * `p.description` snippet. Hrefs are direct target URLs — Startpage does not
 * wrap outbound clicks.
 */
function parseHtmlResults(html: string): ParsedResult[] {
	const results: ParsedResult[] = [];
	const blockRe =
		/<div\b[^>]*\bclass="[^"]*\bresult\b[^"]*"[^>]*>([\s\S]*?)(?=<div\b[^>]*\bclass="[^"]*\bresult\b|<div\b[^>]*\bclass="[^"]*\ba-bg-result\b|$)/gi;
	for (const blockMatch of html.matchAll(blockRe)) {
		const block = blockMatch[1];
		const anchorMatch =
			/<a\b[^>]*\bclass="[^"]*\bresult-link\b[^"]*"[^>]*\bhref="([^"]+)"[^>]*>([\s\S]*?)<\/a>/i.exec(block);
		if (!anchorMatch) continue;
		const url = sanitizeResultUrl(anchorMatch[1]);
		if (!url) continue;
		const headingMatch = /<h[23]\b[^>]*>([\s\S]*?)<\/h[23]>/i.exec(anchorMatch[2]);
		const titleText = normalizeText(headingMatch ? decodeHtmlText(headingMatch[1]) : decodeHtmlText(anchorMatch[2]));
		if (!titleText) continue;
		const snippetMatch = /<p\b[^>]*\bclass="[^"]*\bdescription\b[^"]*"[^>]*>([\s\S]*?)<\/p>/i.exec(block);
		const snippetText = snippetMatch ? normalizeText(decodeHtmlText(snippetMatch[1])) : undefined;
		results.push({ title: titleText, url, snippet: snippetText || undefined });
	}
	return results;
}

async function fetchFormInputs(
	fetchImpl: FetchImpl,
	signal: AbortSignal,
	timeoutMs?: number,
): Promise<Record<string, string> | undefined> {
	let page: LoadedHtmlPage;
	try {
		page = await browserFetch(STARTPAGE_HOME_URL, { fetch: fetchImpl, signal, timeoutMs });
	} catch (error) {
		if (signal.aborted) throw error;
		return undefined;
	}
	if (page.status < 200 || page.status >= 300 || isChallengeResponse(page)) return undefined;
	return parseSearchFormInputs(page.html);
}

async function callStartpageHtml(params: SearchParams): Promise<string> {
	const fetchImpl = params.fetch ?? fetch;
	const signal = withHardTimeout(params.signal, params.timeoutMs);
	const withDate = params.recency ? RECENCY_TO_STARTPAGE_WITH_DATE[params.recency] : undefined;
	const query = formatScraperQuery(params.query, params.parsedQuery);

	const formInputs = await fetchFormInputs(fetchImpl, signal, params.timeoutMs);
	let page: LoadedHtmlPage;
	if (formInputs) {
		const form = new URLSearchParams(formInputs);
		form.set("query", query);
		if (withDate) form.set("with_date", withDate);
		page = await browserFetch(STARTPAGE_SEARCH_URL, {
			fetch: fetchImpl,
			signal,
			timeoutMs: params.timeoutMs,
			referer: STARTPAGE_HOME_URL,
			init: { method: "POST", body: form.toString() },
			headers: { "Content-Type": "application/x-www-form-urlencoded" },
		});
	} else {
		const url = new URL(STARTPAGE_SEARCH_URL);
		url.searchParams.set("query", query);
		if (withDate) url.searchParams.set("with_date", withDate);
		page = await browserFetch(url.href, {
			fetch: fetchImpl,
			signal,
			timeoutMs: params.timeoutMs,
			referer: STARTPAGE_HOME_URL,
		});
	}

	if (isChallengeResponse(page)) {
		throw new SearchProviderError(
			"startpage",
			"Startpage blocked the request with a CAPTCHA challenge. Startpage rate-limits automated searches from datacenter/shared-egress IPs; try another provider such as DuckDuckGo or Mojeek, or retry later.",
			429,
		);
	}
	if (page.status < 200 || page.status >= 300) {
		const classified = classifyProviderHttpError("startpage", page.status, page.html);
		if (classified) throw classified;
		throw new SearchProviderError("startpage", `Startpage HTML error (${page.status})`, page.status);
	}
	return page.html;
}

/** Execute a Startpage web search via the homepage-token form flow. */
export async function searchStartpage(params: SearchParams): Promise<SearchResponse> {
	const numResults = clampNumResults(params.numSearchResults ?? params.limit, DEFAULT_NUM_RESULTS, MAX_NUM_RESULTS);
	const html = await callStartpageHtml(params);
	const parsed = parseHtmlResults(html);

	const sources: SearchSource[] = [];
	const seen = new Set<string>();
	for (const result of parsed) {
		if (seen.has(result.url)) continue;
		seen.add(result.url);
		sources.push({ title: result.title, url: result.url, snippet: result.snippet });
		if (sources.length >= numResults) break;
	}

	return { provider: "startpage", sources };
}

/** Search provider for Startpage (no API key required). */
export class StartpageProvider extends SearchProvider {
	readonly id = "startpage" as SearchProviderId;
	readonly label = "Startpage";

	isAvailable(): boolean {
		return true;
	}

	override isExplicitlyAvailable(): boolean {
		return true;
	}

	search(params: SearchParams): Promise<SearchResponse> {
		return searchStartpage(params);
	}
}
