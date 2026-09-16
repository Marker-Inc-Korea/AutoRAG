/**
 * Browser-profiled fetch for credential-free search engines.
 *
 * Ported from oh-my-pi (can1357/oh-my-pi, MIT)
 * `web/search/providers/browser-page.ts`. Escalation uses the loader from
 * `browser-loader.ts`: an explicitly registered loader when present,
 * otherwise the default puppeteer-core loader probed from the local
 * Chrome/Chromium install. Without a loader the plain fetch result is
 * returned as-is and the provider's own challenge detection (anomaly pages,
 * status codes) routes the chain to the next provider.
 */
import { buildBrowserNavigationHeaders } from "./browser-headers.ts";
import { resolveWebSearchBrowserLoader, setWebSearchBrowserLoader } from "./browser-loader.ts";
import type { FetchImpl } from "./providers/base.ts";
import { SEARCH_HARD_TIMEOUT_MS } from "./providers/utils.ts";

export { setWebSearchBrowserLoader };

/** HTML plus the response status and final URL after redirects or browser navigation. */
export interface LoadedHtmlPage {
	html: string;
	status: number;
	url: string;
}

export interface BrowserFallbackOptions {
	homeUrl?: string;
	ready?: { selector: string; timeoutMs: number };
	afterNavigation?: (page: unknown, signal: AbortSignal) => Promise<void>;
	shouldFallback: (page: LoadedHtmlPage) => boolean;
	attempts?: number;
	retryDelayMs?: number;
}

/** Controls a browser-profiled fetch and its optional headless-browser fallback. */
export interface BrowserFetchOptions {
	fetch?: FetchImpl;
	signal: AbortSignal;
	timeoutMs?: number;
	randomizeHeaders?: boolean;
	referer?: string;
	init?: Omit<RequestInit, "headers" | "signal">;
	headers?: Readonly<Record<string, string>>;
	browser?: BrowserFallbackOptions;
}

/**
 * Headless-browser loader. Receives the target URL and the provider's
 * fallback options and must return the loaded page. The registry lives in
 * `browser-loader.ts`, which ships a default puppeteer-core loader probed
 * from the local Chrome/Chromium install; an explicit registration always
 * takes precedence over the default.
 */
export type WebSearchBrowserLoader = (
	url: string,
	options: BrowserFallbackOptions,
	signal: AbortSignal,
	timeoutMs: number,
) => Promise<LoadedHtmlPage>;

async function fetchHtmlPage(url: string, options: BrowserFetchOptions, fetchImpl: FetchImpl): Promise<LoadedHtmlPage> {
	const response = await fetchImpl(url, {
		...options.init,
		headers: {
			...buildBrowserNavigationHeaders({ randomized: options.randomizeHeaders }),
			...(options.referer ? { Referer: options.referer, "Sec-Fetch-Site": "same-origin" } : {}),
			...options.headers,
		},
		signal: options.signal,
	});
	return { html: await response.text(), status: response.status, url: response.url || url };
}

/** Fetch with a fresh browser profile, escalating rejected production responses to the installed headless browser. */
export async function browserFetch(url: string, options: BrowserFetchOptions): Promise<LoadedHtmlPage> {
	const fetchImpl = options.fetch ?? fetch;
	// An injected `fetch` transport marks a test/proxy context and never
	// escalates into a real browser; otherwise the explicit-or-default loader
	// handles bot-challenged responses.
	const loader = options.browser && !options.fetch ? await resolveWebSearchBrowserLoader() : undefined;
	const escalate = options.browser && loader && !options.fetch;
	let page: LoadedHtmlPage;
	try {
		page = await fetchHtmlPage(url, options, fetchImpl);
	} catch (error) {
		if (!escalate) throw error;
		return loader!(url, options.browser!, options.signal, options.timeoutMs ?? SEARCH_HARD_TIMEOUT_MS);
	}

	if (!escalate) return page;
	const isSuccessful = page.status >= 200 && page.status < 300;
	if (isSuccessful && !options.browser!.shouldFallback(page)) return page;
	return loader!(url, options.browser!, options.signal, options.timeoutMs ?? SEARCH_HARD_TIMEOUT_MS);
}
