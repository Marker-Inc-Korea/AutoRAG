/**
 * Default headless-browser escalation for bot-challenged engines (PR #1592
 * review comment 5699346366, item 2): a default loader is registered from
 * the locally installed Chrome/Chromium; when no browser exists the
 * degradation is an explicit diagnostic, never a crash, and the provider
 * chain simply continues without escalation.
 *
 * All browser interaction is stubbed through the injected `launch` seam;
 * no real browser starts in unit tests.
 */
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	ensureWebSearchBrowserLoader,
	getWebSearchBrowserDiagnostic,
	resetWebSearchBrowserLoaderForTests,
	type WebSearchBrowser,
	type WebSearchBrowserPage,
} from "../../src/web/search/browser-loader.ts";
import { browserFetch, setWebSearchBrowserLoader } from "../../src/web/search/browser-page.ts";

const CHALLENGED_PAGE = { html: "<html>challenge</html>", status: 403, url: "https://engine.example/search" };
const REAL_PAGE = { html: "<html><body>results</body></html>", status: 200, url: "https://engine.example/search" };

afterEach(() => {
	vi.unstubAllGlobals();
	resetWebSearchBrowserLoaderForTests();
});

describe("ensureWebSearchBrowserLoader", () => {
	it("records a diagnostic and installs nothing when no Chrome/Chromium exists", async () => {
		const installed = await ensureWebSearchBrowserLoader({ exists: () => false });
		expect(installed).toBe(false);
		expect(getWebSearchBrowserDiagnostic()).toMatch(/Chrome|Chromium|browser/i);
	});

	it("does not throw when the browser probe itself fails", async () => {
		const installed = await ensureWebSearchBrowserLoader({
			exists: () => {
				throw new Error("filesystem exploded");
			},
		});
		expect(installed).toBe(false);
		expect(getWebSearchBrowserDiagnostic()).toBeDefined();
	});

	it("installs the default loader when Chrome exists, and browserFetch escalates through it", async () => {
		const installed = await ensureWebSearchBrowserLoader({
			exists: (path) => path === "/fake/chrome",
			candidates: ["/fake/chrome"],
			launch: async () => fakeBrowser(REAL_PAGE),
		});
		expect(installed).toBe(true);
		expect(getWebSearchBrowserDiagnostic()).toBeUndefined();

		// The plain fetch gets a challenged page; the browser fallback must run.
		// (browserFetch never escalates when its `fetch` transport is injected,
		// so the plain transport is stubbed globally instead.)
		vi.stubGlobal("fetch", async () => new Response(CHALLENGED_PAGE.html, { status: CHALLENGED_PAGE.status }));
		const page = await browserFetch("https://engine.example/search?q=x", {
			signal: AbortSignal.timeout(5_000),
			browser: { shouldFallback: (candidate) => candidate.status >= 400 },
		});
		expect(page.html).toBe(REAL_PAGE.html);
	});

	it("leaves no teardown timer behind after an escalation", async () => {
		vi.useFakeTimers();
		try {
			await ensureWebSearchBrowserLoader({
				exists: (path) => path === "/fake/chrome",
				candidates: ["/fake/chrome"],
				launch: async () => fakeBrowser(REAL_PAGE),
			});
			vi.stubGlobal("fetch", async () => new Response(CHALLENGED_PAGE.html, { status: CHALLENGED_PAGE.status }));
			await browserFetch("https://engine.example/search?q=x", {
				signal: new AbortController().signal,
				browser: { shouldFallback: (candidate) => candidate.status >= 400 },
			});
			// A pending teardown deadline would keep the Node event loop alive
			// and hang the CLI for the full timeout after a completed search.
			expect(vi.getTimerCount()).toBe(0);
		} finally {
			vi.useRealTimers();
		}
	});

	it("an explicitly registered loader takes precedence over the default", async () => {
		setWebSearchBrowserLoader(async () => ({ html: "explicit", status: 200, url: "https://x" }));
		vi.stubGlobal("fetch", async () => new Response(CHALLENGED_PAGE.html, { status: CHALLENGED_PAGE.status }));
		const page = await browserFetch("https://engine.example/search?q=x", {
			signal: AbortSignal.timeout(5_000),
			browser: { shouldFallback: () => true },
		});
		expect(page.html).toBe("explicit");
	});

	it("without any browser the chain degrades to the plain page, never a crash", async () => {
		await ensureWebSearchBrowserLoader({ exists: () => false });
		vi.stubGlobal("fetch", async () => new Response(CHALLENGED_PAGE.html, { status: CHALLENGED_PAGE.status }));
		const page = await browserFetch("https://engine.example/search?q=x", {
			signal: AbortSignal.timeout(5_000),
			browser: { shouldFallback: () => true },
		});
		// No loader: the challenged plain response passes through so the
		// provider's own challenge detection can advance the chain.
		expect(page.status).toBe(403);
	});
});

function fakeBrowser(page: { html: string; status: number; url: string }): WebSearchBrowser {
	const fakePage: WebSearchBrowserPage = {
		content: async () => page.html,
		url: () => page.url,
		close: async () => {},
		goto: async (_target: string) => ({ status: () => page.status }),
		setViewport: async () => {},
		evaluateOnNewDocument: async () => {},
		waitForSelector: async () => null,
	};
	return { newPage: async () => fakePage, close: async () => {} };
}
