// Ported from oh-my-pi (can1357/oh-my-pi, MIT) packages/coding-agent/test/tools/web-search-mojeek.test.ts and provider parsing behavior.
import { describe, expect, it } from "vitest";
import type { FetchImpl } from "../../src/web/search/providers/base.ts";
import { EcosiaProvider, searchEcosia } from "../../src/web/search/providers/ecosia.ts";
import { GoogleProvider, searchGoogle } from "../../src/web/search/providers/google.ts";
import { MojeekProvider, searchMojeek } from "../../src/web/search/providers/mojeek.ts";
import { StartpageProvider, searchStartpage } from "../../src/web/search/providers/startpage.ts";

function response(html: string, status = 200): Promise<Response> {
	return Promise.resolve(new Response(html, { status, headers: { "Content-Type": "text/html" } }));
}

function startpageResult(href: string, title: string, snippet?: string): string {
	return `<div class="result"><a class="result-link" href="${href}"><h2 class="wgl-title">${title}</h2></a>${snippet ? `<p class="description">${snippet}</p>` : ""}</div>`;
}

function googleResult(href: string, title: string, snippet?: string): string {
	return `<div class="MjjYud"><div class="tF2Cxc"><a href="${href}"><h3>${title}</h3></a>${snippet ? `<div data-sncf="1"><div class="VwiC3b">${snippet}</div></div>` : ""}</div></div>`;
}

function ecosiaResult(href: string, title: string, snippet?: string): string {
	return `<article data-test-id="organic-result"><a href="${href}"><h2 data-test-id="result-title">${title}</h2></a>${snippet ? `<p data-test-id="web-result-description">${snippet}</p>` : ""}</article>`;
}

function mojeekResult(href: string, title: string, snippet?: string, liClass = "r1"): string {
	return `<li class="${liClass}"><h2><a class="title" href="${href}">${title}</a></h2>${snippet ? `<p class="s">${snippet}</p>` : ""}</li>`;
}

function mojeekPage(items: string): string {
	return `<html><body><ul class="results-standard">${items}</ul></body></html>`;
}

describe("credential-free provider availability", () => {
	it("is always available explicitly and automatically", async () => {
		for (const provider of [
			new StartpageProvider(),
			new GoogleProvider(),
			new EcosiaProvider(),
			new MojeekProvider(),
		]) {
			expect(await provider.isAvailable()).toBe(true);
			expect(await provider.isExplicitlyAvailable()).toBe(true);
		}
	});
});

describe("Startpage provider", () => {
	it("uses the homepage token form, maps recency, parses direct URLs, deduplicates, and clamps to 20", async () => {
		const calls: Array<{ url: string; init?: RequestInit }> = [];
		const fetchMock: FetchImpl = (input, init) => {
			const url = input.toString();
			calls.push({ url, init });
			if (calls.length === 1) {
				return response(
					'<form action="/sp/search"><input type="hidden" name="sc" value="token-1"><input type="hidden" name="cat" value="web"></form>',
				);
			}
			const rows = [...Array(22).keys()]
				.map((index) =>
					startpageResult(`https://example.com/${index}`, `Result <b>${index}</b>`, `Snippet &amp; ${index}`),
				)
				.join("");
			return response(
				`${rows}${startpageResult("https://example.com/0", "Duplicate")}${startpageResult("/sp/search", "Internal")}`,
			);
		};

		const result = await searchStartpage({
			query: "privacy domain:example.com",
			recency: "month",
			numSearchResults: 99,
			fetch: fetchMock,
		});
		expect(calls).toHaveLength(2);
		expect(calls[0].url).toBe("https://www.startpage.com/");
		expect(calls[1].url).toBe("https://www.startpage.com/sp/search");
		const form = new URLSearchParams(String(calls[1].init?.body));
		expect(Object.fromEntries(form)).toEqual({
			sc: "token-1",
			cat: "web",
			query: "privacy site:example.com",
			with_date: "m",
		});
		expect(result.sources).toHaveLength(20);
		expect(result.sources[0]).toEqual({
			title: "Result 0",
			url: "https://example.com/0",
			snippet: "Snippet & 0",
		});
	});

	it("falls back to a tokenless GET and reports CAPTCHA shells as 429", async () => {
		const calls: string[] = [];
		const fetchMock: FetchImpl = (input) => {
			calls.push(input.toString());
			return calls.length === 1
				? response("<html>markup drift</html>")
				: response('<script src="component---src-pages-captcha.js"></script>');
		};
		await expect(searchStartpage({ query: "blocked", recency: "day", fetch: fetchMock })).rejects.toMatchObject({
			provider: "startpage",
			status: 429,
		});
		const url = new URL(calls[1]);
		expect(url.searchParams.get("query")).toBe("blocked");
		expect(url.searchParams.get("with_date")).toBe("d");
	});
});

describe("Google provider", () => {
	it("unwraps redirect URLs, maps recency, normalizes snippets, deduplicates, and clamps count", async () => {
		let requested = "";
		const rows = [...Array(22).keys()]
			.map((index) =>
				googleResult(
					`/url?q=${encodeURIComponent(`https://example.com/${index}`)}&sa=U`,
					`Google <em>${index}</em>`,
					`Snippet ${index} Read more`,
				),
			)
			.join("");
		const fetchMock: FetchImpl = (input) => {
			requested = input.toString();
			return response(`${rows}${googleResult("/search?q=internal", "Internal")}`);
		};
		const result = await searchGoogle({
			query: "news since:2024",
			recency: "year",
			numSearchResults: 99,
			fetch: fetchMock,
		});
		const url = new URL(requested);
		expect(url.searchParams.get("q")).toBe("news after:2024-01-01");
		expect(url.searchParams.get("num")).toBe("20");
		expect(url.searchParams.get("tbs")).toBe("qdr:y");
		expect(result.sources).toHaveLength(20);
		expect(result.sources[0]).toEqual({ title: "Google 0", url: "https://example.com/0", snippet: "Snippet 0" });
	});

	it.each([
		["traffic", "<html>detected unusual traffic</html>"],
		["javascript", '<a href="/httpservice/retry/enablejs">enable javascript</a>'],
	])("reports %s challenge pages as 429", async (_kind, html) => {
		await expect(searchGoogle({ query: "blocked", fetch: () => response(html) })).rejects.toMatchObject({
			provider: "google",
			status: 429,
		});
	});
});

describe("Ecosia provider", () => {
	it("parses organic results, skips internal URLs, clamps count, and ignores recency", async () => {
		let requested = "";
		const rows = [...Array(22).keys()]
			.map((index) => ecosiaResult(`https://example.com/${index}`, `Eco <strong>${index}</strong>`, `Tree ${index}`))
			.join("");
		const fetchMock: FetchImpl = (input) => {
			requested = input.toString();
			return response(`${rows}${ecosiaResult("/images", "Internal")}`);
		};
		const result = await searchEcosia({ query: "trees", recency: "week", numSearchResults: 99, fetch: fetchMock });
		const url = new URL(requested);
		expect(url.searchParams.get("q")).toBe("trees");
		expect(url.searchParams.has("freshness")).toBe(false);
		expect(url.searchParams.has("tbs")).toBe(false);
		expect(result.sources).toHaveLength(20);
		expect(result.sources[0]).toEqual({ title: "Eco 0", url: "https://example.com/0", snippet: "Tree 0" });
	});

	it("reports Cloudflare challenge pages as 429", async () => {
		await expect(
			searchEcosia({ query: "blocked", fetch: () => response("<script>window._cf_chl_opt = {}</script>", 403) }),
		).rejects.toMatchObject({ provider: "ecosia", status: 429 });
	});
});

describe("Mojeek provider", () => {
	it("requests locale, clamped count, recency, and supported query operators", async () => {
		let requested = "";
		let capturedInit: RequestInit | undefined;
		const fetchMock: FetchImpl = (input, init) => {
			requested = input.toString();
			capturedInit = init;
			return response(mojeekPage(mojeekResult("https://example.com/result", "Result", "Search snippet")));
		};
		const result = await searchMojeek({
			query: "independent index site:example.com after:2024",
			recency: "week",
			numSearchResults: 99,
			fetch: fetchMock,
		});
		const url = new URL(requested);
		expect(url.origin + url.pathname).toBe("https://www.mojeek.de/search");
		expect(url.searchParams.get("q")).toBe("independent index site:example.com");
		expect(url.searchParams.get("t")).toBe("20");
		expect(url.searchParams.get("since")).toBe("week");
		expect(url.searchParams.get("arc")).toBe("none");
		expect(url.searchParams.get("lang")).toBe("en");
		const headers = new Headers(capturedInit?.headers);
		expect(headers.get("user-agent")).toMatch(/Chrome\/\d+\.0\.0\.0/);
		expect(headers.get("referer")).toBe("https://www.mojeek.de/?arc=none&lang=en&lb=en&theme=dark");
		expect(result.sources).toEqual([
			{ title: "Result", url: "https://example.com/result", snippet: "Search snippet" },
		]);
	});

	it("parses, normalizes, deduplicates, and skips junk and internal rows", async () => {
		const html = mojeekPage(
			[
				mojeekResult(
					"https://bun.sh/",
					"Bun &amp; friends — a <em>fast</em> runtime",
					"<strong>Bun</strong> is a runtime.",
				),
				mojeekResult("https://bun.com/docs", "Bun docs", undefined, "r2 clu-result"),
				mojeekResult("https://bun.sh/", "Duplicate", "duplicate"),
				'<li><p class="s">No title</p></li>',
				mojeekResult("/search?q=bun", "Internal"),
				mojeekResult("https://www.mojeek.com/about/", "About"),
			].join(""),
		);
		const result = await searchMojeek({ query: "bun", fetch: () => response(html) });
		expect(result.sources).toEqual([
			{ title: "Bun & friends — a fast runtime", url: "https://bun.sh/", snippet: "Bun is a runtime." },
			{ title: "Bun docs", url: "https://bun.com/docs", snippet: undefined },
		]);
	});

	it.each([
		[200, '<div class="captcha-wrap"><altcha-widget></altcha-widget></div>'],
		[403, "<h2>Your network appears to be sending automated queries</h2>"],
	])("maps robot wall status %s to a provider-tagged 429", async (status, html) => {
		await expect(searchMojeek({ query: "blocked", fetch: () => response(html, status) })).rejects.toMatchObject({
			provider: "mojeek",
			status: 429,
		});
	});
});
