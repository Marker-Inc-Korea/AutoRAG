// Ported from oh-my-pi (can1357/oh-my-pi, MIT) packages/coding-agent/test/tools/web-search-duckduckgo.test.ts and packages/coding-agent/test/web/search/duckduckgo.test.ts.
import { describe, expect, it } from "vitest";
import type { FetchImpl, SearchParams } from "../../src/web/search/providers/base.ts";
import { localeToKl, searchDuckDuckGo } from "../../src/web/search/providers/duckduckgo.ts";
import { applyQueryConstraints, parseSearchQuery } from "../../src/web/search/query.ts";

function makeParams(query: string, fetch: FetchImpl): SearchParams {
	return { query, systemPrompt: "DuckDuckGo search test prompt", fetch };
}

function resultBlock(
	url: string,
	title: string,
	snippet: string,
	timestamp?: string,
	snippetTag: "a" | "span" = "a",
): string {
	return `
		<div class="result results_links results_links_deep web-result ">
			<div class="links_main links_deep result__body">
				<h2 class="result__title">
					<a rel="nofollow" class="result__a" href="${url}">${title}</a>
				</h2>
				<div class="result__extras">
					<div class="result__extras__url">
						<a class="result__url" href="${url}">${url}</a>
						${timestamp ? `<span>&nbsp; &nbsp; ${timestamp}</span>` : ""}
					</div>
				</div>
				<${snippetTag} class="result__snippet" href="${url}">${snippet}</${snippetTag}>
			</div>
		</div>`;
}

function resultsPage(blocks: string, continuation = false): string {
	return `<!DOCTYPE html><html><body><div id="links" class="results">${blocks}${
		continuation
			? `<div class="nav-link"><form action="/html/" method="post">
				<input type="hidden" name="q" value="open source software" />
				<input value="10" type="hidden" name="s" />
				<input type="hidden" name="nextParams" value="" />
				<input type="hidden" name="v" value="l" />
				<input type="hidden" name="o" value="json" />
				<input type="hidden" name="dc" value="11" />
				<input type="hidden" name="api" value="d.js" />
				<input value="test-vqd" name="vqd" type="hidden" />
				<input name="kl" value="us-en" type="hidden" />
			</form></div>`
			: '<div class="nav-link"></div>'
	}</div></body></html>`;
}

function numberedResult(index: number): string {
	return resultBlock(
		`//duckduckgo.com/l/?uddg=${encodeURIComponent(`https://example.com/${index}`)}&rut=tracking`,
		`Result <b>${index}</b>`,
		`Snippet &amp; ${index}`,
	);
}

describe("localeToKl", () => {
	it("maps standard and provider-specific locales", () => {
		expect(localeToKl("de-de")).toBe("de-de");
		expect(localeToKl("en-us")).toBe("us-en");
		expect(localeToKl("pt-br")).toBe("br-pt");
		expect(localeToKl("en-gb")).toBe("uk-en");
		expect(localeToKl("ja-jp")).toBe("jp-jp");
		expect(localeToKl("zh-tw")).toBe("tw-tzh");
		expect(localeToKl("EN_US")).toBe("us-en");
	});

	it("rejects unsupported or malformed locales", () => {
		expect(localeToKl(undefined)).toBeUndefined();
		expect(localeToKl("de")).toBeUndefined();
		expect(localeToKl("en-jp")).toBeUndefined();
		expect(localeToKl("zz-zz")).toBeUndefined();
	});
});

describe("DuckDuckGo web search provider", () => {
	it("submits continuation forms, unwraps URLs, deduplicates, and clamps the limit to 20", async () => {
		const requests: URLSearchParams[] = [];
		const fetchMock: FetchImpl = async (_input, init) => {
			expect(init?.method).toBe("POST");
			const body = new URLSearchParams(String(init?.body));
			requests.push(body);
			const indices = requests.length === 1 ? [...Array(10).keys()] : [...Array(11).keys()].map((i) => i + 9);
			return new Response(resultsPage(indices.map(numberedResult).join(""), requests.length === 1), { status: 200 });
		};

		const response = await searchDuckDuckGo({
			...makeParams("open source software", fetchMock),
			numSearchResults: 99,
		});

		expect(requests).toHaveLength(2);
		expect(Object.fromEntries(requests[0])).toEqual({ q: "open source software", kl: "us-en", b: "" });
		expect(Object.fromEntries(requests[1])).toEqual({
			q: "open source software",
			s: "10",
			nextParams: "",
			v: "l",
			o: "json",
			dc: "11",
			api: "d.js",
			vqd: "test-vqd",
			kl: "us-en",
		});
		expect(response.sources).toHaveLength(20);
		expect(response.sources[0]).toMatchObject({
			title: "Result 0",
			url: "https://example.com/0",
			snippet: "Snippet & 0",
		});
		expect(response.sources.at(-1)?.url).toBe("https://example.com/19");
	});

	it("maps recency and lang directives into DDG form fields", async () => {
		let form = new URLSearchParams();
		const fetchMock: FetchImpl = async (_input, init) => {
			form = new URLSearchParams(String(init?.body));
			return new Response(resultsPage(""), { status: 200 });
		};
		await searchDuckDuckGo({ ...makeParams("weather lang:de-de", fetchMock), recency: "week" });
		expect(Object.fromEntries(form)).toEqual({ q: "weather", kl: "de-de", df: "w", b: "" });
	});

	it("extracts timestamps without mistaking a date-leading snippet for publication metadata", async () => {
		const html = resultsPage(
			[
				resultBlock("https://example.com/fresh", "Fresh page", "A recent article.", "2026-07-30T20:19:00.0000000"),
				resultBlock("https://example.com/undated", "Undated page", "2020-01-02", undefined, "span"),
			].join(""),
		);
		const response = await searchDuckDuckGo(makeParams("weather", () => Promise.resolve(new Response(html))));
		expect(response.sources[0].publishedDate).toBe("2026-07-30T20:19:00.0000000");
		expect(response.sources[0].ageSeconds).toBeGreaterThan(0);
		expect(response.sources[1].publishedDate).toBeUndefined();
		expect(response.sources[1].ageSeconds).toBeUndefined();
	});

	it("supports date-bound post-filtering with extracted timestamps", async () => {
		const html = resultsPage(
			[
				resultBlock("https://example.com/in-range", "In range", "Within.", "2026-07-10T09:00:00.0000000"),
				resultBlock("https://example.com/too-new", "Too new", "After.", "2026-07-30T09:00:00.0000000"),
				resultBlock("https://example.com/undated", "Undated", "Unknown."),
			].join(""),
		);
		const query = "weather after:2026-07-01 before:2026-07-15";
		const response = await searchDuckDuckGo(makeParams(query, () => Promise.resolve(new Response(html))));
		const urls = applyQueryConstraints(response.sources, parseSearchQuery(query)).sources.map((source) => source.url);
		expect(urls).toContain("https://example.com/in-range");
		expect(urls).toContain("https://example.com/undated");
		expect(urls).not.toContain("https://example.com/too-new");
	});

	it("surfaces anomaly pages as a provider-tagged 429", async () => {
		const promise = searchDuckDuckGo(
			makeParams("blocked", () => Promise.resolve(new Response('<div id="anomaly-modal"></div>', { status: 202 }))),
		);
		await expect(promise).rejects.toMatchObject({ provider: "duckduckgo", status: 429 });
	});
});
