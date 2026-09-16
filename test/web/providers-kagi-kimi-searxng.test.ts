import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FetchImpl } from "../../src/web/search/providers/base.ts";
import { KagiProvider } from "../../src/web/search/providers/kagi.ts";
import { KimiProvider } from "../../src/web/search/providers/kimi.ts";
import { SearXNGProvider } from "../../src/web/search/providers/searxng.ts";
import { SearchProviderError } from "../../src/web/search/types.ts";

const ENV_NAMES = [
	"KAGI_API_KEY",
	"KIMI_SEARCH_API_KEY",
	"MOONSHOT_SEARCH_API_KEY",
	"KIMI_SEARCH_BASE_URL",
	"MOONSHOT_SEARCH_BASE_URL",
	"SEARXNG_ENDPOINT",
	"SEARXNG_TOKEN",
	"SEARXNG_BASIC_USERNAME",
	"SEARXNG_BASIC_PASSWORD",
] as const;
const originalEnv = Object.fromEntries(ENV_NAMES.map((name) => [name, process.env[name]]));

beforeEach(() => {
	for (const name of ENV_NAMES) delete process.env[name];
});

afterEach(() => {
	vi.useRealTimers();
	for (const name of ENV_NAMES) {
		const value = originalEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
});

describe("Kagi provider", () => {
	it("gates availability on KAGI_API_KEY", () => {
		const provider = new KagiProvider();
		expect(provider.isAvailable()).toBe(false);
		process.env.KAGI_API_KEY = "kagi-key";
		expect(provider.isAvailable()).toBe(true);
	});

	it("posts a bearer-authenticated V1 request and maps categorized results", async () => {
		process.env.KAGI_API_KEY = "kagi-key";
		let body: Record<string, unknown> = {};
		let auth: string | null = null;
		const fetch: FetchImpl = (_input, init) => {
			body = JSON.parse(String(init?.body)) as Record<string, unknown>;
			auth = new Headers(init?.headers).get("Authorization");
			return Promise.resolve(
				new Response(
					JSON.stringify({
						meta: { trace: "kagi-request" },
						data: {
							search: [{ url: "https://example.com/search", title: "Search", snippet: "Search snippet" }],
							video: [{ url: "https://example.com/video", title: "Video" }],
							related_search: [{ title: "Related", props: { question: "related query" } }],
							direct_answer: [{ snippet: "Direct answer" }],
						},
					}),
					{ status: 200 },
				),
			);
		};

		const response = await new KagiProvider().search({
			query: "runtime domain:example.com until:2025",
			numSearchResults: 5,
			fetch,
		});

		expect(auth).toBe("Bearer kagi-key");
		expect(body).toMatchObject({
			query: "runtime site:example.com before:2025-01-01",
			workflow: "search",
			limit: 5,
		});
		expect(response).toMatchObject({
			provider: "kagi",
			requestId: "kagi-request",
			answer: "Direct answer",
			relatedQuestions: ["related query"],
			sources: [
				{ title: "Search", url: "https://example.com/search", snippet: "Search snippet" },
				{ title: "[Video] Video", url: "https://example.com/video" },
			],
		});
	});

	it.each([401, 402, 403])("classifies HTTP %s", async (status) => {
		process.env.KAGI_API_KEY = "kagi-key";
		await expect(
			new KagiProvider().search({
				query: "denied",
				fetch: () => Promise.resolve(new Response("denied", { status })),
			}),
		).rejects.toSatisfy((error: unknown) => error instanceof SearchProviderError && error.status === status);
	});
});

describe("Kimi provider", () => {
	it("accepts either dedicated search-key environment variable", () => {
		const provider = new KimiProvider();
		expect(provider.isAvailable()).toBe(false);
		process.env.MOONSHOT_SEARCH_API_KEY = "moonshot-search-key";
		expect(provider.isAvailable()).toBe(true);
		delete process.env.MOONSHOT_SEARCH_API_KEY;
		process.env.KIMI_SEARCH_API_KEY = "kimi-search-key";
		expect(provider.isAvailable()).toBe(true);
	});

	it("uses the REST path, prefers KIMI_SEARCH_API_KEY, and maps results", async () => {
		process.env.MOONSHOT_SEARCH_API_KEY = "moonshot-search-key";
		process.env.KIMI_SEARCH_API_KEY = "kimi-search-key";
		let body: Record<string, unknown> = {};
		let auth: string | null = null;
		const fetch: FetchImpl = (_input, init) => {
			body = JSON.parse(String(init?.body)) as Record<string, unknown>;
			auth = new Headers(init?.headers).get("Authorization");
			return Promise.resolve(
				new Response(
					JSON.stringify({
						search_results: [
							{
								site_name: "Example",
								title: "Kimi result",
								url: "https://example.com/kimi",
								snippet: "Kimi snippet",
								date: "2026-04-01",
							},
						],
					}),
					{ status: 200, headers: { "x-request-id": "kimi-request" } },
				),
			);
		};

		const response = await new KimiProvider().search({
			query: 'runtime site:example.com after:2025-01-01 "exact phrase"',
			numSearchResults: 3,
			fetch,
		});

		expect(auth).toBe("Bearer kimi-search-key");
		expect(body).toMatchObject({
			text_query: 'runtime "exact phrase" site:example.com',
			limit: 3,
			enable_page_crawling: false,
			timeout_seconds: 30,
		});
		expect(response).toMatchObject({
			provider: "kimi",
			requestId: "kimi-request",
			sources: [
				{
					title: "Kimi result",
					url: "https://example.com/kimi",
					snippet: "Kimi snippet",
					author: "Example",
				},
			],
		});
	});
});

describe("SearXNG provider", () => {
	it("gates availability on SEARXNG_ENDPOINT", () => {
		const provider = new SearXNGProvider();
		expect(provider.isAvailable()).toBe(false);
		process.env.SEARXNG_ENDPOINT = "https://searx.example.org";
		expect(provider.isAvailable()).toBe(true);
	});

	it("shapes JSON search requests, authenticates, and maps results", async () => {
		process.env.SEARXNG_ENDPOINT = "https://searx.example.org/";
		process.env.SEARXNG_TOKEN = "searx-token";
		let url: URL | undefined;
		let auth: string | null = null;
		const fetch: FetchImpl = (input, init) => {
			url = new URL(input.toString());
			auth = new Headers(init?.headers).get("Authorization");
			return Promise.resolve(
				new Response(
					JSON.stringify({
						answers: ["  Forty-two  "],
						suggestions: ["related query"],
						results: [
							{
								title: "SearXNG result",
								url: "https://example.com/searxng",
								content: "SearXNG snippet",
								publishedDate: "2026-05-01",
							},
						],
					}),
					{ status: 200 },
				),
			);
		};

		const response = await new SearXNGProvider().search({
			query: "!!g runtime lang:de site:example.com",
			recency: "week",
			fetch,
		});

		expect(url?.origin).toBe("https://searx.example.org");
		expect(url?.pathname).toBe("/search");
		expect(url?.searchParams.get("q")).toBe("runtime site:example.com");
		expect(url?.searchParams.get("format")).toBe("json");
		expect(url?.searchParams.get("time_range")).toBe("month");
		expect(url?.searchParams.get("language")).toBe("de");
		expect(auth).toBe("Bearer searx-token");
		expect(response).toMatchObject({
			provider: "searxng",
			answer: "Forty-two",
			relatedQuestions: ["related query"],
			sources: [{ title: "SearXNG result", url: "https://example.com/searxng", snippet: "SearXNG snippet" }],
		});
	});

	it("supports RFC 7617 basic auth with precedence over bearer auth", async () => {
		process.env.SEARXNG_ENDPOINT = "https://searx.example.org";
		process.env.SEARXNG_TOKEN = "ignored";
		process.env.SEARXNG_BASIC_USERNAME = "alice";
		process.env.SEARXNG_BASIC_PASSWORD = "s3cret";
		let auth: string | null = null;
		await new SearXNGProvider().search({
			query: "basic auth",
			fetch: (_input, init) => {
				auth = new Headers(init?.headers).get("Authorization");
				return Promise.resolve(new Response(JSON.stringify({ results: [] }), { status: 200 }));
			},
		});
		expect(auth).toBe(`Basic ${Buffer.from("alice:s3cret", "utf8").toString("base64")}`);
	});

	it("turns empty results with failed upstream engines into a provider error", async () => {
		process.env.SEARXNG_ENDPOINT = "https://searx.example.org";
		await expect(
			new SearXNGProvider().search({
				query: "blocked",
				fetch: () =>
					Promise.resolve(
						new Response(JSON.stringify({ results: [], unresponsive_engines: [["brave", "rate limited"]] }), {
							status: 200,
						}),
					),
			}),
		).rejects.toMatchObject({ provider: "searxng", status: 503 });
	});
});
