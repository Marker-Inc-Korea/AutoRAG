import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { FetchImpl } from "../../src/web/search/providers/base.ts";
import { SearXNGProvider } from "../../src/web/search/providers/searxng.ts";

const ENV_NAMES = ["SEARXNG_ENDPOINT", "SEARXNG_TOKEN", "SEARXNG_BASIC_USERNAME", "SEARXNG_BASIC_PASSWORD"] as const;
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
