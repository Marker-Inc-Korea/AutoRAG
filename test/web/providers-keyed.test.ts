import { afterEach, describe, expect, it } from "vitest";
import type { FetchImpl } from "../../src/web/search/providers/base.ts";
import { BraveProvider } from "../../src/web/search/providers/brave.ts";
import { ExaProvider } from "../../src/web/search/providers/exa.ts";
import { JinaProvider } from "../../src/web/search/providers/jina.ts";
import { TavilyProvider } from "../../src/web/search/providers/tavily.ts";
import { SearchProviderError } from "../../src/web/search/types.ts";

const ENV_NAMES = ["BRAVE_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY", "JINA_API_KEY"] as const;
const originalEnv = Object.fromEntries(ENV_NAMES.map((name) => [name, process.env[name]]));

afterEach(() => {
	for (const name of ENV_NAMES) {
		const value = originalEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
});

describe("keyed REST provider availability", () => {
	it.each([
		["BRAVE_API_KEY", new BraveProvider()],
		["TAVILY_API_KEY", new TavilyProvider()],
		["EXA_API_KEY", new ExaProvider()],
		["JINA_API_KEY", new JinaProvider()],
	] as const)("gates %s on a non-empty environment credential", (name, provider) => {
		delete process.env[name];
		expect(provider.isAvailable()).toBe(false);
		process.env[name] = "  ";
		expect(provider.isAvailable()).toBe(false);
		process.env[name] = "test-key";
		expect(provider.isAvailable()).toBe(true);
	});
});

describe("Brave provider", () => {
	it("shapes the request, authenticates, and maps results", async () => {
		process.env.BRAVE_API_KEY = "brave-key";
		let url: URL | undefined;
		let headers: Headers | undefined;
		const fetch: FetchImpl = (input, init) => {
			url = new URL(input.toString());
			headers = new Headers(init?.headers);
			return Promise.resolve(
				new Response(
					JSON.stringify({
						web: {
							results: [
								{
									title: "<b>Brave</b> result",
									url: "https://example.com/brave",
									description: "Primary snippet",
									extra_snippets: ["Extra snippet"],
									age: "2026-01-01",
								},
							],
						},
					}),
					{ status: 200, headers: { "x-request-id": "brave-request" } },
				),
			);
		};

		const response = await new BraveProvider().search({
			query: "runtime site:example.com after:2025-01-01",
			numSearchResults: 3,
			fetch,
		});

		expect(url?.origin).toBe("https://api.search.brave.com");
		expect(url?.pathname).toBe("/res/v1/web/search");
		expect(url?.searchParams.get("q")).toBe("runtime site:example.com");
		expect(url?.searchParams.get("count")).toBe("3");
		expect(url?.searchParams.get("freshness")).toMatch(/^2025-01-01to\d{4}-\d{2}-\d{2}$/);
		expect(headers?.get("X-Subscription-Token")).toBe("brave-key");
		expect(response).toMatchObject({
			provider: "brave",
			requestId: "brave-request",
			authMode: "api_key",
			sources: [
				{
					title: "Brave result",
					url: "https://example.com/brave",
					snippet: "Primary snippet\nExtra snippet",
					publishedDate: "2026-01-01",
				},
			],
		});
	});
});

describe("Tavily provider", () => {
	it("maps native filters, bearer auth, and result fields", async () => {
		process.env.TAVILY_API_KEY = "tavily-key";
		let body: Record<string, unknown> = {};
		let auth: string | null = null;
		const fetch: FetchImpl = (_input, init) => {
			body = JSON.parse(String(init?.body)) as Record<string, unknown>;
			auth = new Headers(init?.headers).get("Authorization");
			return Promise.resolve(
				new Response(
					JSON.stringify({
						answer: "Tavily answer",
						request_id: "tavily-request",
						results: [
							{
								title: "Tavily result",
								url: "https://example.com/tavily",
								content: "Tavily snippet",
								published_date: "2026-02-01",
							},
						],
					}),
					{ status: 200 },
				),
			);
		};

		const response = await new TavilyProvider().search({
			query: "runtime site:example.com -site:blocked.example after:2025-01-01 before:2026-01-01",
			recency: "week",
			fetch,
		});

		expect(auth).toBe("Bearer tavily-key");
		expect(body).toMatchObject({
			query: "runtime",
			include_domains: ["example.com"],
			exclude_domains: ["blocked.example"],
			start_date: "2025-01-01",
			end_date: "2026-01-01",
			include_answer: "advanced",
		});
		expect(body).not.toHaveProperty("time_range");
		expect(response).toMatchObject({
			provider: "tavily",
			answer: "Tavily answer",
			requestId: "tavily-request",
			sources: [{ title: "Tavily result", url: "https://example.com/tavily", snippet: "Tavily snippet" }],
		});
	});
});

describe("Exa provider", () => {
	it("uses API-key mode only, maps native constraints, and synthesizes summaries", async () => {
		process.env.EXA_API_KEY = "exa-key";
		let body: Record<string, unknown> = {};
		let auth: string | null = null;
		const fetch: FetchImpl = (_input, init) => {
			body = JSON.parse(String(init?.body)) as Record<string, unknown>;
			auth = new Headers(init?.headers).get("x-api-key");
			return Promise.resolve(
				new Response(
					JSON.stringify({
						requestId: "exa-request",
						results: [
							{
								title: "Exa result",
								url: "https://example.com/exa",
								summary: "Exa summary",
								publishedDate: "2026-03-01",
								author: "Example author",
							},
						],
					}),
					{ status: 200 },
				),
			);
		};

		const response = await new ExaProvider().search({
			query: '"runtime docs" site:example.com -site:blocked.example after:2025-01-01 before:2026-01-01',
			numSearchResults: 4,
			fetch,
		});

		expect(auth).toBe("exa-key");
		expect(body).toMatchObject({
			query: '"runtime docs"',
			numResults: 4,
			type: "auto",
			includeDomains: ["example.com"],
			excludeDomains: ["blocked.example"],
			startPublishedDate: "2025-01-01",
			endPublishedDate: "2026-01-01",
		});
		expect(response).toMatchObject({
			provider: "exa",
			answer: "**Exa result**: Exa summary",
			requestId: "exa-request",
			sources: [
				{
					title: "Exa result",
					url: "https://example.com/exa",
					snippet: "Exa summary",
					author: "Example author",
				},
			],
		});
	});

	it("rejects without EXA_API_KEY instead of using the dropped MCP fallback", async () => {
		delete process.env.EXA_API_KEY;
		await expect(new ExaProvider().search({ query: "keyless exa", fetch: () => Promise.reject() })).rejects.toThrow(
			/EXA_API_KEY/,
		);
	});
});

describe("Jina provider", () => {
	it("uses X-Site, bearer auth, and maps the response envelope", async () => {
		process.env.JINA_API_KEY = "jina-key";
		let url: URL | undefined;
		let headers: Headers | undefined;
		const fetch: FetchImpl = (input, init) => {
			url = new URL(input.toString());
			headers = new Headers(init?.headers);
			return Promise.resolve(
				new Response(
					JSON.stringify({
						code: 200,
						data: [
							{
								title: "Jina result",
								url: "https://example.com/jina",
								description: "Jina snippet",
							},
						],
					}),
					{ status: 200 },
				),
			);
		};

		const response = await new JinaProvider().search({
			query: "runtime site:example.com",
			numSearchResults: 2,
			fetch,
		});

		expect(url?.pathname).toBe("/runtime");
		expect(url?.searchParams.get("count")).toBe("2");
		expect(headers?.get("Authorization")).toBe("Bearer jina-key");
		expect(headers?.get("X-Site")).toBe("example.com");
		expect(response).toMatchObject({
			provider: "jina",
			sources: [{ title: "Jina result", url: "https://example.com/jina", snippet: "Jina snippet" }],
		});
	});
});

describe("keyed provider HTTP classification", () => {
	const cases = [
		{
			name: "brave",
			env: "BRAVE_API_KEY",
			provider: () => new BraveProvider(),
		},
		{
			name: "tavily",
			env: "TAVILY_API_KEY",
			provider: () => new TavilyProvider(),
		},
		{
			name: "exa",
			env: "EXA_API_KEY",
			provider: () => new ExaProvider(),
		},
		{
			name: "jina",
			env: "JINA_API_KEY",
			provider: () => new JinaProvider(),
		},
	] as const;

	it.each(cases)("classifies quota responses from $name", async ({ env, provider }) => {
		process.env[env] = "key";
		const promise = provider().search({
			query: "quota",
			fetch: () => Promise.resolve(new Response("quota exceeded", { status: 429 })),
		});
		await expect(promise).rejects.toMatchObject({
			status: 429,
			message: expect.stringContaining("credits exhausted"),
		});
	});

	it.each(cases.flatMap((entry) => [401, 402, 403].map((status) => ({ ...entry, status }))))(
		"classifies HTTP $status from $name",
		async ({ env, provider, status }) => {
			process.env[env] = "key";
			const promise = provider().search({
				query: "auth",
				fetch: () => Promise.resolve(new Response("denied", { status })),
			});
			await expect(promise).rejects.toSatisfy(
				(error: unknown) => error instanceof SearchProviderError && error.status === status,
			);
		},
	);
});
