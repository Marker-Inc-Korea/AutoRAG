import { afterEach, describe, expect, it } from "vitest";
import { executeWebSearch } from "../../src/web/search/index.ts";
import { clearRegisteredSearchProviders, registerSearchProvider } from "../../src/web/search/provider.ts";
import type { SearchParams, SearchProviderContract } from "../../src/web/search/providers/base.ts";
import { classifyProviderHttpError, normalizeSearchText } from "../../src/web/search/providers/utils.ts";
import {
	SEARCH_PROVIDER_ORDER,
	SearchProviderError,
	type SearchProviderId,
	type SearchResponse,
} from "../../src/web/search/types.ts";

/**
 * Routing for one call: run `ids` in order and exclude every other built-in
 * provider, so chain tests stay hermetic (no real network) and carry their
 * own configuration instead of mutating shared state.
 */
function chain(...ids: SearchProviderId[]): { order: SearchProviderId[]; exclude: SearchProviderId[] } {
	return { order: ids, exclude: SEARCH_PROVIDER_ORDER.filter((id) => !ids.includes(id)) };
}

function fakeProvider(
	id: SearchProviderId,
	behavior: (params: SearchParams) => Promise<SearchResponse>,
	available = true,
): SearchProviderContract {
	return {
		id,
		label: `Fake ${id}`,
		isAvailable: () => available,
		search: behavior,
	};
}

function oneSourceResponse(provider: SearchProviderId): SearchResponse {
	return {
		provider,
		sources: [{ title: "Example", url: "https://example.com/doc", snippet: "an example result" }],
	};
}

afterEach(() => {
	clearRegisteredSearchProviders();
});

describe("executeWebSearch provider chain", () => {
	it("returns the first available provider's response", async () => {
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		const result = await executeWebSearch({ query: "autorag librarian" }, chain("duckduckgo"));
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.provider).toBe("duckduckgo");
		const text = result.content[0]?.text ?? "";
		expect(text).toContain("https://example.com/doc");
		expect(text).toContain("[1] Example");
	});

	it("advances past a quota-exhausted provider to the next in the chain", async () => {
		registerSearchProvider(
			fakeProvider("startpage", async () => {
				throw new SearchProviderError("startpage", "startpage: 402 credits exhausted", 402);
			}),
		);
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		const result = await executeWebSearch({ query: "quota fallback" }, chain("startpage", "duckduckgo"));
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("skips providers that report unavailable (no credentials) in the auto chain", async () => {
		registerSearchProvider(
			fakeProvider(
				"startpage",
				async () => {
					throw new Error("must not be called");
				},
				false,
			),
		);
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		const result = await executeWebSearch({ query: "skip unavailable" }, chain("startpage", "duckduckgo"));
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("treats a response with no renderable content as a failure and falls through", async () => {
		registerSearchProvider(fakeProvider("startpage", async () => ({ provider: "startpage", sources: [] })));
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		const result = await executeWebSearch({ query: "empty first" }, chain("startpage", "duckduckgo"));
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("treats search-attempt metadata without an answer or source as a failure", async () => {
		// A model-native provider can report the queries it ran and nothing
		// else; that is evidence of an attempt, not a result, so the chain must
		// keep going instead of presenting it as the answer.
		registerSearchProvider(
			fakeProvider("startpage", async () => ({
				provider: "startpage",
				sources: [],
				searchQueries: ["autorag librarian"],
				relatedQuestions: ["what is autorag?"],
			})),
		);
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		const result = await executeWebSearch({ query: "metadata only" }, chain("startpage", "duckduckgo"));
		expect(result.details.response.provider).toBe("duckduckgo");
		expect(result.details.response.sources).toHaveLength(1);
	});

	it("stops the chain when the overall deadline elapses", async () => {
		let secondCalled = false;
		registerSearchProvider(
			fakeProvider("startpage", async () => {
				await new Promise((resolve) => setTimeout(resolve, 30));
				throw new SearchProviderError("startpage", "startpage: 429 rate limited", 429);
			}),
		);
		registerSearchProvider(
			fakeProvider("duckduckgo", async () => {
				secondCalled = true;
				return oneSourceResponse("duckduckgo");
			}),
		);
		const result = await executeWebSearch(
			{ query: "deadline" },
			{ ...chain("startpage", "duckduckgo"), totalTimeoutMs: 10 },
		);
		expect(secondCalled).toBe(false);
		expect(result.details.error).toContain("deadline elapsed");
	});

	it("fails explicitly selected providers when they are unavailable", async () => {
		registerSearchProvider(fakeProvider("startpage", async () => oneSourceResponse("startpage"), false));
		const result = await executeWebSearch({ query: "x", provider: "startpage" });
		expect(result.details.error).toBeDefined();
		expect(result.content[0]?.text).toContain("unavailable");
	});

	it("summarizes every provider failure when the whole chain fails", async () => {
		registerSearchProvider(
			fakeProvider("startpage", async () => {
				throw new SearchProviderError("startpage", "startpage: 429 rate limited", 429);
			}),
		);
		registerSearchProvider(
			fakeProvider("duckduckgo", async () => {
				throw new SearchProviderError("duckduckgo", "duckduckgo bot challenge", 429);
			}),
		);
		const result = await executeWebSearch({ query: "doomed" }, chain("startpage", "duckduckgo"));
		expect(result.details.error).toContain("All web search providers failed");
		expect(result.details.error).toContain("startpage");
		expect(result.details.error).toContain("duckduckgo");
	});

	it("never calls excluded providers", async () => {
		let called = false;
		registerSearchProvider(
			fakeProvider("duckduckgo", async () => {
				called = true;
				return oneSourceResponse("duckduckgo");
			}),
		);
		const result = await executeWebSearch(
			{ query: "excluded" },
			{ order: ["duckduckgo"], exclude: [...SEARCH_PROVIDER_ORDER] },
		);
		expect(called).toBe(false);
		expect(result.details.error).toBeDefined();
	});

	it("post-filters with query constraints and notes relaxed dimensions", async () => {
		registerSearchProvider(
			fakeProvider("duckduckgo", async () => ({
				provider: "duckduckgo",
				sources: [{ title: "Off site", url: "https://other.example/page" }],
			})),
		);
		const result = await executeWebSearch({ query: "anything site:example.com" }, chain("duckduckgo"));
		// site:example.com would eliminate every result, so it is relaxed and noted.
		expect(result.content[0]?.text).toContain("Note:");
		expect(result.content[0]?.text).toContain("site:example.com");
		expect(result.details.response.sources).toHaveLength(1);
	});
});

describe("classifyProviderHttpError", () => {
	it("maps quota and auth signals to compact provider errors", () => {
		expect(classifyProviderHttpError("startpage", 402, "")?.message).toContain("credits exhausted");
		expect(classifyProviderHttpError("google", 200, "Your quota is exceeded")?.status).toBe(200);
		expect(classifyProviderHttpError("ecosia", 401, "")?.message).toContain("401 unauthorized");
		expect(classifyProviderHttpError("mojeek", 403, "")?.message).toContain("403 forbidden");
	});

	it("returns null for ordinary failures", () => {
		expect(classifyProviderHttpError("startpage", 500, "internal error")).toBeNull();
		expect(classifyProviderHttpError("startpage", 429, "slow down")).toBeNull();
	});
});

describe("normalizeSearchText", () => {
	it("collapses whitespace and drops blanks", () => {
		expect(normalizeSearchText("  a\n b\tc ")).toBe("a b c");
		expect(normalizeSearchText("   ")).toBeUndefined();
		expect(normalizeSearchText(42)).toBeUndefined();
	});
});
