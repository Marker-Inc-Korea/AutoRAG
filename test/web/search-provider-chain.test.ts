import { afterEach, describe, expect, it } from "vitest";
import { executeWebSearch } from "../../src/web/search/index.ts";
import {
	clearRegisteredSearchProviders,
	registerSearchProvider,
	setExcludedSearchProviders,
	setSearchProviderOrder,
} from "../../src/web/search/provider.ts";
import type { SearchParams, SearchProviderContract } from "../../src/web/search/providers/base.ts";
import { classifyProviderHttpError, normalizeSearchText } from "../../src/web/search/providers/utils.ts";
import { SearchProviderError, type SearchProviderId, type SearchResponse } from "../../src/web/search/types.ts";

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
	setSearchProviderOrder([]);
	setExcludedSearchProviders([]);
});

describe("executeWebSearch provider chain", () => {
	it("returns the first available provider's response", async () => {
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		setSearchProviderOrder(["duckduckgo"]);
		const result = await executeWebSearch({ query: "autorag librarian" });
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.provider).toBe("duckduckgo");
		const text = result.content[0]?.text ?? "";
		expect(text).toContain("https://example.com/doc");
		expect(text).toContain("[1] Example");
	});

	it("advances past a quota-exhausted provider to the next in the chain", async () => {
		registerSearchProvider(
			fakeProvider("brave", async () => {
				throw new SearchProviderError("brave", "brave: 402 credits exhausted", 402);
			}),
		);
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		setSearchProviderOrder(["brave", "duckduckgo"]);
		const result = await executeWebSearch({ query: "quota fallback" });
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("skips providers that report unavailable (no credentials) in the auto chain", async () => {
		registerSearchProvider(
			fakeProvider(
				"brave",
				async () => {
					throw new Error("must not be called");
				},
				false,
			),
		);
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		setSearchProviderOrder(["brave", "duckduckgo"]);
		const result = await executeWebSearch({ query: "skip unavailable" });
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("treats a response with no renderable content as a failure and falls through", async () => {
		registerSearchProvider(fakeProvider("brave", async () => ({ provider: "brave", sources: [] })));
		registerSearchProvider(fakeProvider("duckduckgo", async () => oneSourceResponse("duckduckgo")));
		setSearchProviderOrder(["brave", "duckduckgo"]);
		const result = await executeWebSearch({ query: "empty first" });
		expect(result.details.response.provider).toBe("duckduckgo");
	});

	it("fails explicitly selected providers when they are unavailable", async () => {
		registerSearchProvider(fakeProvider("brave", async () => oneSourceResponse("brave"), false));
		const result = await executeWebSearch({ query: "x", provider: "brave" });
		expect(result.details.error).toBeDefined();
		expect(result.content[0]?.text).toContain("unavailable");
	});

	it("summarizes every provider failure when the whole chain fails", async () => {
		registerSearchProvider(
			fakeProvider("brave", async () => {
				throw new SearchProviderError("brave", "brave: 429 rate limited", 429);
			}),
		);
		registerSearchProvider(
			fakeProvider("duckduckgo", async () => {
				throw new SearchProviderError("duckduckgo", "duckduckgo bot challenge", 429);
			}),
		);
		setSearchProviderOrder(["brave", "duckduckgo"]);
		const result = await executeWebSearch({ query: "doomed" });
		expect(result.details.error).toContain("All web search providers failed");
		expect(result.details.error).toContain("brave");
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
		setSearchProviderOrder(["duckduckgo"]);
		setExcludedSearchProviders(["duckduckgo"]);
		const result = await executeWebSearch({ query: "excluded" });
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
		setSearchProviderOrder(["duckduckgo"]);
		const result = await executeWebSearch({ query: "anything site:example.com" });
		// site:example.com would eliminate every result, so it is relaxed and noted.
		expect(result.content[0]?.text).toContain("Note:");
		expect(result.content[0]?.text).toContain("site:example.com");
		expect(result.details.response.sources).toHaveLength(1);
	});
});

describe("classifyProviderHttpError", () => {
	it("maps quota and auth signals to compact provider errors", () => {
		expect(classifyProviderHttpError("brave", 402, "")?.message).toContain("credits exhausted");
		expect(classifyProviderHttpError("tavily", 200, "Your quota is exceeded")?.status).toBe(200);
		expect(classifyProviderHttpError("exa", 401, "")?.message).toContain("401 unauthorized");
		expect(classifyProviderHttpError("kagi", 403, "")?.message).toContain("403 forbidden");
	});

	it("returns null for ordinary failures", () => {
		expect(classifyProviderHttpError("brave", 500, "internal error")).toBeNull();
		expect(classifyProviderHttpError("brave", 429, "slow down")).toBeNull();
	});
});

describe("normalizeSearchText", () => {
	it("collapses whitespace and drops blanks", () => {
		expect(normalizeSearchText("  a\n b\tc ")).toBe("a b c");
		expect(normalizeSearchText("   ")).toBeUndefined();
		expect(normalizeSearchText(42)).toBeUndefined();
	});
});
