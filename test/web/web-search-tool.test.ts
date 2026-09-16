import { afterEach, describe, expect, it } from "vitest";
import { createWebSearchTool, WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";
import {
	clearRegisteredSearchProviders,
	registerSearchProvider,
	setExcludedSearchProviders,
	setSearchProviderOrder,
} from "../../src/web/search/provider.ts";
import type { SearchProviderContract } from "../../src/web/search/providers/base.ts";
import { SearchProviderError, type SearchResponse } from "../../src/web/search/types.ts";

function fakeProvider(id: "duckduckgo" | "brave", response: SearchResponse): SearchProviderContract {
	return {
		id,
		label: `Fake ${id}`,
		isAvailable: () => true,
		search: async () => response,
	};
}

afterEach(() => {
	clearRegisteredSearchProviders();
	setSearchProviderOrder([]);
	setExcludedSearchProviders([]);
});

describe("web_search tool", () => {
	it("exposes the oh-my-pi tool contract", () => {
		const tool = createWebSearchTool();
		expect(tool.name).toBe("web_search");
		expect(tool.label).toBe("Web Search");
		expect(tool.description).toContain("Web search");
		const props = (tool.parameters as { properties: Record<string, unknown> }).properties;
		expect(Object.keys(props)).toEqual(
			expect.arrayContaining(["query", "recency", "limit", "num_search_results", "provider"]),
		);
		expect((tool.parameters as { required?: string[] }).required).toEqual(["query"]);
	});

	it("returns formatted sources for a successful search", async () => {
		registerSearchProvider(
			fakeProvider("duckduckgo", {
				provider: "duckduckgo",
				sources: [{ title: "AutoRAG repo", url: "https://github.com/Marker-Inc-Korea/AutoRAG", snippet: "RAG" }],
			}),
		);
		setSearchProviderOrder(["duckduckgo"]);
		const tool = createWebSearchTool();
		const result = await tool.execute("call-1", { query: "autorag" });
		const details = result.details as {
			method: string;
			provider: string;
			resultCount: number;
			sources: string[];
			available: boolean;
		};
		expect(details.method).toBe(WEB_SEARCH_TOOL_NAME);
		expect(details.provider).toBe("duckduckgo");
		expect(details.resultCount).toBe(1);
		expect(details.sources).toEqual(["https://github.com/Marker-Inc-Korea/AutoRAG"]);
		expect(details.available).toBe(true);
		expect(result.content[0]?.text).toContain("[1] AutoRAG repo");
	});

	it("never calls a provider for an empty query", async () => {
		let called = false;
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake",
			isAvailable: () => true,
			search: async () => {
				called = true;
				return { provider: "duckduckgo", sources: [] };
			},
		});
		setSearchProviderOrder(["duckduckgo"]);
		const tool = createWebSearchTool();
		const result = await tool.execute("call-2", { query: "   " });
		expect(called).toBe(false);
		expect(result.content[0]?.text).toContain("empty");
		expect((result.details as { resultCount: number }).resultCount).toBe(0);
	});

	it("reports unavailability without leaking configuration paths when every provider fails", async () => {
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake",
			isAvailable: () => true,
			search: async () => {
				throw new SearchProviderError("duckduckgo", "duckduckgo bot challenge", 429);
			},
		});
		setSearchProviderOrder(["duckduckgo"]);
		const tool = createWebSearchTool();
		const result = await tool.execute("call-3", { query: "doomed query" });
		const details = result.details as { resultCount: number; sources: string[]; available: boolean; error?: string };
		expect(details.available).toBe(false);
		expect(details.resultCount).toBe(0);
		expect(details.sources).toEqual([]);
		expect(details.error).toContain("duckduckgo");
		expect(result.content[0]?.text).toContain("unavailable");
	});

	it("applies order and exclusion options before searching", async () => {
		let ddgCalled = false;
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake",
			isAvailable: () => true,
			search: async () => {
				ddgCalled = true;
				return { provider: "duckduckgo", sources: [] };
			},
		});
		registerSearchProvider(fakeProvider("brave", { provider: "brave", sources: [{ title: "t", url: "https://b.example" }] }));
		const tool = createWebSearchTool({ order: ["brave", "duckduckgo"], exclude: ["duckduckgo"] });
		const result = await tool.execute("call-4", { query: "option routing" });
		expect(ddgCalled).toBe(false);
		expect((result.details as { provider: string }).provider).toBe("brave");
	});

	it("forwards recency and result-count hints to the provider", async () => {
		let seen: { recency?: string; numSearchResults?: number } = {};
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake",
			isAvailable: () => true,
			search: async (params) => {
				seen = { recency: params.recency, numSearchResults: params.numSearchResults };
				return { provider: "duckduckgo", sources: [{ title: "t", url: "https://d.example" }] };
			},
		});
		setSearchProviderOrder(["duckduckgo"]);
		const tool = createWebSearchTool();
		await tool.execute("call-5", { query: "hints", recency: "week", num_search_results: 5 });
		expect(seen).toEqual({ recency: "week", numSearchResults: 5 });
	});
});
