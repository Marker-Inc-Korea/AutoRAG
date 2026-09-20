import { afterEach, describe, expect, it } from "vitest";
import { createWebSearchTool, WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";
import { clearRegisteredSearchProviders, registerSearchProvider } from "../../src/web/search/provider.ts";
import type { SearchProviderContract } from "../../src/web/search/providers/base.ts";
import {
	SEARCH_PROVIDER_ORDER,
	SearchProviderError,
	type SearchProviderId,
	type SearchResponse,
} from "../../src/web/search/types.ts";

/** Tool options that run only `keep`, so tool tests stay hermetic (no real network). */
function isolatedTo(...keep: SearchProviderId[]): { order: SearchProviderId[]; exclude: SearchProviderId[] } {
	return { order: keep, exclude: SEARCH_PROVIDER_ORDER.filter((id) => !keep.includes(id)) };
}

function fakeProvider(id: "duckduckgo" | "google", response: SearchResponse): SearchProviderContract {
	return {
		id,
		label: `Fake ${id}`,
		isAvailable: () => true,
		search: async () => response,
	};
}

afterEach(() => {
	clearRegisteredSearchProviders();
});

describe("web_search tool", () => {
	it("exposes the oh-my-pi tool contract", () => {
		const tool = createWebSearchTool();
		expect(tool.name).toBe("web_search");
		expect(tool.label).toBe("Web Search");
		expect(tool.description).toContain("Web search");
		const props = (tool.parameters as { properties: Record<string, unknown> }).properties;
		expect(Object.keys(props)).toEqual(expect.arrayContaining(["query", "recency", "limit", "num_search_results"]));
		// The agent must not pick providers: routing belongs to the auto chain
		// (an explicit model choice bypasses quota/challenge fallback).
		expect(Object.keys(props)).not.toContain("provider");
		expect((tool.parameters as { required?: string[] }).required).toEqual(["query"]);
	});

	it("returns formatted sources for a successful search", async () => {
		registerSearchProvider(
			fakeProvider("duckduckgo", {
				provider: "duckduckgo",
				sources: [{ title: "AutoRAG repo", url: "https://github.com/Marker-Inc-Korea/AutoRAG", snippet: "RAG" }],
			}),
		);
		const tool = createWebSearchTool(isolatedTo("duckduckgo"));
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
		expect((result.content[0] as { text: string }).text).toContain("[1] AutoRAG repo");
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
		const tool = createWebSearchTool(isolatedTo("duckduckgo"));
		const result = await tool.execute("call-2", { query: "   " });
		expect(called).toBe(false);
		expect((result.content[0] as { text: string }).text).toContain("empty");
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
		const tool = createWebSearchTool(isolatedTo("duckduckgo"));
		const result = await tool.execute("call-3", { query: "doomed query" });
		const details = result.details as unknown as {
			resultCount: number;
			sources: string[];
			available: boolean;
			error?: string;
		};
		expect(details.available).toBe(false);
		expect(details.resultCount).toBe(0);
		expect(details.sources).toEqual([]);
		expect(details.error).toContain("duckduckgo");
		expect((result.content[0] as { text: string }).text).toContain("unavailable");
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
		registerSearchProvider(
			fakeProvider("google", { provider: "google", sources: [{ title: "t", url: "https://b.example" }] }),
		);
		const tool = createWebSearchTool({ order: ["google", "duckduckgo"], exclude: ["duckduckgo"] });
		const result = await tool.execute("call-4", { query: "option routing" });
		expect(ddgCalled).toBe(false);
		expect((result.details as { provider: string }).provider).toBe("google");
	});

	it("keeps routing on the tool instance so a configured tool cannot reroute a default one", async () => {
		let googleCalled = false;
		registerSearchProvider({
			id: "google",
			label: "Fake",
			isAvailable: () => true,
			search: async () => {
				googleCalled = true;
				return { provider: "google", sources: [{ title: "g", url: "https://g.example" }] };
			},
		});
		registerSearchProvider(
			fakeProvider("duckduckgo", {
				provider: "duckduckgo",
				sources: [{ title: "d", url: "https://d.example" }],
			}),
		);
		// A configured tool exists first; the default tool created afterwards
		// must still walk its own chain, not the configured one.
		createWebSearchTool({ order: ["google"], exclude: ["duckduckgo"] });
		const defaultTool = createWebSearchTool(isolatedTo("duckduckgo", "google"));
		const result = await defaultTool.execute("call-6", { query: "instance routing" });
		expect(googleCalled).toBe(false);
		expect((result.details as { provider: string }).provider).toBe("duckduckgo");
	});

	it("passes its own agent model credential to model-native providers", async () => {
		let seenKey: string | undefined;
		registerSearchProvider({
			id: "anthropic",
			label: "Fake",
			isAvailable: (context) => context?.modelAuth?.provider === "anthropic",
			search: async (params) => {
				seenKey = params.modelAuth?.apiKey;
				return { provider: "anthropic", sources: [{ title: "a", url: "https://a.example" }] };
			},
		});
		const tool = createWebSearchTool({
			...isolatedTo("anthropic"),
			modelAuth: () => ({ provider: "anthropic", apiKey: "sk-agent-a" }),
		});
		const result = await tool.execute("call-7", { query: "credential" });
		expect(seenKey).toBe("sk-agent-a");
		expect((result.details as { provider: string }).provider).toBe("anthropic");
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
		const tool = createWebSearchTool(isolatedTo("duckduckgo"));
		await tool.execute("call-5", { query: "hints", recency: "week", num_search_results: 5 });
		expect(seen).toEqual({ recency: "week", numSearchResults: 5 });
	});
});
