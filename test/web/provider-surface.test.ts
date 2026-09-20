/**
 * Shipped provider surface: the credential-free path is the only shipped path.
 *
 * PR #1592 review direction (comment 5699346366): vendor-key providers
 * (brave, tavily, exa, jina, kagi, kimi) must not appear in the default
 * chain, the `webSearch.provider|order|exclude` config id set, or the
 * LLM-facing tool schema. SearXNG stays as an explicitly-advanced,
 * env-gated option only.
 */
import { describe, expect, it } from "vitest";
import { createWebSearchTool } from "../../src/agent/web-search-tool.ts";
import { isSearchProviderId, SEARCH_PROVIDER_OPTIONS, SEARCH_PROVIDER_ORDER } from "../../src/web/search/types.ts";

const REMOVED_KEYED_IDS = ["brave", "tavily", "exa", "jina", "kagi", "kimi"] as const;

describe("shipped web-search provider surface", () => {
	it("default chain contains no vendor-key providers", () => {
		for (const id of REMOVED_KEYED_IDS) {
			expect(SEARCH_PROVIDER_ORDER, `${id} must not be in the default chain`).not.toContain(id);
			expect(isSearchProviderId(id), `${id} must not be a recognized provider id`).toBe(false);
		}
	});

	it("searxng remains available only as an explicitly-advanced option", () => {
		expect(isSearchProviderId("searxng")).toBe(true);
		const option = SEARCH_PROVIDER_OPTIONS.find((entry) => entry.value === "searxng");
		expect(option?.description).toMatch(/SEARXNG_ENDPOINT|advanced/i);
	});

	it("LLM-facing tool schema exposes no provider selection at all", () => {
		// Provider routing is the auto chain's job; the model only passes a
		// query, so keyed/gg engine ids can never leak to the LLM surface.
		const tool = createWebSearchTool();
		expect(tool.parameters.properties).not.toHaveProperty("provider");
	});
});
