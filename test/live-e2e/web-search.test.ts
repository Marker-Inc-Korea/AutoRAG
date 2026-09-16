/**
 * Live E2E: real-internet proof for the web tools.
 *
 * Gated on AUTORAG_WEB_LIVE=1 so the offline suite never touches the
 * network. Run explicitly:
 *
 *   AUTORAG_WEB_LIVE=1 bun run vitest run test/live-e2e/web-search.test.ts
 *
 * These tests prove the credential-free default path end to end: no API key
 * is set, the provider chain must still return real web results, and
 * web_fetch must render a real page. They also prove the explicit Public
 * Web fan-out consolidates multiple credential-free engines.
 */
import { describe, expect, it } from "vitest";
import { renderUrl } from "../../src/web/fetch/render.ts";
import { executeWebSearch } from "../../src/web/search/index.ts";

const LIVE = process.env.AUTORAG_WEB_LIVE === "1";
const KEYED_PROVIDER_ENVS = [
	"BRAVE_API_KEY",
	"TAVILY_API_KEY",
	"EXA_API_KEY",
	"JINA_API_KEY",
	"KAGI_API_KEY",
	"KIMI_SEARCH_API_KEY",
	"MOONSHOT_SEARCH_API_KEY",
	"SEARXNG_ENDPOINT",
] as const;

describe.skipIf(!LIVE)("web tools live e2e (real internet)", () => {
	it("web_search returns real results with no API key configured", async () => {
		for (const name of KEYED_PROVIDER_ENVS) {
			expect(process.env[name], `${name} must be unset for the credential-free proof`).toBeUndefined();
		}
		const result = await executeWebSearch({ query: "AutoRAG Marker-Inc-Korea GitHub repository" }, { timeoutMs: 60_000 });
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.sources.length).toBeGreaterThan(0);
		for (const source of result.details.response.sources) {
			expect(source.url).toMatch(/^https?:\/\//);
			expect(source.title.length).toBeGreaterThan(0);
		}
		// The winning provider must be one of the credential-free engines.
		expect(["startpage", "duckduckgo", "ecosia", "google", "mojeek"]).toContain(result.details.response.provider);
	}, 90_000);

	it("web_search honors the explicit public fan-out and dedupes across engines", async () => {
		const result = await executeWebSearch({ query: "Model Context Protocol specification", provider: "public" }, { timeoutMs: 60_000 });
		expect(result.details.error).toBeUndefined();
		expect(result.details.response.provider).toBe("public");
		expect(result.details.response.sources.length).toBeGreaterThan(0);
		const urls = result.details.response.sources.map((source) => source.url);
		expect(new Set(urls).size).toBe(urls.length);
	}, 90_000);

	it("web_fetch renders a real page as text", async () => {
		const rendered = await renderUrl("https://example.com/", { timeoutSeconds: 30 });
		expect(rendered.method).not.toBe("failed");
		expect(rendered.content).toContain("Example Domain");
	}, 60_000);

	it("web_fetch reports HTTP failures without throwing", async () => {
		const rendered = await renderUrl("https://example.com/definitely-missing-404", { timeoutSeconds: 30 });
		expect(rendered.method).toBe("failed");
		expect(rendered.notes.join(" ")).toContain("404");
	}, 60_000);
});
