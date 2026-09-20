/**
 * Live E2E: real-internet proof for the web tools.
 *
 * Gated on AUTORAG_WEB_LIVE=1 so the offline suite never touches the
 * network. Run explicitly:
 *
 *   AUTORAG_WEB_LIVE=1 bun run vitest run test/live-e2e/web-search.test.ts
 *
 * Acceptance bar (PR #1592 review, comment 5699346366): with every
 * search-provider env var unset and no model credential, `web_search` must
 * return real results from a plain residential connection on REPEATED runs
 * (not 1-in-2). The chain achieves that without any key through the
 * always-on routes (perplexity anonymous, parallel MCP), model-native
 * reuse when the agent's model credential exists, and headless-browser
 * escalation for bot-challenged scraped engines.
 */
import { afterEach, describe, expect, it } from "vitest";
import { renderUrl } from "../../src/web/fetch/render.ts";
import { ensureWebSearchBrowserLoader, getWebSearchBrowserDiagnostic } from "../../src/web/search/browser-loader.ts";
import { executeWebSearch } from "../../src/web/search/index.ts";

const LIVE = process.env.AUTORAG_WEB_LIVE === "1";

/** Env-gated provider configuration that must be absent for the credential-free proof. */
const GATED_PROVIDER_ENVS = ["SEARXNG_ENDPOINT", "SEARXNG_TOKEN", "AUTORAG_WEB_BROWSER_PATH"] as const;

/** Model credential envs neutralized in-process so the winner must be a zero-credential route. */
const MODEL_CREDENTIAL_ENVS = [
	"ANTHROPIC_API_KEY",
	"OPENAI_API_KEY",
	"GEMINI_API_KEY",
	"GOOGLE_API_KEY",
	"XAI_API_KEY",
] as const;

/** Providers allowed to win the auto chain with zero credentials of any kind. */
const ZERO_CREDENTIAL_PROVIDERS = ["perplexity", "parallel", "startpage", "duckduckgo", "ecosia", "google", "mojeek"];

const savedModelEnv = new Map<string, string | undefined>();

function neutralizeModelCredentials(): void {
	for (const name of MODEL_CREDENTIAL_ENVS) {
		savedModelEnv.set(name, process.env[name]);
		delete process.env[name];
	}
}

afterEach(() => {
	for (const [name, value] of savedModelEnv) {
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	savedModelEnv.clear();
});

describe.skipIf(!LIVE)("web tools live e2e (real internet)", () => {
	it("web_search returns real results with zero credentials on REPEATED runs", async () => {
		for (const name of GATED_PROVIDER_ENVS) {
			expect(process.env[name], `${name} must be unset for the credential-free proof`).toBeUndefined();
		}
		neutralizeModelCredentials();
		// Three consecutive runs: the review measured 1-in-2 flakiness on the
		// old chain, so a single green run proves nothing.
		for (let run = 1; run <= 3; run++) {
			const result = await executeWebSearch(
				{ query: "AutoRAG Marker-Inc-Korea GitHub repository" },
				{ timeoutMs: 60_000 },
			);
			expect(result.details.error, `run ${run} must succeed`).toBeUndefined();
			expect(result.details.response.sources.length).toBeGreaterThan(0);
			for (const source of result.details.response.sources) {
				expect(source.url).toMatch(/^https?:\/\//);
				expect(source.title.length).toBeGreaterThan(0);
			}
			expect(
				ZERO_CREDENTIAL_PROVIDERS,
				`run ${run} winner must be a zero-credential route, got ${result.details.response.provider}`,
			).toContain(result.details.response.provider);
		}
	}, 240_000);

	it("headless-browser escalation is either armed or an explicit diagnostic, never a crash", async () => {
		const installed = await ensureWebSearchBrowserLoader();
		if (installed) {
			expect(getWebSearchBrowserDiagnostic()).toBeUndefined();
		} else {
			// No local Chrome/Chromium: the degrade contract is a recorded
			// diagnostic with the chain continuing on plain fetches.
			expect(getWebSearchBrowserDiagnostic()).toMatch(/Chrome|Chromium|browser/i);
		}
	}, 30_000);

	it("web_search public fan-out consolidates answering engines or degrades with per-engine diagnostics", async () => {
		// The fan-out is a best-effort aggregate over the scraped engines,
		// which are bot-challenged on some egress IPs (engine-side policy, not
		// a client bug — the reliability bar lives in the zero-credential
		// auto-chain test above). Its honest live contract: consolidate and
		// dedupe whatever answers, or fail cleanly naming the engines.
		const result = await executeWebSearch(
			{ query: "Model Context Protocol specification", provider: "public" },
			{ timeoutMs: 60_000 },
		);
		if (result.details.error === undefined) {
			expect(result.details.response.provider).toBe("public");
			expect(result.details.response.sources.length).toBeGreaterThan(0);
			const urls = result.details.response.sources.map((source) => source.url);
			expect(new Set(urls).size).toBe(urls.length);
		} else {
			expect(result.details.error).toMatch(/public engines failed|no renderable search content/i);
		}
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
