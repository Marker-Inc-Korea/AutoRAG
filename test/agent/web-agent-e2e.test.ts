import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { WEB_FETCH_TOOL_NAME } from "../../src/agent/web-fetch-tool.ts";
import { WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";
import {
	clearRegisteredSearchProviders,
	registerSearchProvider,
	setExcludedSearchProviders,
	setSearchProviderOrder,
} from "../../src/web/search/provider.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-web-e2e-"));
	registrations = [];
});

afterEach(() => {
	vi.unstubAllGlobals();
	for (const reg of registrations) reg.unregister();
	clearRegisteredSearchProviders();
	setSearchProviderOrder([]);
	setExcludedSearchProviders([]);
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

const POLICY_HTML = `<!doctype html><html><head><title>Refund Policy 2026</title></head><body><main>
<h1>Refund Policy</h1>
<p>Refund exceptions require director approval before payout. Finance acknowledged the policy in the July review.</p>
</main></body></html>`;

describe("AutoRAGAgent actively uses the web tools", () => {
	it("calls web_search and web_fetch during searchDocuments and curates from them", async () => {
		// The provider chain is stubbed through the registry seam; the web_fetch
		// transport is stubbed at global fetch. No network is touched.
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake DuckDuckGo",
			isAvailable: () => true,
			search: async () => ({
				provider: "duckduckgo",
				sources: [
					{
						title: "Refund Policy 2026",
						url: "https://policy.example.com/refunds",
						snippet: "Refund exceptions require director approval before payout.",
					},
				],
			}),
		});
		setSearchProviderOrder(["duckduckgo"]);
		vi.stubGlobal("fetch", async (input: unknown) => {
			const url = String(input);
			if (url.startsWith("https://policy.example.com/")) {
				return new Response(POLICY_HTML, { status: 200, headers: { "content-type": "text/html; charset=utf-8" } });
			}
			return new Response("not found", { status: 404 });
		});

		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall(WEB_SEARCH_TOOL_NAME, { query: "refund director approval policy 2026" })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage([fauxToolCall(WEB_FETCH_TOOL_NAME, { url: "https://policy.example.com/refunds" })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage([
				fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
					answer: "[1] Refund exceptions require director approval before payout.",
					results: [
						{
							number: 1,
							title: "Refund approval rule",
							summary:
								"Refund exceptions require director approval before payout, confirmed on the public policy page.",
							evidence: [{ excerpt: "Refund exceptions require director approval before payout." }],
							confidence: 0.9,
						},
					],
					mapping: [
						{
							number: 1,
							source: "https://policy.example.com/refunds",
							method: WEB_FETCH_TOOL_NAME,
							content: "Refund exceptions require director approval before payout.",
							evidenceRefs: [
								{
									method: WEB_SEARCH_TOOL_NAME,
									source: "https://policy.example.com/refunds",
									excerpt: "Refund exceptions require director approval before payout.",
								},
							],
						},
					],
				}),
			]),
		);

		const agent = new AutoRAGAgent({
			model,
			searchPaths: [root],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: false,
			jikji: false,
		});

		const names = (
			(agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent.state.tools ?? []
		).map((tool) => tool.name);
		expect(names).toContain(WEB_SEARCH_TOOL_NAME);
		expect(names).toContain(WEB_FETCH_TOOL_NAME);

		const response = await agent.searchDocuments("What is the current refund approval policy?", { topK: 2 });
		expect(response.results).toHaveLength(1);
		expect(response.answer).toContain("[1]");
		const registry = agent.getResultRegistry(response.sessionId);
		expect(registry.get(1)?.source).toBe("https://policy.example.com/refunds");
	});
});
