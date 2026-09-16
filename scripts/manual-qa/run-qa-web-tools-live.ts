/**
 * Live manual QA for the `web_search` + `web_fetch` agent tools.
 *
 * Drives a real AutoRAGAgent with the operator's configured model (via
 * AUTORAG_CONFIG or the default ~/.autorag/config.json) and an isolated
 * empty corpus, asking a question that requires current public web
 * information. The agent must actively call the web tools: the harness
 * wraps global fetch and records every outbound HTTP request, then asserts
 * that (a) at least one request hit a search-provider endpoint and (b) the
 * curated answer carries a public http(s) source URL.
 *
 * Prerequisites: a resolvable agent model (same as `autorag search`).
 *
 * Usage:
 *   bun scripts/manual-qa/run-qa-web-tools-live.ts
 *   WEB_QA_QUERY="latest AutoRAG release" bun scripts/manual-qa/run-qa-web-tools-live.ts
 *   WEB_QA_EVIDENCE=.omo/evidence/qa-web-tools-live.json bun scripts/manual-qa/run-qa-web-tools-live.ts
 */
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildAgentOptions, resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const query =
	process.env.WEB_QA_QUERY ??
	"What is the latest published release of the AutoRAG project on GitHub? Search the public web and read the release page.";
const evidencePath = process.env.WEB_QA_EVIDENCE ?? join(".omo", "evidence", "qa-web-tools-live.json");

/** Search-provider hosts the QA recognizes as web_search traffic. */
const SEARCH_PROVIDER_HOSTS = [
	"html.duckduckgo.com",
	"duckduckgo.com",
	"www.startpage.com",
	"www.google.com",
	"www.ecosia.org",
	"www.mojeek.com",
	"www.perplexity.ai",
	"search.parallel.ai",
	"api.anthropic.com",
	"generativelanguage.googleapis.com",
	"api.openai.com",
	"api.x.ai",
];

interface OutboundRequest {
	readonly url: string;
	readonly method: string;
	readonly status: number;
}

const outbound: OutboundRequest[] = [];
const realFetch = globalThis.fetch;
globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
	const response = await realFetch(input, init);
	const url = typeof input === "string" ? input : input instanceof URL ? input.href : input.url;
	outbound.push({ url, method: init?.method ?? "GET", status: response.status });
	return response;
}) as typeof fetch;

const root = mkdtempSync(join(tmpdir(), "autorag-web-qa-"));
try {
	mkdirSync(join(root, "docs"), { recursive: true });
	const modelProvider = process.env.WEB_QA_MODEL_PROVIDER ?? "openrouter";
	const modelId = process.env.WEB_QA_MODEL_ID ?? "moonshotai/kimi-k2.5";
	const config = resolveConfig({ flags: { "model-provider": modelProvider, "model-id": modelId }, cwd: root });
	const isolated = {
		...config,
		searchPaths: [join(root, "docs")],
		workspacePath: join(root, "workspace"),
		memoryPath: join(root, "memory.json"),
		minSync: { enabled: false } as const,
		jikji: false as const,
	};
	const resolvedModel = resolveAgentModel(isolated);
	const agent = new AutoRAGAgent({
		...buildAgentOptions(isolated),
		model: resolvedModel.model,
		...(resolvedModel.apiKey !== undefined ? { apiKey: resolvedModel.apiKey } : {}),
		...(resolvedModel.providerApiKeys !== undefined ? { providerApiKeys: resolvedModel.providerApiKeys } : {}),
	});

	const progress: string[] = [];
	let final:
		| Awaited<ReturnType<AutoRAGAgent["searchDocuments"]>>
		| undefined;
	for await (const event of agent.searchDocumentsStream(query, {})) {
		if (event.type === "progress") progress.push(event.text);
		if (event.type === "complete") final = event.response;
	}
	if (final === undefined) throw new Error("search ended without a complete event");

	const searchRequests = outbound.filter((request) =>
		SEARCH_PROVIDER_HOSTS.some((host) => request.url.includes(host)),
	);
	const registrySources: string[] = [];
	for (const entry of agent.getResultRegistry(final.sessionId).values()) {
		registrySources.push(entry.source);
	}
	const publicSources = registrySources.filter((source) => /^https?:\/\//.test(source));

	const pass = searchRequests.length > 0 && final.results.length > 0 && publicSources.length > 0;

	const evidence = {
		pass,
		query,
		progress,
		answer: final.answer,
		results: final.results.map((result) => ({ number: result.number, title: result.title })),
		mappingSources: registrySources,
		outboundRequests: outbound,
		searchProviderRequests: searchRequests,
	};
	mkdirSync(join(evidencePath, ".."), { recursive: true });
	writeFileSync(evidencePath, JSON.stringify(evidence, null, 2));

	console.log(`progress events: ${progress.length}`);
	console.log(`outbound requests: ${outbound.length} (search providers: ${searchRequests.length})`);
	for (const request of searchRequests.slice(0, 5)) {
		console.log(`  search: ${request.method} ${request.url} -> ${request.status}`);
	}
	for (const request of outbound.filter((r) => !searchRequests.includes(r)).slice(0, 5)) {
		console.log(`  fetch: ${request.method} ${request.url} -> ${request.status}`);
	}
	console.log(`mapping sources: ${evidence.mappingSources.join(", ") || "(none)"}`);
	console.log(`evidence: ${evidencePath}`);
	if (!pass) {
		console.error("WEB_TOOLS_LIVE_QA_FAIL");
		process.exit(1);
	}
	console.log("WEB_TOOLS_LIVE_QA_PASS");
} finally {
	globalThis.fetch = realFetch;
	rmSync(root, { recursive: true, force: true });
}
