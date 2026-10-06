import { randomUUID } from "node:crypto";
import { once } from "node:events";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	type Context,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { EMIT_FAST_ANSWER_TOOL_NAME } from "../../src/agent/fast-answer-tool.ts";
import {
	DECOMPOSE_QUESTION_ID,
	datasourceQuestionId,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
} from "../../src/agent/query-routing.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import type { DatasourceIndexResult, DatasourceSkill, PollingMetadata } from "../../src/datasource/types.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";
import { clearRegisteredSearchProviders, registerSearchProvider } from "../../src/web/search/provider.ts";
import { SEARCH_PROVIDER_ORDER } from "../../src/web/search/types.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jev-pipeline-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	clearRegisteredSearchProviders();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	registration.setResponses(responses);
	registrations.push(registration);
	return registration.getModel();
}

function lastUserText(context: Context): string {
	const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
	if (lastUser === undefined) return "";
	return typeof lastUser.content === "string"
		? lastUser.content
		: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

function capture(step: FauxResponseStep, prompts: string[]): FauxResponseStep {
	return (context) => {
		prompts.push(lastUserText(context));
		return step as AssistantMessage;
	};
}

function fastAnswer(answer: string, source?: string): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
				answer,
				results:
					source === undefined
						? []
						: [
								{
									number: 1,
									title: "Evidence",
									summary: answer,
									evidence: [{ excerpt: answer }],
									confidence: 0.7,
								},
							],
				...(source === undefined ? {} : { sources: [{ number: 1, source }] }),
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function finalEmit(answer: string, source: string): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer,
				results: [
					{ number: 1, title: "Verified", summary: answer, evidence: [{ excerpt: answer }], confidence: 0.9 },
				],
				mapping: [{ number: 1, source, method: "bash", content: answer }],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function jevRouting(
	route: QueryRoute,
	decomposeProbability: number,
	followUpProbability = 0.9,
	datasourceProbabilities: Readonly<Record<string, number>> = {},
): JevBackend {
	const distribution = { local: 0.1, web: 0.1, direct: 0.1, [route]: 0.8 };
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: { answer: route, distribution, confidence: 0.8 },
		[DECOMPOSE_QUESTION_ID]: { answer: decomposeProbability },
		[FOLLOW_UP_QUESTION_ID]: { answer: followUpProbability },
		...Object.fromEntries(
			Object.entries(datasourceProbabilities).map(([datasourceId, answer]) => [
				datasourceQuestionId(datasourceId),
				{ answer },
			]),
		),
	});
}

/** A chat datasource whose single retrieval method records every query it receives. */
function recordingDatasource(datasourceId: string, description: string) {
	const queries: string[] = [];
	const skill: DatasourceSkill = {
		describe: () => ({
			name: datasourceId,
			type: "chat",
			description,
			capabilities: ["keyword", "polling"],
			tags: [datasourceId],
			status: "active",
			datasourceId,
			instanceId: "default",
		}),
		polling: (): PollingMetadata => ({ mode: "none" }),
		skillManifest: () => ({
			name: `datasource-${datasourceId}`,
			description: `Search ${datasourceId}.`,
			content: `# ${datasourceId}`,
		}),
		index: async (): Promise<DatasourceIndexResult> => ({
			ok: true,
			instanceId: "default",
			skill: datasourceId,
			chunkCount: 1,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [
			{
				describe: () => ({
					name: `${datasourceId}.keyword`,
					type: "bm25" as const,
					description: `${datasourceId} keyword method`,
					status: "active" as const,
					capabilities: ["keyword"],
					datasourceId,
					tags: [datasourceId],
				}),
				retrieve: async (query: string) => {
					queries.push(query);
					return [
						{
							id: `${datasourceId}-${query}`,
							source: `/${datasourceId}/default/${query.replace(/\W+/gu, "-")}`,
							content: `${datasourceId} evidence for ${query}`,
							score: 1,
							metadata: {},
						},
					];
				},
			},
		],
		describeSources: () => [
			{
				source: `/${datasourceId}/default`,
				datasourceId,
				skill: datasourceId,
				instanceId: "default",
				contentType: "chat",
				metadata: {},
			},
		],
	};
	return { queries, skill };
}

/** OpenRouter-compatible rerank endpoint that records requests and reverses the pool order. */
async function startRerankServer() {
	const requests: { query: unknown; documents: unknown[] }[] = [];
	const server = createServer((request, response) => {
		let body = "";
		request.on("data", (chunk) => {
			body += chunk;
		});
		request.on("end", () => {
			const parsed: unknown = JSON.parse(body.length > 0 ? body : "{}");
			const query = typeof parsed === "object" && parsed !== null && "query" in parsed ? parsed.query : undefined;
			const documents =
				typeof parsed === "object" && parsed !== null && "documents" in parsed && Array.isArray(parsed.documents)
					? parsed.documents
					: [];
			requests.push({ query, documents });
			const results = documents
				.map((document, index) => ({
					document: { text: typeof document === "string" ? document : "" },
					index,
					relevance_score: (index + 1) / documents.length,
				}))
				.reverse();
			response.writeHead(200, { "content-type": "application/json" });
			response.end(JSON.stringify({ model: "test-rerank", results }));
		});
	});
	server.listen(0, "127.0.0.1");
	await once(server, "listening");
	const address = server.address();
	const port = typeof address === "object" && address !== null ? address.port : 0;
	return {
		requests,
		rerank: {
			provider: "openrouter",
			model: "test-rerank",
			apiKey: "test-key",
			baseUrl: `http://127.0.0.1:${port}/`,
		},
		close: () => {
			server.closeAllConnections();
			server.close();
		},
	};
}

/** MinSync stand-in that records every query and the peak number of retrievals in flight at once. */
function recordingMinSync() {
	const queries: string[] = [];
	const stats = { inFlight: 0, maxInFlight: 0 };
	return {
		queries,
		stats,
		method: {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> => {
				queries.push(query);
				stats.inFlight += 1;
				stats.maxInFlight = Math.max(stats.maxInFlight, stats.inFlight);
				// Yield so sibling retrievals started in the same tick overlap this one.
				await Promise.resolve();
				await Promise.resolve();
				stats.inFlight -= 1;
				return [
					{
						id: `hit-${query}`,
						source: join(docs, `${query.replace(/\W+/gu, "-")}.txt`),
						content: `evidence for ${query}`,
						score: 1,
						metadata: {},
					},
				];
			},
		},
	};
}

function agentWith(options: Partial<AutoRAGAgentOptions> & Pick<AutoRAGAgentOptions, "model">): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: false,
		jikji: false,
		...options,
	});
}

/** Replaces the agent's private MinSync method; the field is the established test seam. */
function injectMinSync(agent: AutoRAGAgent, method: unknown): void {
	const internals = agent as unknown as { minSyncMethod: unknown };
	internals.minSyncMethod = method;
}

async function collect(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

describe("Jev query pipeline before the fast answer", () => {
	it("answers a direct question immediately without any retrieval or verification phase", async () => {
		const prompts: string[] = [];
		const model = fauxModel(
			capture(fastAnswer("Paris is the capital of France."), prompts),
			fauxAssistantMessage("done", { stopReason: "stop" }),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("direct", 0.9) } });
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const events = await collect(agent, "What is the capital of France?");

		expect(minSync.queries).toEqual([]);
		expect(prompts).toHaveLength(1);
		expect(prompts[0]).toContain("What is the capital of France?");
		expect(prompts[0]).not.toContain("Baseline retrieval evidence");
		expect(events.some((event) => event.type === "preliminary")).toBe(false);
		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("expected a complete event");
		expect(complete.response.answer).toBe("Paris is the capital of France.");
		expect(complete.response.results).toEqual([]);
		expect(complete.response.diagnostics?.some((diagnostic) => diagnostic.code === "missing-final-emit")).toBe(false);
	});

	it("decomposes a local question with the configured model and searches every sub-query in parallel", async () => {
		const prompts: string[] = [];
		const decompositionPrompts: string[] = [];
		const decompositionModel = fauxModel(
			capture(
				fauxAssistantMessage('{"queries": ["Q3 budget approver", "Q4 budget approver"]}', { stopReason: "stop" }),
				decompositionPrompts,
			),
		);
		const source = join(docs, "Q3-budget-approver.txt");
		const model = fauxModel(
			capture(fastAnswer("Fast: Q3 and Q4 approvers found.", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			capture(finalEmit("Verified approvers.", source), prompts),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("local", 0.9) },
			queryDecomposition: { model: decompositionModel },
		});
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const events = await collect(agent, "Who approved the Q3 and Q4 budgets?");

		expect(decompositionPrompts).toHaveLength(1);
		expect(decompositionPrompts[0]).toContain("Who approved the Q3 and Q4 budgets?");
		expect([...minSync.queries].sort()).toEqual(["Q3 budget approver", "Q4 budget approver"]);
		expect(minSync.stats.maxInFlight).toBe(2);
		const fastPrompt = prompts[0] ?? "";
		expect(fastPrompt).toContain("Baseline retrieval evidence");
		expect(fastPrompt).toContain("evidence for Q3 budget approver");
		expect(fastPrompt).toContain("evidence for Q4 budget approver");
		expect(fastPrompt).toContain("Q3 budget approver");
		expect(events.some((event) => event.type === "preliminary")).toBe(true);
		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toBe("Verified approvers.");
	});

	it("reranks the merged pool of every decomposed local query against the original question", async () => {
		const rerankServer = await startRerankServer();
		try {
			const prompts: string[] = [];
			const decompositionModel = fauxModel(
				fauxAssistantMessage('{"queries": ["Q3 budget approver", "Q4 budget approver"]}', { stopReason: "stop" }),
			);
			const source = join(docs, "Q3-budget-approver.txt");
			const model = fauxModel(
				capture(fastAnswer("Fast: approvers found.", source), prompts),
				fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
				finalEmit("Verified approvers.", source),
			);
			const agent = agentWith({
				model,
				jev: { backend: jevRouting("local", 0.9) },
				queryDecomposition: { model: decompositionModel },
				rerank: rerankServer.rerank,
			});
			const minSync = recordingMinSync();
			injectMinSync(agent, minSync.method);

			await agent.searchDocuments("Who approved the Q3 and Q4 budgets?");

			expect(rerankServer.requests).toHaveLength(1);
			expect(rerankServer.requests[0]?.query).toBe("Who approved the Q3 and Q4 budgets?");
			const pool = JSON.stringify(rerankServer.requests[0]?.documents);
			expect(pool).toContain("evidence for Q3 budget approver");
			expect(pool).toContain("evidence for Q4 budget approver");
			const fastPrompt = prompts[0] ?? "";
			expect(fastPrompt).toContain("Reranked initial candidates");
			expect(fastPrompt).toContain("Q3 budget approver");
			expect(fastPrompt.indexOf("evidence for Q4 budget approver")).toBeLessThan(
				fastPrompt.indexOf("evidence for Q3 budget approver"),
			);
		} finally {
			rerankServer.close();
		}
	});

	it("searches only the datasources Jev judges needed and adds their chunks to the fast-answer evidence", async () => {
		const prompts: string[] = [];
		const source = "/slack/default/release-date";
		const model = fauxModel(
			capture(fastAnswer("Fast: the release moved to Friday [1].", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified: Friday.", source),
		);
		const slack = recordingDatasource("slack", "Company Slack: engineering and release channels");
		const discord = recordingDatasource("discord", "Gaming community Discord server");
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("local", 0.1, 0.9, { slack: 0.9, discord: 0.1 }) },
			datasourceSkills: [slack.skill, discord.skill],
			datasourceAccess: { allowedTags: ["slack", "discord"] },
		});
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const response = await agent.searchDocuments("when is the release?");

		expect(slack.queries).toEqual(["when is the release?"]);
		expect(discord.queries).toEqual([]);
		expect(minSync.queries).toEqual(["when is the release?"]);
		const fastPrompt = prompts[0] ?? "";
		expect(fastPrompt).toContain("slack evidence for when is the release?");
		// The MinSync chunk is still there alongside the datasource chunk.
		expect(fastPrompt).toContain(join(docs, "when-is-the-release-.txt"));
		expect(fastPrompt).not.toContain("discord evidence");
		expect(
			response.diagnostics?.some(
				(diagnostic) =>
					diagnostic.code === "datasources-selected" &&
					diagnostic.message.includes("slack") &&
					diagnostic.message.includes("0.90"),
			),
		).toBe(true);
	});

	it("searches a selected datasource with every decomposed query and reranks its chunks in the same pool", async () => {
		const rerankServer = await startRerankServer();
		try {
			const prompts: string[] = [];
			const decompositionModel = fauxModel(
				fauxAssistantMessage('{"queries": ["release date", "release owner"]}', { stopReason: "stop" }),
			);
			const source = "/slack/default/release-date";
			const model = fauxModel(
				capture(fastAnswer("Fast release answer [1].", source), prompts),
				fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
				finalEmit("Verified release answer.", source),
			);
			const slack = recordingDatasource("slack", "Company Slack: engineering and release channels");
			const agent = agentWith({
				model,
				jev: { backend: jevRouting("local", 0.9, 0.9, { slack: 0.7 }) },
				queryDecomposition: { model: decompositionModel },
				rerank: rerankServer.rerank,
				datasourceSkills: [slack.skill],
				datasourceAccess: { allowedTags: ["slack"] },
			});
			const minSync = recordingMinSync();
			injectMinSync(agent, minSync.method);

			await agent.searchDocuments("When is the release and who owns it?");

			expect([...slack.queries].sort()).toEqual(["release date", "release owner"]);
			expect([...minSync.queries].sort()).toEqual(["release date", "release owner"]);
			expect(rerankServer.requests).toHaveLength(1);
			expect(rerankServer.requests[0]?.query).toBe("When is the release and who owns it?");
			const documents = [...(rerankServer.requests[0]?.documents ?? [])].sort();
			// Datasource and MinSync chunks compete in one pool, not replace each other.
			expect(documents).toEqual(
				[
					"evidence for release date",
					"evidence for release owner",
					"slack evidence for release date",
					"slack evidence for release owner",
				].sort(),
			);
			expect(prompts[0]).toContain("slack evidence for release owner");
		} finally {
			rerankServer.close();
		}
	});

	it("does not search datasources on the web route", async () => {
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake DuckDuckGo",
			isAvailable: () => true,
			search: async ({ query }) => ({
				provider: "duckduckgo",
				sources: [{ title: query, url: "https://example.com/x", snippet: `web evidence for ${query}` }],
			}),
		});
		const source = "https://example.com/x";
		const model = fauxModel(
			fastAnswer("Fast web answer.", source),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified web answer.", source),
		);
		const slack = recordingDatasource("slack", "Company Slack: engineering and release channels");
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("web", 0.1, 0.9, { slack: 0.9 }) },
			webSearch: { order: ["duckduckgo"], exclude: SEARCH_PROVIDER_ORDER.filter((id) => id !== "duckduckgo") },
			datasourceSkills: [slack.skill],
			datasourceAccess: { allowedTags: ["slack"] },
		});

		await agent.searchDocuments("latest Node.js LTS?");

		expect(slack.queries).toEqual([]);
	});

	it("searches no datasource and records why when the datasource check fails", async () => {
		const failing: JevBackend = {
			name: "partial",
			async judge(request) {
				if (request.questions.some((question) => question.id === QUERY_ROUTE_QUESTION_ID)) {
					return {
						answers: [
							{ answer: "local", distribution: { local: 0.9, web: 0.05, direct: 0.05 } },
							{ answer: 0.1 },
						],
					};
				}
				throw new Error("datasource check exploded");
			},
		};
		const source = join(docs, "x.txt");
		const model = fauxModel(
			fastAnswer("Fast.", source),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified.", source),
		);
		const slack = recordingDatasource("slack", "Company Slack: engineering and release channels");
		const agent = agentWith({
			model,
			jev: { backend: failing },
			datasourceSkills: [slack.skill],
			datasourceAccess: { allowedTags: ["slack"] },
		});
		injectMinSync(agent, recordingMinSync().method);

		const response = await agent.searchDocuments("when is the release?");

		expect(slack.queries).toEqual([]);
		expect(
			response.diagnostics?.some(
				(diagnostic) =>
					diagnostic.code === "datasource-selection-fallback" &&
					diagnostic.message.includes("datasource check exploded"),
			),
		).toBe(true);
	});

	it("defaults the decomposition model to the session model", async () => {
		const prompts: string[] = [];
		const source = join(docs, "lease-signer.txt");
		const model = fauxModel(
			fauxAssistantMessage('["lease signer", "lease start date"]', { stopReason: "stop" }),
			capture(fastAnswer("Fast lease answer.", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified lease answer.", source),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("local", 0.8) } });
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const response = await agent.searchDocuments("Who signed the lease and when does it start?");

		expect([...minSync.queries].sort()).toEqual(["lease signer", "lease start date"]);
		expect(prompts[0]).toContain("evidence for lease start date");
		expect(response.answer).toBe("Verified lease answer.");
	});

	it("searches the original question once when Jev says decomposition is unnecessary", async () => {
		const source = join(docs, "x.txt");
		const model = fauxModel(
			fastAnswer("Fast.", source),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified.", source),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("local", 0.1) } });
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		await agent.searchDocuments("where is the signed lease?");

		expect(minSync.queries).toEqual(["where is the signed lease?"]);
	});

	it("routes an internet question to parallel web searches instead of local retrieval", async () => {
		const webQueries: string[] = [];
		const web = { inFlight: 0, maxInFlight: 0 };
		registerSearchProvider({
			id: "duckduckgo",
			label: "Fake DuckDuckGo",
			isAvailable: () => true,
			search: async ({ query }) => {
				webQueries.push(query);
				web.inFlight += 1;
				web.maxInFlight = Math.max(web.maxInFlight, web.inFlight);
				// Yield so sibling searches started in the same tick overlap this one.
				await Promise.resolve();
				await Promise.resolve();
				web.inFlight -= 1;
				return {
					provider: "duckduckgo",
					sources: [
						{
							title: `Result for ${query}`,
							url: `https://example.com/${encodeURIComponent(query)}`,
							snippet: `web evidence for ${query}`,
						},
					],
				};
			},
		});
		const prompts: string[] = [];
		const decompositionModel = fauxModel(
			fauxAssistantMessage('{"queries": ["Node.js latest LTS", "Bun latest release"]}', { stopReason: "stop" }),
		);
		const source = "https://example.com/Node.js%20latest%20LTS";
		const model = fauxModel(
			capture(fastAnswer("Fast web answer.", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			capture(finalEmit("Verified web answer.", source), prompts),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("web", 0.9) },
			queryDecomposition: { model: decompositionModel },
			webSearch: { order: ["duckduckgo"], exclude: SEARCH_PROVIDER_ORDER.filter((id) => id !== "duckduckgo") },
		});
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const response = await agent.searchDocuments("What are the latest Node.js LTS and Bun releases?");

		expect(minSync.queries).toEqual([]);
		expect([...webQueries].sort()).toEqual(["Bun latest release", "Node.js latest LTS"]);
		expect(web.maxInFlight).toBe(2);
		expect(prompts[0]).toContain("web evidence for Node.js latest LTS");
		expect(prompts[0]).toContain("web evidence for Bun latest release");
		// The verification phase must verify on the web, not in local files.
		expect(prompts[1]).toContain("web_search");
		expect(prompts[1]).toMatch(/internet/iu);
		expect(response.answer).toBe("Verified web answer.");
	});

	it("keeps today's single-query local search when Jev is unreachable", async () => {
		const down: JevBackend = {
			name: "down",
			async judge() {
				throw new Error("connection refused");
			},
		};
		const source = join(docs, "x.txt");
		const model = fauxModel(
			fastAnswer("Fast.", source),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified.", source),
		);
		const agent = agentWith({ model, jev: { backend: down } });
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const response = await agent.searchDocuments("where is the signed lease?");

		expect(minSync.queries).toEqual(["where is the signed lease?"]);
		expect(
			response.diagnostics?.some(
				(diagnostic) =>
					diagnostic.code === "query-route-fallback" && diagnostic.message.includes("connection refused"),
			),
		).toBe(true);
	});

	it("does not route when Jev is disabled", async () => {
		const source = join(docs, "x.txt");
		const model = fauxModel(
			fastAnswer("Fast.", source),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmit("Verified.", source),
		);
		const agent = agentWith({ model });
		const minSync = recordingMinSync();
		injectMinSync(agent, minSync.method);

		const response = await agent.searchDocuments("What is the capital of France?");

		expect(minSync.queries).toEqual(["What is the capital of France?"]);
		expect(response.diagnostics?.some((diagnostic) => diagnostic.code === "query-routed")).toBe(false);
	});

	it("ends after the fast answer when Jev says no correction, clarification, or further research is needed", async () => {
		const prompts: string[] = [];
		const source = join(docs, "budget.txt");
		const model = fauxModel(
			capture(fastAnswer("The Q3 budget was approved by Mina Park [1].", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("local", 0.1, 0.1) } });
		injectMinSync(agent, recordingMinSync().method);

		const events = await collect(agent, "who approved the Q3 budget?");

		// Only the fast prompt reached the model: no verification phase ran.
		expect(prompts).toHaveLength(1);
		expect(events.some((event) => event.type === "preliminary")).toBe(false);
		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("expected a complete event");
		expect(complete.response.answer).toBe("The Q3 budget was approved by Mina Park [1].");
		expect(complete.response.results.map((result) => result.source)).toEqual([source]);
		expect(complete.response.diagnostics?.some((diagnostic) => diagnostic.code === "missing-final-emit")).toBe(false);
		expect(
			complete.response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "follow-up-skipped" && diagnostic.message.includes("0.10"),
			),
		).toBe(true);
	});

	it("keeps a fast answer's results and citations when the model omits the sources mapping", async () => {
		// Live models routinely leave the optional `sources` field out; the
		// final response must still carry every result the answer cites.
		const model = fauxModel(
			fauxAssistantMessage(
				[
					fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
						answer: "The Q3 budget was approved by Mina Park [1].",
						results: [
							{
								number: 1,
								title: "Q3 budget approval",
								summary: "Mina Park approved the Q3 budget.",
								evidence: [{ excerpt: "The Q3 2026 budget was approved by Mina Park." }],
								confidence: 0.8,
							},
						],
					}),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("local", 0.1, 0.1) } });
		injectMinSync(agent, recordingMinSync().method);

		const response = await agent.searchDocuments("who approved the Q3 budget?");

		expect(response.answer).toBe("The Q3 budget was approved by Mina Park [1].");
		expect(response.results).toHaveLength(1);
		expect(response.results[0]).toMatchObject({ number: 1, title: "Q3 budget approval" });
		expect(response.results[0]?.evidence[0]?.excerpt).toBe("The Q3 2026 budget was approved by Mina Park.");
		// No source was reported, so none is invented.
		expect(response.results[0]?.source).toBeUndefined();
		expect(response.diagnostics?.some((diagnostic) => diagnostic.code === "citation-without-result")).toBe(false);
	});

	it("continues to verification when Jev says the fast answer needs follow-up", async () => {
		const prompts: string[] = [];
		const source = join(docs, "budget.txt");
		const model = fauxModel(
			capture(fastAnswer("The approver is not stated in the evidence.", source), prompts),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			capture(finalEmit("Verified: Mina Park approved the Q3 budget.", source), prompts),
		);
		const agent = agentWith({ model, jev: { backend: jevRouting("local", 0.1, 0.8) } });
		injectMinSync(agent, recordingMinSync().method);

		const events = await collect(agent, "who approved the Q3 budget?");

		expect(prompts).toHaveLength(2);
		expect(events.some((event) => event.type === "preliminary")).toBe(true);
		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toBe(
			"Verified: Mina Park approved the Q3 budget.",
		);
	});
});
