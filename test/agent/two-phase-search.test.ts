import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { EMIT_FAST_ANSWER_TOOL_NAME } from "../../src/agent/fast-answer-tool.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import type {
	DatasourceIndexResult,
	DatasourceSkill,
	PollingMetadata,
	SourceDescription,
} from "../../src/datasource/types.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../src/retrieval/types.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-two-phase-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "refund-policy.txt"),
		[
			"Refund exceptions require director approval before payout.",
			"Finance acknowledged the policy in the July review.",
		].join("\n"),
	);
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(reasoning: boolean, ...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({
		api: `faux-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning }],
	});
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

function recordStep(step: FauxResponseStep, log: (string | undefined)[]): FauxResponseStep {
	return (_context, options) => {
		log.push(options?.reasoning);
		return step as ReturnType<typeof fauxAssistantMessage>;
	};
}

function fastAnswerCall(): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
				answer: "Fast answer: refund exceptions require director approval before payout.",
				results: [
					{
						number: 1,
						title: "Refund approval rule",
						summary: "Refund exceptions now require director approval before payout.",
						evidence: [{ excerpt: "Refund exceptions require director approval before payout.", lineNumber: 1 }],
						confidence: 0.6,
					},
				],
				sources: [{ number: 1, source: join(docs, "refund-policy.txt") }],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function finalEmitCall(answer: string): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer,
				results: [
					{
						number: 1,
						title: "Verified refund approval rule",
						summary: "Verified: refund exceptions require director approval before payout.",
						evidence: [{ excerpt: "Refund exceptions require director approval before payout.", lineNumber: 1 }],
						confidence: 0.95,
					},
				],
				mapping: [
					{
						number: 1,
						source: join(docs, "refund-policy.txt"),
						method: "bash",
						content: "Refund exceptions require director approval before payout.",
					},
				],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

class StaticMethod implements RetrievalMethod {
	private readonly rows: readonly RetrievalResult[];

	constructor(rows: readonly RetrievalResult[]) {
		this.rows = rows;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "kakao.keyword",
			type: "bm25",
			description: "KakaoTalk test datasource method",
			status: "active",
			capabilities: ["keyword"],
			datasourceId: "kakao",
			tags: ["kakao", "chat"],
		};
	}

	async retrieve(_query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return this.rows.slice(0, options.topK ?? this.rows.length);
	}
}

function makeSkill(rows: readonly RetrievalResult[]): DatasourceSkill {
	const method = new StaticMethod(rows);
	return {
		describe() {
			return {
				name: "kakao",
				type: "chat",
				description: "KakaoTalk chats exported through lazykatok",
				capabilities: ["keyword", "polling"],
				tags: ["kakao", "chat"],
				status: "active",
				datasourceId: "kakao",
				instanceId: "acct-1",
				instances: ["acct-1"],
			};
		},
		polling(): PollingMetadata {
			return { mode: "poll", intervalMs: 60_000 };
		},
		skillManifest() {
			return {
				name: "datasource-kakao",
				description: "Search indexed KakaoTalk chats.",
				content: "# KakaoTalk\nSearch with search_datasource_kakao; scope /kakao/acct-1.",
			};
		},
		async index(): Promise<DatasourceIndexResult> {
			return {
				ok: true,
				instanceId: "acct-1",
				skill: "kakao",
				chunkCount: rows.length,
				indexedAt: 1,
				diagnostics: [],
			};
		},
		retrievalMethods() {
			return [method];
		},
		describeSources(): readonly SourceDescription[] {
			return [
				{
					source: "/kakao/acct-1",
					datasourceId: "kakao",
					skill: "kakao",
					instanceId: "acct-1",
					contentType: "chat",
					metadata: { description: "authorized KakaoTalk chat history" },
				},
			];
		},
	};
}

function agentOptions(model: ReturnType<typeof fauxModel>): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: false,
		jikji: false,
	};
}

async function collectEvents(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as { tools: readonly AgentTool[] }).tools.map((tool) => tool.name);
}

describe("two-phase progressive answers (thinking off fast → thinking on final)", () => {
	it("yields the fast preliminary answer before the verified complete response", async () => {
		const model = fauxModel(
			true,
			fastAnswerCall(),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			finalEmitCall("Final answer: refund exceptions require director approval before payout."),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "what approval do refund exceptions need?");

		const types = events.map((event) => event.type);
		const preliminaryIndex = types.indexOf("preliminary");
		const completeIndex = types.indexOf("complete");
		expect(preliminaryIndex).toBeGreaterThanOrEqual(0);
		expect(completeIndex).toBeGreaterThan(preliminaryIndex);

		const preliminary = events[preliminaryIndex];
		if (preliminary.type !== "preliminary") throw new Error("unreachable");
		expect(preliminary.response.answer).toContain("Fast answer");
		expect(preliminary.response.answer).toContain("director approval");
		expect(preliminary.response.results).toHaveLength(1);
		expect(preliminary.response.results[0]?.source).toBe(join(docs, "refund-policy.txt"));

		const complete = events[completeIndex];
		if (complete.type !== "complete") throw new Error("unreachable");
		expect(complete.response.answer).toContain("Final answer");
	});

	it("delivers the preliminary event before the fast phase prompt continues", async () => {
		// Given a provider that pauses after the fast-answer tool has executed.
		let markContinuationReady: (() => void) | undefined;
		const continuationReady = new Promise<void>((resolve) => {
			markContinuationReady = resolve;
		});
		let releaseContinuation: ((message: ReturnType<typeof fauxAssistantMessage>) => void) | undefined;
		const continuation: FauxResponseStep = () => {
			const result = new Promise<ReturnType<typeof fauxAssistantMessage>>((resolve) => {
				releaseContinuation = resolve;
			});
			markContinuationReady?.();
			return result;
		};
		const model = fauxModel(true, fastAnswerCall(), continuation, finalEmitCall("Verified final answer."));
		const agent = new AutoRAGAgent(agentOptions(model));
		const events: SearchDocumentsStreamEvent[] = [];
		const collecting = (async () => {
			for await (const event of agent.searchDocumentsStream("refund approval?")) events.push(event);
		})();

		// When the post-tool model turn is still blocked, the answer must be visible.
		try {
			await continuationReady;
			expect(events.some((event) => event.type === "preliminary")).toBe(true);
		} finally {
			releaseContinuation?.(fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }));
			await collecting;
		}
		// Then finishing the prompt must not publish the same answer twice.
		expect(events.filter((event) => event.type === "preliminary")).toHaveLength(1);
	});

	it("returns a degraded fallback with reason and retrieval trace when the final emit never happens", async () => {
		const rows: RetrievalResult[] = [
			{
				id: "msg-1",
				source: "/kakao/acct-1/chunks/msg-1",
				content: "Director approval is required before payout.",
				score: 1,
				metadata: {},
			},
		];
		const model = fauxModel(
			true,
			fastAnswerCall(),
			fauxAssistantMessage("Fast answer delivered.", { stopReason: "stop" }),
			fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "refund approval", topK: 5 })], {
				stopReason: "toolUse",
			}),
			fauxAssistantMessage(
				"I searched the configured datasources but could not find enough evidence to finalize an answer.",
				{ stopReason: "stop" },
			),
		);
		const agent = new AutoRAGAgent({
			...agentOptions(model),
			datasourceSkills: [makeSkill(rows)],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/**"] },
		});

		const response = await agent.searchDocuments("refund approval");

		expect(response.results).toEqual([]);
		expect(
			response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "missing-final-emit" && diagnostic.severity === "warning",
			),
		).toBe(true);
		expect(response.answer).toContain("without calling emit_autorag_results");
		expect(response.answer).toContain("could not find enough evidence");
		expect(response.answer).toContain("searchPaths");
		expect(response.searched).toBe(1);
		expect(response.retrievalTrace).toHaveLength(1);
		const entry = response.retrievalTrace?.[0];
		expect(entry?.tool).toBe("search_datasource_kakao");
		expect(entry?.resultCount).toBe(1);
		expect(entry?.results[0]?.source).toBe("/kakao/acct-1/chunks/msg-1");
		expect(entry?.results[0]?.excerpt).toContain("Director approval");
	});

	it("ends the run without a preliminary event when the model emits final results immediately", async () => {
		const model = fauxModel(true, finalEmitCall("Immediate final answer."));
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "refund approval?");

		expect(events.some((event) => event.type === "preliminary")).toBe(false);
		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toContain("Immediate final answer");
	});

	it("runs the fast phase with thinking off and the verification phase with thinking on by default", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			true,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(fauxAssistantMessage("done", { stopReason: "stop" }), reasoningLog),
			recordStep(finalEmitCall("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("refund approval?");

		expect(response.answer).toContain("Final answer");
		expect(reasoningLog).toEqual([undefined, undefined, "high"]);
	});

	it("honours explicit thinking-level overrides for both phases", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			true,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(fauxAssistantMessage("done", { stopReason: "stop" }), reasoningLog),
			recordStep(finalEmitCall("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent({
			...agentOptions(model),
			thinking: { fast: "low", final: "medium" },
		});

		await agent.searchDocuments("refund approval?");

		expect(reasoningLog).toEqual(["low", "low", "medium"]);
	});

	it("clamps thinking levels to off for models without reasoning support", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			false,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(fauxAssistantMessage("done", { stopReason: "stop" }), reasoningLog),
			recordStep(finalEmitCall("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		await agent.searchDocuments("refund approval?");

		expect(reasoningLog).toEqual([undefined, undefined, undefined]);
	});

	it("keeps the legacy single-phase flow when thinking is disabled", async () => {
		const model = fauxModel(
			true,
			fauxAssistantMessage(
				[fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, { answer: "should not surface", results: [] })],
				{
					stopReason: "toolUse",
				},
			),
			finalEmitCall("Legacy final answer."),
		);
		const agent = new AutoRAGAgent({ ...agentOptions(model), thinking: false });
		expect(toolNames(agent)).not.toContain(EMIT_FAST_ANSWER_TOOL_NAME);

		const events = await collectEvents(agent, "refund approval?");

		expect(events.some((event) => event.type === "preliminary")).toBe(false);
		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toContain("Legacy final answer");
	});

	it("still yields a preliminary answer when the fast phase responds with text only", async () => {
		const model = fauxModel(
			true,
			fauxAssistantMessage("Quick take: refund exceptions need director approval.", { stopReason: "stop" }),
			finalEmitCall("Final answer."),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "refund approval?");

		const preliminary = events.find((event) => event.type === "preliminary");
		expect(preliminary?.type === "preliminary" && preliminary.response.answer).toContain("Quick take");
		expect(preliminary?.type === "preliminary" && preliminary.response.results).toEqual([]);
	});
});
