import { randomUUID } from "node:crypto";
import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	registerFauxProvider,
} from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { JIKJI_FIND_TOOL_NAME } from "../../src/agent/jikji-find-tool.ts";
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
let binaryPath: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-live-e2e-"));
	docs = join(root, "docs");
	binaryPath = join(root, "fake-jikji.mjs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "q3.txt"),
		[
			"Q3 refund policy update.",
			"Refund exceptions now require director approval before payout.",
			"Customer support should cite the July handbook addendum.",
		].join("\n"),
	);
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function explorerAssignment(originalQuery: string): string {
	return [
		`Original query: ${originalQuery}`,
		"Selected retrieval method: POSIX",
		`Query variants: ${originalQuery}; ${originalQuery} evidence`,
		"Required handoff fields: Retrieved at and Temporal metadata.",
	].join("\n");
}

function fauxModel(originalQuery: string, ...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall("subagent", {
					agent: "autorag-explorer",
					agentScope: "user",
					model: "faux/gpt-5.6-luna",
					task: explorerAssignment(originalQuery),
					cwd: docs,
					artifacts: false,
				}),
			],
			{ stopReason: "toolUse" },
		),
		...responses,
	]);
	registrations.push(reg);
	return reg.getModel();
}

function fauxSessionFactory(): NonNullable<ConstructorParameters<typeof AutoRAGAgent>[0]["sessionFactory"]> {
	return async (options) => {
		const subagentTool: AgentTool = {
			name: "subagent",
			label: "Subagent",
			description: "Test explorer",
			parameters: Type.Object({
				agent: Type.String(),
				agentScope: Type.Literal("user"),
				model: Type.String(),
				task: Type.String(),
				cwd: Type.Optional(Type.String()),
				artifacts: Type.Optional(Type.Boolean()),
			}),
			execute: async () => ({
				content: [
					{
						type: "text",
						text: "Explorer run completed successfully.",
					},
				],
				details: {
					mode: "single",
					runId: "run-live",
					results: [
						{
							agent: "autorag-explorer",
							task: explorerAssignment("What is the refund approval policy and was it acknowledged?"),
							exitCode: 0,
							usage: { input: 120, output: 90, cacheRead: 0, cacheWrite: 0, cost: 0, turns: 1 },
							model: "faux/gpt-5.6-luna",
							structuredOutput: {
								assignment: {
									originalQuery: "What is the refund approval policy and was it acknowledged?",
									method: "posix",
									queryVariant: "refund director approval",
									queryVariants: ["refund director approval", "refund exception approval"],
								},
								evidenceCandidates: [
									{
										source: join(docs, "q3.txt"),
										method: "posix",
										evidence: "Refund exceptions now require director approval before payout.",
										retrievedAt: "2026-07-14T05:30:00.000Z",
										sourceTemporal: { status: "updatedAt", updatedAt: "2026-07-01T00:00:00.000Z" },
										locator: "q3.txt:2",
									},
								],
								summary: "The Q3 policy note requires director approval for refund exceptions.",
							},
						},
					],
				},
			}),
		};
		const agent = new Agent({
			initialState: {
				systemPrompt: options.systemPrompt,
				model: options.model,
				tools: [subagentTool, ...options.tools],
			},
			convertToLlm: (messages) =>
				messages.filter(
					(message) => message.role === "user" || message.role === "assistant" || message.role === "toolResult",
				),
		});
		return {
			agent,
			prompt: async (prompt) => agent.prompt(prompt),
			abort: async () => agent.abort(),
			dispose: () => {},
		};
	};
}

function emitResults(): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer:
					"[1] Refund exceptions require director approval. [2] KakaoTalk confirms finance acknowledged the policy.",
				results: [
					{
						number: 1,
						title: "Refund approval rule",
						summary: "Refund exceptions now require director approval before payout.",
						evidence: [
							{ excerpt: "Refund exceptions now require director approval before payout.", lineNumber: 2 },
						],
						confidence: 0.95,
					},
					{
						number: 2,
						title: "KakaoTalk finance acknowledgement",
						summary: "Finance acknowledged the director-approval refund policy in KakaoTalk.",
						evidence: [{ excerpt: "Finance acknowledged director approval for refunds." }],
						confidence: 0.9,
					},
				],
				mapping: [
					{
						number: 1,
						source: "/docs/q3.txt",
						method: "search_all_documents",
						content: "Refund exceptions now require director approval before payout.",
						evidenceRefs: [
							{ method: "grep", source: "/docs/q3.txt", excerpt: "director approval" },
							{
								method: "posix",
								source: "/docs/q3.txt",
								content: "Refund exceptions now require director approval",
							},
							{ method: "bm25", source: "/docs/q3.txt", content: "refund director approval" },
						],
					},
					{
						number: 2,
						source: "/kakao/acct-1/chunks/refund-policy",
						method: "search_all_documents",
						content: "Finance acknowledged director approval for refunds.",
						evidenceRefs: [
							{
								method: "kakao.keyword",
								source: "/kakao/acct-1/chunks/refund-policy",
								content: "Finance acknowledged director approval for refunds.",
							},
						],
					},
				],
				warnings: ["MinSync semantic search unavailable; fallback retrieval paths were used."],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

class StaticDatasourceMethod implements RetrievalMethod {
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

function kakaoSkill(rows: readonly RetrievalResult[]): DatasourceSkill {
	const method = new StaticDatasourceMethod(rows);
	return {
		describe() {
			return {
				name: "kakao",
				type: "chat",
				description: "KakaoTalk chats exported through katok",
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
				content:
					"# KakaoTalk\nSearch with search_all_documents or search_datasource_documents; scope /kakao/acct-1.",
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

function writeFakeJikji(): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "find") {
	console.log(JSON.stringify({
		answer_paths: ["q3.txt"],
		paths: ["q3.txt"],
		candidates: [{ path: "q3.txt", next_read: "original", label: "Q3 refund policy" }],
		evidence_pack: [{ path: "q3.txt", next_read: "original" }],
		handoff_action: "raw_fallback_after_retry",
		tool_call_policy: { stop_after_find: false, forbidden_tools: [], allowed_followups: [] },
		agent_should_not_rerank: false,
	}));
} else {
	console.log(JSON.stringify({ prepared: true }));
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent.state.tools.map(
		(tool) => tool.name,
	);
}

describe("AutoRAGAgent live searchDocuments orchestration e2e", () => {
	it("uses mandatory built-ins, retrieval tools, datasource fan-out, Jikji find, and verbatim real paths", async () => {
		writeFakeJikji();
		const datasourceRows: RetrievalResult[] = [
			{
				id: "kakao-1",
				source: "/kakao/acct-1/chunks/refund-policy",
				content: "Finance acknowledged director approval for refunds.",
				score: 0.97,
				metadata: { method: "kakao.keyword", datasourceId: "kakao" },
			},
		];
		const model = fauxModel(
			"What is the refund approval policy and was it acknowledged?",
			fauxAssistantMessage(
				[
					fauxToolCall(JIKJI_FIND_TOOL_NAME, { query: "refund director approval" }),
					fauxToolCall("bash", { command: "grep -rn 'director approval' docs/q3.txt" }),
					fauxToolCall("search_minsync_documents", { query: "refund exception semantics", topK: 2 }),
					fauxToolCall("search_bm25_documents", { query: "refund director approval", topK: 3 }),
					fauxToolCall("search_all_documents", { query: "refund director approval finance kakao", topK: 6 }),
				],
				{ stopReason: "toolUse" },
			),
			emitResults(),
		);
		const agent = new AutoRAGAgent({
			model,
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
			bm25: { indexPath: join(root, ".autorag", "bm25-live"), forceEngine: "typescript-fallback" },
			jikji: { binaryPath },
			datasourceSkills: [kakaoSkill(datasourceRows)],
			datasourceAccess: { allowedTags: ["kakao"], allowedScopes: ["/kakao/acct-1/**"] },
			sessionFactory: fauxSessionFactory(),
		});

		for (const name of [
			"bash",
			"check_memory",
			"search_minsync_documents",
			"search_bm25_documents",
			"search_all_documents",
			"search_datasource_documents",
			"jikji_find",
			"emit_autorag_results",
		]) {
			expect(toolNames(agent)).toContain(name);
		}

		const refresh = await agent.refresh(true);
		expect(["ready", "degraded_fallback"]).toContain(refresh.bm25?.readiness);
		expect(refresh.datasources?.[0]).toMatchObject({ ok: true, skill: "kakao" });
		expect(agent.getSystemPrompt()).toContain("## Jikji Local Discovery");
		expect(agent.getSystemPrompt()).toContain("jikji_find");
		expect(agent.getSystemPrompt()).not.toContain(root);

		const all = await agent.searchAllDocuments("refund director approval finance kakao", { topK: 8 });
		expect(all.results.some((result) => result.source === "/docs/q3.txt")).toBe(true);
		expect(all.results.some((result) => result.source === "/kakao/acct-1/chunks/refund-policy")).toBe(true);
		expect(all.diagnostics.some((diagnostic) => diagnostic.code === "minsync-unavailable")).toBe(true);

		const response = await agent.searchDocuments("What is the refund approval policy and was it acknowledged?", {
			topK: 2,
		});
		const serialized = JSON.stringify(response);
		expect(response.results).toHaveLength(2);
		expect(response.answer).toContain("[1]");
		expect(response.answer).toContain("[2]");
		expect((response.diagnostics ?? []).some((diagnostic) => diagnostic.message.includes("MinSync"))).toBe(true);
		expect(serialized).toContain("Refund exceptions now require director approval");
		// path opacity is gone: the curated source path is retained verbatim in
		// the internal registry (below); the public response is not scrubbed.

		const registry = agent.getResultRegistry(response.sessionId);
		expect(registry.get(1)?.source).toBe("/docs/q3.txt");
		expect(registry.get(2)?.source).toBe("/kakao/acct-1/chunks/refund-policy");
	});
});
