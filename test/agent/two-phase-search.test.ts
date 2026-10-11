import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
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
		return step as AssistantMessage;
	};
}

/** Records the latest user prompt the agent sent, then returns the scripted step. */
function capturePromptStep(step: FauxResponseStep, prompts: string[]): FauxResponseStep {
	return (context) => {
		const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
		const text =
			lastUser === undefined
				? ""
				: typeof lastUser.content === "string"
					? lastUser.content
					: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		prompts.push(text);
		return step as AssistantMessage;
	};
}

/** The fast phase now ends in a plain assistant message: no tool call. */
function fastAnswerCall(): FauxResponseStep {
	return fauxAssistantMessage("Fast answer: refund exceptions require director approval before payout.", {
		stopReason: "stop",
	});
}

/** The verification phase ends in a plain assistant message citing the file it opened. */
function finalAnswer(answer: string): FauxResponseStep {
	return fauxAssistantMessage(`${answer} [file:${join(docs, "refund-policy.txt")}]`, { stopReason: "stop" });
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
					metadata: { description: "KakaoTalk chat history for acct-1" },
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
		minSync: { autoInstall: false },
		jikji: false,
	};
}

async function collectEvents(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

describe("two-phase progressive answers (thinking off fast → thinking on final)", () => {
	it("yields the fast preliminary answer before the verified complete response", async () => {
		const model = fauxModel(
			true,
			fastAnswerCall(),
			finalAnswer("Final answer: refund exceptions require director approval before payout."),
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
		// The fast phase cited nothing, so it derives no results.
		expect(preliminary.response.results).toEqual([]);

		const complete = events[completeIndex];
		if (complete.type !== "complete") throw new Error("unreachable");
		expect(complete.response.answer).toContain("Final answer");
		// The model's `[file:<path>]` marker became a numbered citation backed by a result.
		expect(complete.response.answer).toContain("[1]");
		expect(complete.response.answer).not.toContain("[file:");
		expect(complete.response.results[0]?.source).toBe(join(docs, "refund-policy.txt"));
	});

	it("publishes the preliminary answer exactly once", async () => {
		const model = fauxModel(true, fastAnswerCall(), finalAnswer("Verified final answer."));
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "refund approval?");

		expect(events.filter((event) => event.type === "preliminary")).toHaveLength(1);
	});

	it("never reports an answer message as progress, only the text written before a tool call", async () => {
		const model = fauxModel(
			true,
			fastAnswerCall(),
			fauxAssistantMessage(
				[
					{ type: "text", text: "Checking the datasource for the July review." },
					fauxToolCall("search_datasource_kakao", { query: "refund approval", topK: 5 }),
				],
				{ stopReason: "toolUse" },
			),
			finalAnswer("Final answer."),
		);
		const agent = new AutoRAGAgent({ ...agentOptions(model), datasourceSkills: [makeSkill([])] });

		const events = await collectEvents(agent, "refund approval?");

		const progress = events.flatMap((event) => (event.type === "progress" ? [event.text] : []));
		expect(progress).toContain("Checking the datasource for the July review.");
		expect(progress.some((text) => text.includes("Fast answer"))).toBe(false);
		expect(progress.some((text) => text.includes("Final answer"))).toBe(false);
	});

	it("returns a degraded fallback with reason and retrieval trace when the model request fails", async () => {
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
			fauxAssistantMessage(
				[
					{ type: "text", text: "Searching the kakao datasource." },
					fauxToolCall("search_datasource_kakao", { query: "refund approval", topK: 5 }),
				],
				{ stopReason: "toolUse" },
			),
			// The faux provider reports a drained script as a failed model request.
			fauxAssistantMessage("", { stopReason: "error", errorMessage: "provider request failed" }),
		);
		const agent = new AutoRAGAgent({
			...agentOptions(model),
			datasourceSkills: [makeSkill(rows)],
		});

		const response = await agent.searchDocuments("refund approval");

		expect(response.results).toEqual([]);
		expect(
			response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "no-final-answer" && diagnostic.severity === "warning",
			),
		).toBe(true);
		expect(response.diagnostics?.some((diagnostic) => diagnostic.code === "model-request-failed")).toBe(true);
		expect(response.answer).toContain("The model request failed");
		expect(response.answer).toContain("fix the model provider error");
		expect(response.searched).toBe(1);
		expect(response.retrievalTrace).toHaveLength(1);
		const entry = response.retrievalTrace?.[0];
		expect(entry?.tool).toBe("search_datasource_kakao");
		expect(entry?.resultCount).toBe(1);
		expect(entry?.results[0]?.source).toBe("/kakao/acct-1/chunks/msg-1");
		expect(entry?.results[0]?.excerpt).toContain("Director approval");
	});

	it("takes a prose answer as the final answer instead of asking the model to emit it", async () => {
		const prompts: string[] = [];
		const model = fauxModel(
			true,
			fastAnswerCall(),
			capturePromptStep(
				fauxAssistantMessage("- Refund exceptions require director approval before payout.", {
					stopReason: "stop",
				}),
				prompts,
			),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("refund approval");

		expect(prompts).toHaveLength(1);
		expect(response.answer).toBe("- Refund exceptions require director approval before payout.");
		expect(response.diagnostics?.some((diagnostic) => diagnostic.code === "no-final-answer")).toBe(false);
	});

	it("ends the run without a preliminary event when the first reply is already the final answer", async () => {
		const model = fauxModel(true, finalAnswer("Immediate final answer."));
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "refund approval?");

		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toContain("Immediate final answer");
	});

	it("runs the fast phase with thinking off and the verification phase with thinking on by default", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			true,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(finalAnswer("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("refund approval?");

		expect(response.answer).toContain("Final answer");
		expect(reasoningLog).toEqual([undefined, "high"]);
	});

	it("honours explicit thinking-level overrides for both phases", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			true,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(finalAnswer("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent({
			...agentOptions(model),
			thinking: { fast: "low", final: "medium" },
		});

		await agent.searchDocuments("refund approval?");

		expect(reasoningLog).toEqual(["low", "medium"]);
	});

	it("clamps thinking levels to off for models without reasoning support", async () => {
		const reasoningLog: (string | undefined)[] = [];
		const model = fauxModel(
			false,
			recordStep(fastAnswerCall(), reasoningLog),
			recordStep(finalAnswer("Final answer."), reasoningLog),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		await agent.searchDocuments("refund approval?");

		expect(reasoningLog).toEqual([undefined, undefined]);
	});

	it("drops a citation id no tool returned instead of attaching a source to it", async () => {
		const model = fauxModel(
			true,
			fastAnswerCall(),
			fauxAssistantMessage("Refund exceptions need director approval [e99].", { stopReason: "stop" }),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("refund approval?");

		expect(response.answer).toBe("Refund exceptions need director approval.");
		expect(response.results).toEqual([]);
		expect(
			response.diagnostics?.find((diagnostic) => diagnostic.code === "citation-without-result")?.message,
		).toContain("e99");
	});

	it("asks for a delta-only final answer when the fast answer was already delivered", () => {
		const model = fauxModel(true, fauxAssistantMessage("noop", { stopReason: "stop" }));
		const agent = new AutoRAGAgent(agentOptions(model));

		const delta = agent.buildRefinementPrompt(
			"what approval do refund exceptions need?",
			{},
			"Fast: refund exceptions need director approval before payout [e1].",
			true,
		);
		// The model must be able to see the exact first answer it is diffing against.
		expect(delta).toContain("Fast: refund exceptions need director approval before payout [e1].");
		expect(delta).toMatch(/MUST contain only/i);
		expect(delta).toMatch(/never restate/i);

		const complete = agent.buildRefinementPrompt("what approval do refund exceptions need?", {}, undefined, false);
		expect(complete).toContain("the fast phase produced no answer");
		expect(complete).not.toMatch(/MUST contain only/i);
	});

	it("walks the verification phase through the already-delivered fast answer as a delta task", async () => {
		const prompts: string[] = [];
		const model = fauxModel(
			true,
			capturePromptStep(fastAnswerCall(), prompts),
			capturePromptStep(finalAnswer("Correction: exceptions also need finance sign-off."), prompts),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const events = await collectEvents(agent, "what approval do refund exceptions need?");

		const finalPrompt = prompts.at(-1) ?? "";
		expect(finalPrompt).toContain("Fast answer: refund exceptions require director approval before payout.");
		expect(finalPrompt).toMatch(/MUST contain only/i);
		const complete = events.find((event) => event.type === "complete");
		expect(complete?.type === "complete" && complete.response.answer).toContain("Correction:");
	});

	it("keeps a complete-answer instruction when no preliminary reaches a caller", async () => {
		const prompts: string[] = [];
		const model = fauxModel(
			true,
			capturePromptStep(fastAnswerCall(), prompts),
			capturePromptStep(finalAnswer("Complete verified answer."), prompts),
		);
		const agent = new AutoRAGAgent(agentOptions(model));

		const response = await agent.searchDocuments("what approval do refund exceptions need?");

		const finalPrompt = prompts.at(-1) ?? "";
		expect(finalPrompt).not.toMatch(/MUST contain only/i);
		expect(response.answer).toContain("Complete verified answer");
	});
});
