import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
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
import {
	DECOMPOSE_QUESTION_ID,
	datasourceQuestionId,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";
import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

// `searchDocuments()` / `searchDocumentsStream()` callers (the desktop app, embedding
// apps) steer a running search through `AutoRAGAgent.steer()` and stop it with
// `abort()`. The TUI routes typed input through the same machinery; these specs
// pin the programmatic surface.

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-search-steering-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	registration.setResponses(responses);
	registrations.push(registration);
	return { model: registration.getModel(), registration };
}

function textOf(content: string | readonly { type: string; text?: string }[]): string {
	return typeof content === "string"
		? content
		: content.map((block) => (block.type === "text" ? (block.text ?? "") : "")).join("");
}

function lastUserText(context: Context): string {
	const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
	return lastUser === undefined ? "" : textOf(lastUser.content);
}

/** Records every user message of the request the model received, then answers. */
function seeingStep(answer: string, seen: string[][]): FauxResponseStep {
	return (context) => {
		seen.push(context.messages.flatMap((message) => (message.role === "user" ? [textOf(message.content)] : [])));
		return fauxAssistantMessage(answer, { stopReason: "stop" });
	};
}

/** A faux model step that stays in flight until `release` is called. */
function heldStep(step: FauxResponseStep) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	const held: FauxResponseStep = async (...args) => {
		markEntered();
		await gate;
		return typeof step === "function" ? step(...args) : step;
	};
	return { held, entered, release: () => release() };
}

/** A datasource whose retrieval stays in flight until released or aborted through its signal. */
function slowDatasource(datasourceId: string) {
	let markEntered = (): void => {};
	let release = (): void => {};
	const entered = new Promise<void>((resolve) => {
		markEntered = resolve;
	});
	const gate = new Promise<void>((resolve) => {
		release = resolve;
	});
	let aborted = false;
	const method: RetrievalMethod = {
		describe: () => ({
			name: `${datasourceId}.keyword`,
			type: "bm25",
			description: `${datasourceId} test method`,
			status: "active",
			capabilities: ["keyword"],
			datasourceId,
		}),
		async retrieve(_query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
			markEntered();
			const signal = options.signal;
			await Promise.race([
				gate,
				new Promise<void>((resolve) => {
					signal?.addEventListener("abort", () => resolve(), { once: true });
				}),
			]);
			if (signal?.aborted) aborted = true;
			return [];
		},
	};
	const skill: DatasourceSkill = {
		describe: () => ({
			name: datasourceId,
			type: "chat",
			description: `${datasourceId} test datasource`,
			capabilities: ["keyword"],
			status: "active",
			datasourceId,
			tags: [datasourceId],
		}),
		polling: () => ({ mode: "poll", intervalMs: 60_000 }),
		skillManifest: () => ({
			name: `datasource-${datasourceId}`,
			description: `Search ${datasourceId}.`,
			content: `# ${datasourceId}`,
		}),
		index: async () => ({
			ok: true,
			instanceId: "default",
			skill: datasourceId,
			chunkCount: 0,
			indexedAt: 1,
			diagnostics: [],
		}),
		retrievalMethods: () => [method],
		describeSources: () => [],
	};
	return { skill, entered, aborted: () => aborted, release: () => release() };
}

/** Jev on the local route; `followUp` is P(verify) after the fast answer; `datasources` are searched before it. */
function jevLocal(followUp: number, datasources: readonly string[] = []): JevBackend {
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: {
			answer: "local",
			distribution: { local: 0.8, web: 0.1, direct: 0.1 },
			confidence: 0.8,
		},
		[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
		[FOLLOW_UP_QUESTION_ID]: { answer: followUp },
		...Object.fromEntries(datasources.map((id) => [datasourceQuestionId(id), { answer: 0.9 }])),
	});
}

function agentWith(options: Partial<AutoRAGAgentOptions> & Pick<AutoRAGAgentOptions, "model">): AutoRAGAgent {
	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
		memoryEmbedder: false,
		persistPiSessions: false,
		...options,
	});
	// The private MinSync field is the established test seam (see jev-pipeline.test.ts):
	// a MinSync that is installed and returns nothing.
	const internals = agent as unknown as { minSyncMethod: unknown };
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (): Promise<RetrievalResult[]> => [],
	};
	return agent;
}

async function collect(stream: AsyncGenerator<SearchDocumentsStreamEvent>): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

function answers(events: readonly SearchDocumentsStreamEvent[]) {
	return {
		quick: events.flatMap((event) => (event.type === "preliminary" ? [event.response.answer] : [])),
		final: events.flatMap((event) => (event.type === "complete" ? [event.response.answer] : [])),
	};
}

describe("steering a searchDocuments() run", () => {
	it("delivers a steer sent while the fast phase streams to its next model request, in the same search", async () => {
		const seen: string[][] = [];
		const fast = heldStep(
			fauxAssistantMessage([fauxToolCall("semantic_search_local_docs", { query: "capital of france" })], {
				stopReason: "toolUse",
			}),
		);
		const { model, registration } = fauxModel(
			fast.held,
			seeingStep("Fast answer: Paris, and Lyon is not the capital.", seen),
			seeingStep("Verified: Paris, not Lyon.", seen),
		);
		const agent = agentWith({ model });

		const run = collect(agent.searchDocumentsStream("What is the capital of France?"));
		await fast.entered;
		expect(agent.steer("Check Lyon too.")).toBe(true);
		fast.release();
		const events = await run;

		// The fast phase's very next request carried the steer, framed as a correction.
		expect(seen[0]?.at(-1)).toContain("Check Lyon too.");
		expect(seen[0]?.at(-1)).toContain("The user interrupted the running search with this correction");
		// Verification restates it; no second search ran.
		expect(seen[1]?.at(-1)).toContain("Check Lyon too.");
		expect(registration.state.callCount).toBe(3);
		expect(answers(events)).toEqual({
			quick: ["Fast answer: Paris, and Lyon is not the capital."],
			final: ["Verified: Paris, not Lyon."],
		});
	});

	it("cuts a slow baseline prefetch short when steered before the fast answer, so the steer reaches the fast phase", async () => {
		const datasource = slowDatasource("kakao");
		const seen: string[][] = [];
		const { model } = fauxModel(seeingStep("Fast answer.", seen), seeingStep("Verified.", seen));
		const agent = agentWith({
			model,
			jev: { backend: jevLocal(0.9, ["kakao"]) },
			datasourceSkills: [datasource.skill],
		});
		try {
			const run = collect(agent.searchDocumentsStream("What did Minji say about the launch?"));
			await datasource.entered;
			expect(agent.steer("Only the messages from last week.")).toBe(true);
			// No release: only the steer may end the prefetch.
			await run;

			expect(datasource.aborted()).toBe(true);
			expect(seen[0]?.join("\n")).toContain("Only the messages from last week.");
		} finally {
			datasource.release();
		}
	});

	it("cuts a slow retrieval tool short when steered after the fast answer, and the final answer follows the steer", async () => {
		const datasource = slowDatasource("kakao");
		const toolResults: string[] = [];
		const seen: string[][] = [];
		const { model, registration } = fauxModel(
			fauxAssistantMessage("Fast answer: the launch is in May.", { stopReason: "stop" }),
			fauxAssistantMessage([fauxToolCall("search_datasource_kakao", { query: "launch date" })], {
				stopReason: "toolUse",
			}),
			(context) => {
				for (const message of context.messages) {
					if (message.role === "toolResult") toolResults.push(textOf(message.content));
				}
				seen.push([lastUserText(context)]);
				return fauxAssistantMessage("Correction: per the June plan, the launch is in July.", {
					stopReason: "stop",
				});
			},
		);
		const agent = agentWith({ model, datasourceSkills: [datasource.skill] });
		try {
			const quick: string[] = [];
			const run = (async () => {
				const events: SearchDocumentsStreamEvent[] = [];
				for await (const event of agent.searchDocumentsStream("When is the launch?")) {
					if (event.type === "preliminary") quick.push(event.response.answer);
					events.push(event);
				}
				return events;
			})();
			await datasource.entered;
			expect(quick).toEqual(["Fast answer: the launch is in May."]);
			expect(agent.steer("Use the June plan, not the April one.")).toBe(true);
			// No release: only the steer may end the retrieval.
			const events = await run;

			expect(datasource.aborted()).toBe(true);
			expect(toolResults.some((text) => text.includes("Skipped due to the user's steering message"))).toBe(true);
			expect(seen[0]?.[0]).toContain("Use the June plan, not the April one.");
			expect(registration.state.callCount).toBe(3);
			expect(answers(events).final).toEqual(["Correction: per the June plan, the launch is in July."]);
		} finally {
			datasource.release();
		}
	});

	it("verifies a fast answer Jev would accept as final once the search was steered", async () => {
		const seen: string[][] = [];
		const fast = heldStep(fauxAssistantMessage("Fast answer: Lyon.", { stopReason: "stop" }));
		const { model, registration } = fauxModel(
			fast.held,
			seeingStep("Corrected: Paris.", seen),
			seeingStep("Verified: Paris.", seen),
		);
		const agent = agentWith({ model, jev: { backend: jevLocal(0.05) } });

		const run = collect(agent.searchDocumentsStream("What is the capital of France?"));
		await fast.entered;
		expect(agent.steer("Lyon is wrong.")).toBe(true);
		fast.release();
		const events = await run;

		expect(registration.state.callCount).toBe(3);
		expect(seen[1]?.at(-1)).toContain("Original query: What is the capital of France?");
		expect(seen[1]?.at(-1)).toContain("Lyon is wrong.");
		expect(answers(events).final).toEqual(["Verified: Paris."]);
	});

	it("answers a steer that lands after verification's last request with one more turn of the same search", async () => {
		const seen: string[][] = [];
		const verification = heldStep(fauxAssistantMessage("Verified: Paris.", { stopReason: "stop" }));
		const { model, registration } = fauxModel(
			fauxAssistantMessage("Fast answer: Paris.", { stopReason: "stop" }),
			verification.held,
			seeingStep("Paris, about 2.1 million people.", seen),
		);
		const agent = agentWith({ model });
		let settles = 0;
		agent.subscribe((event) => {
			// The moment verification's turn ends, before the search returns.
			if (event.type === "agent_end" && ++settles === 2) agent.steer("Also give the population.");
		});

		const run = collect(agent.searchDocumentsStream("What is the capital of France?"));
		await verification.entered;
		verification.release();
		const events = await run;

		expect(registration.state.callCount).toBe(3);
		expect(seen[0]?.join("\n")).toContain("Also give the population.");
		expect(answers(events).final).toEqual(["Paris, about 2.1 million people."]);
	});

	it("refuses a steer when no search runs, so the caller asks it as a new question", async () => {
		const { model } = fauxModel(
			fauxAssistantMessage("Fast answer.", { stopReason: "stop" }),
			fauxAssistantMessage("Verified.", { stopReason: "stop" }),
		);
		const agent = agentWith({ model });

		expect(agent.steer("Too early.")).toBe(false);
		await collect(agent.searchDocumentsStream("What is the capital of France?"));
		expect(agent.steer("Too late.")).toBe(false);
	});

	it("keeps a steer from one search out of the next search", async () => {
		const firstSeen: string[][] = [];
		const nextSeen: string[][] = [];
		const fast = heldStep(fauxAssistantMessage("Fast answer: Lyon.", { stopReason: "stop" }));
		const { model } = fauxModel(
			fast.held,
			// The steer lands after the fast turn: pi answers it within the fast phase, then verification runs.
			seeingStep("Corrected: Paris.", firstSeen),
			seeingStep("Verified: Paris.", firstSeen),
			seeingStep("Fast answer: Leonardo.", nextSeen),
			seeingStep("Verified: Leonardo da Vinci.", nextSeen),
		);
		const agent = agentWith({ model });

		const first = collect(agent.searchDocumentsStream("What is the capital of France?"));
		await fast.entered;
		agent.steer("Lyon is wrong.");
		fast.release();
		await first;
		const next = await collect(agent.searchDocumentsStream("Who painted the Mona Lisa?"));

		expect(firstSeen.flat().join("\n")).toContain("Lyon is wrong.");
		expect(nextSeen).toHaveLength(2);
		expect(nextSeen.map((messages) => messages.at(-1)).join("\n")).not.toContain("Lyon is wrong.");
		expect(answers(next).final).toEqual(["Verified: Leonardo da Vinci."]);
	});
});

describe("aborting a searchDocuments() run", () => {
	it("stops the whole search: no later phase runs after the fast phase is aborted", async () => {
		const fast = heldStep(fauxAssistantMessage("Fast answer: Paris.", { stopReason: "stop" }));
		const { model, registration } = fauxModel(
			fast.held,
			fauxAssistantMessage("Verification that must not run.", { stopReason: "stop" }),
		);
		const agent = agentWith({ model });

		const run = collect(agent.searchDocumentsStream("What is the capital of France?"));
		await fast.entered;
		agent.abort();
		fast.release();
		const events = await run;

		expect(registration.state.callCount).toBe(1);
		expect(answers(events).final.join("\n")).not.toContain("Verification that must not run.");
	});

	it("stops a slow baseline prefetch and never reaches the model", async () => {
		const datasource = slowDatasource("kakao");
		const { model, registration } = fauxModel(fauxAssistantMessage("Must not run.", { stopReason: "stop" }));
		const agent = agentWith({
			model,
			jev: { backend: jevLocal(0.9, ["kakao"]) },
			datasourceSkills: [datasource.skill],
		});
		try {
			const run = collect(agent.searchDocumentsStream("What did Minji say about the launch?"));
			await datasource.entered;
			agent.abort();
			await run;

			expect(datasource.aborted()).toBe(true);
			expect(registration.state.callCount).toBe(0);
		} finally {
			datasource.release();
		}
	});

	it("does not carry an abort into the next search", async () => {
		const { model, registration } = fauxModel(
			fauxAssistantMessage("Fast answer.", { stopReason: "stop" }),
			fauxAssistantMessage("Verified.", { stopReason: "stop" }),
		);
		const agent = agentWith({ model });

		agent.abort();
		const events = await collect(agent.searchDocumentsStream("What is the capital of France?"));

		expect(registration.state.callCount).toBe(2);
		expect(answers(events).final).toEqual(["Verified."]);
	});
});
