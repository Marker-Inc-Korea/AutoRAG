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
import {
	buildInjectedBaselineBlock,
	createFastPhaseParallelExtension,
	type FastPhaseParallelState,
} from "../../src/agent/fast-phase-parallel.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-parallel-fast-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "refund-policy.txt"), "Refund exceptions require director approval before payout.\n");
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({
		api: `faux-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning: true }],
	});
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

function agentOptions(
	model: ReturnType<typeof fauxModel>,
	extra: Partial<AutoRAGAgentOptions> = {},
): AutoRAGAgentOptions {
	return {
		searchPaths: [docs],
		workspacePath: join(root, "ws"),
		memoryPath: join(root, "memory.json"),
		model,
		jev: baselineJev(),
		jikji: false,
		everything: false,
		fsearch: false,
		dupey: false,
		filee: false,
		webSearch: false,
		rerank: false,
		persistPiSessions: false,
		...extra,
	};
}

describe("parallel grep-first + reranked-MinSync fast phase", () => {
	it("injects arrived reranked baseline into the grep tool_result and blocks subsequent fast-phase tool calls", async () => {
		const capturedToolResults: string[] = [];
		const model = fauxModel(
			// Step 1: fast phase starts with a grep call while MinSync prefetch is running
			fauxAssistantMessage([fauxToolCall("grep", { pattern: "director approval" })], { stopReason: "toolUse" }),
			// Step 2: model ignores the stop directive and tries one more grep — extension must block it
			(context) => {
				for (const msg of context.messages) {
					if (msg.role === "toolResult") {
						const txt = msg.content.map((c) => (c.type === "text" ? c.text : "")).join("");
						capturedToolResults.push(txt);
					}
				}
				return fauxAssistantMessage([fauxToolCall("grep", { pattern: "second grep" })], {
					stopReason: "toolUse",
				}) as AssistantMessage;
			},
			// Step 3: model yields fast answer after the second grep is blocked
			(context) => {
				const last = context.messages.at(-1);
				if (last?.role === "toolResult") {
					capturedToolResults.push(last.content.map((c) => (c.type === "text" ? c.text : "")).join(""));
				}
				return fauxAssistantMessage("Director approval is required [e1].", {
					stopReason: "stop",
				}) as AssistantMessage;
			},
			// Step 4: verification phase (Jev baseline defaults to verifying)
			fauxAssistantMessage("Verified: director approval is required [e1].", { stopReason: "stop" }),
		);

		const agent = new AutoRAGAgent(agentOptions(model, { parallelFastPhase: { toolCap: 5 } }));
		const internals = agent as unknown as { minSyncMethod: unknown };
		internals.minSyncMethod = {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (): Promise<RetrievalResult[]> => [
				{
					id: "1",
					source: join(docs, "refund-policy.txt"),
					content: "Refund exceptions require director approval before payout.",
					score: 0.99,
					metadata: {},
				},
			],
		};

		const res = await agent.searchDocuments("what approval do refund exceptions need?");
		expect(res.answer).toContain("director approval");
		// First tool result received the injected baseline block
		expect(capturedToolResults.some((t) => t.includes("[Baseline retrieval evidence arrived]"))).toBe(true);
		expect(capturedToolResults.some((t) => t.includes("Refund exceptions require director approval"))).toBe(true);
		// Second grep call after baseline injection was blocked
		expect(capturedToolResults.some((t) => t.includes("already been delivered"))).toBe(true);
	});

	it("enforces toolCap by waiting for a slow baseline and stopping exploration", async () => {
		let resolveSlowBaseline!: (rows: RetrievalResult[]) => void;
		const slowBaseline = new Promise<RetrievalResult[]>((resolve) => {
			resolveSlowBaseline = resolve;
		});

		const state: FastPhaseParallelState = {
			baselinePromise: slowBaseline.then(() => "[e1] slow-baseline-hit"),
			baselineReady: undefined,
			baselineSettled: false,
			baselineInjected: false,
			explorationCalls: 0,
			toolCap: 2,
		};

		const handlers = new Map<string, (event: unknown) => Promise<unknown>>();
		const fakePi = {
			on: (name: string, fn: (event: unknown) => Promise<unknown>) => {
				handlers.set(name, fn);
			},
		};
		createFastPhaseParallelExtension(() => state)(fakePi as never);

		const onToolResult = handlers.get("tool_result")!;
		const onToolCall = handlers.get("tool_call")!;

		// Call 1: baseline not ready yet, under cap -> no injection
		const res1 = await onToolResult({ toolName: "grep", content: [{ type: "text", text: "hit 1" }] });
		expect(res1).toBeUndefined();
		expect(state.explorationCalls).toBe(1);
		expect(state.baselineInjected).toBe(false);

		// Call 2: hits toolCap=2 -> awaits baselinePromise and injects
		const pendingRes2 = onToolResult({ toolName: "grep", content: [{ type: "text", text: "hit 2" }] });
		resolveSlowBaseline([]);
		const res2 = (await pendingRes2) as { content: { type: string; text: string }[] };
		expect(state.explorationCalls).toBe(2);
		expect(state.baselineInjected).toBe(true);
		expect(state.stopReason).toBe("tool-cap");
		expect(res2.content.at(-1)?.text).toContain("exploration limit reached");
		expect(res2.content.at(-1)?.text).toContain("[e1] slow-baseline-hit");

		// Call 3: blocked immediately after injection
		const block3 = (await onToolCall({ toolName: "grep" })) as { block?: boolean; reason?: string };
		expect(block3?.block).toBe(true);
		expect(block3?.reason).toContain("already been delivered");
		expect(buildInjectedBaselineBlock("x", "arrived")).toContain("[Baseline retrieval evidence arrived]");
	});

	it("preserves legacy sequential fast phase when parallelFastPhase: false", async () => {
		const seenPrompts: string[] = [];
		const model = fauxModel(
			(context) => {
				const lastUser = [...context.messages].reverse().find((m) => m.role === "user");
				const text =
					lastUser === undefined
						? ""
						: typeof lastUser.content === "string"
							? lastUser.content
							: lastUser.content.map((p) => (p.type === "text" ? p.text : "")).join("");
				seenPrompts.push(text);
				return fauxAssistantMessage("Fast answer [e1].", { stopReason: "stop" }) as AssistantMessage;
			},
			fauxAssistantMessage("Verified answer [e1].", { stopReason: "stop" }),
		);
		const agent = new AutoRAGAgent(agentOptions(model, { parallelFastPhase: false }));
		const internals = agent as unknown as { minSyncMethod: unknown };
		internals.minSyncMethod = {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (): Promise<RetrievalResult[]> => [
				{
					id: "1",
					source: join(docs, "refund-policy.txt"),
					content: "Refund exceptions require director approval.",
					score: 1,
					metadata: {},
				},
			],
		};
		await agent.searchDocuments("what approval do refund exceptions need?");
		expect(seenPrompts[0]).toContain("Baseline retrieval evidence:");
	});

	it("auto-enables parallel fast phase by default when parsed mirrors exist in the workspace", async () => {
		mkdirSync(join(root, "ws", ".autorag", "minsync", "files"), { recursive: true });
		const seenPrompts: string[] = [];
		const model = fauxModel(
			(context) => {
				const lastUser = [...context.messages].reverse().find((m) => m.role === "user");
				const text =
					lastUser === undefined
						? ""
						: typeof lastUser.content === "string"
							? lastUser.content
							: lastUser.content.map((p) => (p.type === "text" ? p.text : "")).join("");
				seenPrompts.push(text);
				return fauxAssistantMessage([fauxToolCall("grep", { pattern: "director" })], {
					stopReason: "toolUse",
				}) as AssistantMessage;
			},
			fauxAssistantMessage("Director approval is required [e1].", { stopReason: "stop" }),
			fauxAssistantMessage("Verified: director approval [e1].", { stopReason: "stop" }),
		);
		const agent = new AutoRAGAgent(agentOptions(model));
		const internals = agent as unknown as { minSyncMethod: unknown };
		internals.minSyncMethod = {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (): Promise<RetrievalResult[]> => [
				{
					id: "1",
					source: join(docs, "refund-policy.txt"),
					content: "Refund exceptions require director approval.",
					score: 1,
					metadata: {},
				},
			],
		};
		const res = await agent.searchDocuments("what approval do refund exceptions need?");
		expect(res.answer).toContain("director approval");
		expect(seenPrompts[0]).toContain("being prepared in parallel");
		expect(seenPrompts[0]).not.toContain("Baseline retrieval evidence:");
	});

	it("unblocks toolCap wait immediately when abortSignal fires", async () => {
		const controller = new AbortController();
		const neverResolves = new Promise<string>(() => {});
		const state: FastPhaseParallelState = {
			baselinePromise: neverResolves,
			baselineReady: undefined,
			baselineSettled: false,
			baselineInjected: false,
			explorationCalls: 1,
			toolCap: 1,
			abortSignal: controller.signal,
		};
		const handlers = new Map<string, (event: unknown) => Promise<unknown>>();
		createFastPhaseParallelExtension(() => state)({
			on: (name: string, fn: (event: unknown) => Promise<unknown>) => handlers.set(name, fn),
		} as never);
		const pending = handlers.get("tool_call")!({ toolName: "grep" });
		controller.abort();
		const res = (await pending) as { block?: boolean; reason?: string };
		expect(res.block).toBe(true);
		expect(state.stopReason).toBe("tool-cap");
	});
});
