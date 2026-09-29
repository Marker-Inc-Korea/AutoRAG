import assert from "node:assert/strict";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { createChatStore, type ChatStore } from "../src/main/chat-store";
import { createSearchService } from "../src/main/search-service";
import { SEARCH_CHANNELS, type ChatRecord, type SearchStreamEvent } from "../src/shared/search-contract";

interface TestResponse {
	readonly sessionId: string;
	readonly query: string;
	readonly results: readonly {
		readonly number: number;
		readonly title: string;
		readonly summary: string;
		readonly evidence: readonly { readonly excerpt: string; readonly lineNumber?: number }[];
		readonly confidence: number;
		readonly feedbackId: string;
		readonly source?: string;
	}[];
	readonly answer: string;
	readonly searched: number;
	readonly warnings: readonly "empty-query"[];
	readonly diagnostics: readonly [];
	readonly retrievalTrace: readonly [];
}

type TestStreamEvent =
	| { readonly type: "progress"; readonly sessionId: string; readonly query: string; readonly text: string }
	| { readonly type: "preliminary"; readonly response: TestResponse }
	| { readonly type: "complete"; readonly response: TestResponse };

const tempDirs: string[] = [];
let currentMs = 0;

function now(): Date {
	return new Date(currentMs);
}

async function createStore(): Promise<ChatStore> {
	const directory = await mkdtemp(join(tmpdir(), "autorag-search-service-"));
	tempDirs.push(directory);
	return createChatStore({ directory, now });
}

afterEach(async () => {
	currentMs = 0;
	await Promise.all(tempDirs.splice(0).map((directory) => rm(directory, { recursive: true, force: true })));
});

function createResponse(sessionId: string, answer: string, searched: number): TestResponse {
	return {
		sessionId,
		query: "refund approval",
		results: [
			{
				number: 1,
				title: "Refund policy",
				summary: "Director approval is required before payout.",
				evidence: [{ excerpt: "Refund exceptions require director approval before payout.", lineNumber: 3 }],
				confidence: 0.92,
				feedbackId: `${sessionId}:1`,
				source: "/Users/jeff/Documents/refund-policy.md",
			},
		],
		answer,
		searched,
		warnings: [],
		diagnostics: [],
		retrievalTrace: [],
	};
}

function createFakeAgent(stream: () => AsyncGenerator<TestStreamEvent, void, void>) {
	const feedbackCalls: { sessionId: string; useful: readonly number[]; notUseful: readonly number[] }[] = [];
	return {
		feedbackCalls,
		searchDocumentsStream: vi.fn(stream),
		recordFeedbackByNumbers: vi.fn(
			(sessionId: string, useful: readonly number[], notUseful: readonly number[] = []) => {
				feedbackCalls.push({ sessionId, useful, notUseful });
			},
		),
		abort: vi.fn(),
	};
}

function collectEvents(): { events: SearchStreamEvent[]; send: (channel: string, payload: SearchStreamEvent) => void } {
	const events: SearchStreamEvent[] = [];
	return {
		events,
		send: (channel, payload) => {
			expect(channel).toBe(SEARCH_CHANNELS.event);
			events.push(payload);
		},
	};
}

describe("createSearchService", () => {
	it("maps the preliminary stream event to a Quick phase", async () => {
		const quickResponse = createResponse("session-quick", "**Quick** answer [1]", 2);
		const agent = createFakeAgent(async function*() {
			yield { type: "progress", sessionId: "", query: "refund approval", text: "Reviewing the query." };
			currentMs = 900;
			yield { type: "preliminary", response: quickResponse };
		});
		const { events, send } = collectEvents();
		const service = createSearchService({
			agentFactory: () => agent,
			chatStore: await createStore(),
			send,
			now,
		});

		await service.start("search-1", "chat-1", "refund approval", []);

		const progress = events.find((event) => event.type === "progress");
		assert(progress !== undefined);
		expect(progress).toEqual({ type: "progress", searchId: "search-1", text: "Reviewing the query." });
		const quick = events.find((event) => event.type === "quick");
		assert(quick !== undefined);
		expect(quick.searchId).toBe("search-1");
		expect(quick.sessionId).toBe("session-quick");
		expect(quick.phase.answer).toBe("**Quick** answer [1]");
		expect(quick.phase.meta).toBe("0.9s · 2 sources");
		expect(quick.phase.evidence).toEqual([
			{
				number: 1,
				title: "Refund policy",
				summary: "Director approval is required before payout.",
				source: "/Users/jeff/Documents/refund-policy.md",
				excerpts: ["Refund exceptions require director approval before payout."],
				confidence: 0.92,
				feedbackId: "session-quick:1",
			},
		]);
	});

	it("maps the complete stream event to a Deep phase", async () => {
		const deepResponse = createResponse("session-deep", "Deep answer", 4);
		const agent = createFakeAgent(async function*() {
			currentMs = 4000;
			yield { type: "complete", response: deepResponse };
		});
		const { events, send } = collectEvents();
		const service = createSearchService({
			agentFactory: () => agent,
			chatStore: await createStore(),
			send,
			now,
		});

		await service.start("search-2", "chat-2", "refund approval", []);

		const deep = events.find((event) => event.type === "deep");
		assert(deep !== undefined);
		expect(deep.searchId).toBe("search-2");
		expect(deep.sessionId).toBe("session-deep");
		expect(deep.phase.answer).toBe("Deep answer");
		expect(deep.phase.meta).toBe("4.0s · 4 sources");
		expect(deep.phase.evidence[0]?.feedbackId).toBe("session-deep:1");
	});

	it("cancels an in-flight stream, emits cancelled once, and stops iteration", async () => {
		let continuedAfterCancel = false;
		let notifyQuick: () => void = () => undefined;
		const quickSeen = new Promise<void>((resolve) => {
			notifyQuick = resolve;
		});
		const agent = createFakeAgent(async function*() {
			yield { type: "preliminary", response: createResponse("session-cancel", "Quick answer", 1) };
			await new Promise<void>(() => undefined);
			continuedAfterCancel = true;
			yield { type: "complete", response: createResponse("session-cancel", "Deep answer", 1) };
		});
		const events: SearchStreamEvent[] = [];
		const service = createSearchService({
			agentFactory: () => agent,
			chatStore: await createStore(),
			send: (_channel, payload) => {
				events.push(payload);
				if (payload.type === "quick") notifyQuick();
			},
			now,
		});

		const started = service.start("search-3", "chat-3", "refund approval", []);
		await quickSeen;
		await service.cancel("search-3");
		await started;

		expect(agent.abort).toHaveBeenCalledTimes(1);
		expect(events.filter((event) => event.type === "cancelled")).toEqual([
			{ type: "cancelled", searchId: "search-3" },
		]);
		expect(events.some((event) => event.type === "deep")).toBe(false);
		expect(continuedAfterCancel).toBe(false);
		const record = await service.historyGet("chat-3");
		expect(record?.messages.at(-1)).toMatchObject({ role: "assistant", stopped: true });
	});

	it("emits an agent failure verbatim as an error event", async () => {
		const agent = createFakeAgent(async function*() {
			yield { type: "progress", sessionId: "", query: "refund approval", text: "Reviewing the query." };
			throw new Error("provider exploded");
		});
		const { events, send } = collectEvents();
		const service = createSearchService({
			agentFactory: () => agent,
			chatStore: await createStore(),
			send,
			now,
		});

		await service.start("search-4", "chat-4", "refund approval", []);

		expect(events).toContainEqual({ type: "error", searchId: "search-4", message: "Error: provider exploded" });
	});

	it("delegates numbered feedback without changing the arrays", async () => {
		const agent = createFakeAgent(async function*() { });
		const service = createSearchService({
			agentFactory: () => agent,
			chatStore: await createStore(),
			send: () => undefined,
			now,
		});

		await service.feedback("session-1", [1, 3], [2]);

		expect(agent.recordFeedbackByNumbers).toHaveBeenCalledWith("session-1", [1, 3], [2]);
		expect(agent.feedbackCalls).toEqual([{ sessionId: "session-1", useful: [1, 3], notUseful: [2] }]);
	});

	it("delegates history reads and clear to the chat store", async () => {
		const store = await createStore();
		const record: ChatRecord = {
			id: "chat-history",
			title: "Refund approval",
			snippet: "Director approval is required.",
			updatedAt: "2026-09-28T10:00:00.000Z",
			messages: [
				{ role: "user", text: "Refund approval", attachments: [], at: "2026-09-28T10:00:00.000Z" },
			],
		};
		await store.save(record);
		const service = createSearchService({
			agentFactory: () => createFakeAgent(async function*() { }),
			chatStore: store,
			send: () => undefined,
			now,
		});

		expect(await service.historyList()).toEqual([
			{
				id: "chat-history",
				title: "Refund approval",
				snippet: "Director approval is required.",
				updatedAt: "2026-09-28T10:00:00.000Z",
			},
		]);
		expect(await service.historyGet("chat-history")).toEqual(record);
		expect(await service.historySearch("approval")).toHaveLength(1);
		await service.historyClear();
		expect(await service.historyList()).toEqual([]);
	});
});
