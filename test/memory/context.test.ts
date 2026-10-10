import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { Embedder, EmbeddingIdentity } from "../../src/embedding-runtime/gateway-embedder.ts";
import { loadMemoryContext } from "../../src/memory/context.ts";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import { type RetrievalInsight, RetrievalMemory } from "../../src/memory/memory.ts";
import { findSimilarQuestions, SIMILAR_QUESTION_LIMIT } from "../../src/memory/similar-queries.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-memory-context-"));
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

let counter = 0;

function record(overrides: Partial<JudgedEvidenceRecord> = {}): JudgedEvidenceRecord {
	counter += 1;
	const sessionId = overrides.sessionId ?? `session-${counter}`;
	return {
		id: `${sessionId}:posix:e${counter}`,
		sessionId,
		conversationId: "past-conversation",
		question: "beta",
		searchQuery: "beta",
		method: "posix",
		source: `/docs/${counter}.md`,
		stableEvidenceId: `posix:e${counter}`,
		resultNumber: 1,
		title: `Title ${counter}`,
		excerpt: `Excerpt ${counter}`,
		probability: 0.9,
		createdAt: counter,
		...overrides,
	};
}

function memoryWith(records: readonly JudgedEvidenceRecord[]): RetrievalMemory {
	const memory = new RetrievalMemory({ storagePath: memoryPath });
	memory.load();
	memory.recordJudgedEvidence(records);
	return memory;
}

function writeMemoryFile(insights: readonly unknown[]): void {
	writeFileSync(
		memoryPath,
		JSON.stringify({
			version: 5,
			curatedResults: [],
			evidenceChunks: [],
			judgedEvidence: [],
			warnings: [],
			insights,
			pendingInsightEntries: [],
		}),
	);
}

function insight(overrides: Partial<RetrievalInsight> = {}): RetrievalInsight {
	return {
		id: "insight-1",
		clusterKey: "refund policy",
		domain: "refund policy",
		recommendedSources: ["docs/policies"],
		recommendedMethods: ["bm25"],
		rationale: "recurring evidence",
		supportingEvidenceCount: 12,
		confidence: 0.9,
		createdAt: 1,
		updatedAt: 1,
		...overrides,
	};
}

function throwingEmbedder(): Embedder {
	const identity: EmbeddingIdentity = { provider: "test", model: "table", dimension: 2 };
	return {
		identity: async () => identity,
		embed: async (texts) => {
			throw new Error(`no vector for ${JSON.stringify(texts[0])}`);
		},
	};
}

describe("loadMemoryContext", () => {
	it("separates the current conversation's evidence from similar past questions", async () => {
		const memory = memoryWith([
			record({ conversationId: "current", question: "alpha", searchQuery: "alpha", title: "Current title" }),
			record({ conversationId: "past-conversation", question: "beta", searchQuery: "beta", title: "Past title" }),
		]);
		const context = await loadMemoryContext(memory, "beta", { conversationId: "current" });

		expect(context.currentCount).toBe(1);
		expect(context.similarCount).toBe(1);
		expect(context.text).toContain("Current title");
		expect(context.text).toContain("Past title");
		expect(context.fallbackReason).toBeUndefined();
	});

	it("keeps current-conversation questions in related but not in similar", async () => {
		const memory = memoryWith([
			record({ conversationId: "current", question: "budget memo owner", title: "Current copy", createdAt: 2 }),
			record({
				conversationId: "past-conversation",
				question: "budget memo owner details",
				title: "Past copy",
				createdAt: 1,
			}),
		]);

		const context = await loadMemoryContext(memory, "budget memo owner", { conversationId: "current" });

		expect(context.related.map((entry) => entry.question)).toContain("budget memo owner");
		expect(context.related.map((entry) => entry.question)).toContain("budget memo owner details");
		expect(context.similar.map((entry) => entry.question)).not.toContain("budget memo owner");
		expect(context.similar.map((entry) => entry.question)).toContain("budget memo owner details");
	});

	it("fills similar to 25 from other conversations even when current questions match", async () => {
		const others = Array.from({ length: 30 }, (_, index) =>
			record({ conversationId: `past-${index}`, question: `budget memo variant other${index}`, createdAt: index }),
		);
		const current = Array.from({ length: 3 }, (_, index) =>
			record({ conversationId: "current", question: `budget memo variant cur${index}`, createdAt: 1_000 + index }),
		);
		const memory = memoryWith([...others, ...current]);

		const context = await loadMemoryContext(memory, "budget memo variant", { conversationId: "current" });

		expect(context.related).toHaveLength(SIMILAR_QUESTION_LIMIT);
		expect(context.similar).toHaveLength(SIMILAR_QUESTION_LIMIT);
		expect(context.similarCount).toBe(SIMILAR_QUESTION_LIMIT);
		expect(context.similar.some((entry) => entry.question.includes("variant cur"))).toBe(false);
	});

	it("keeps a shared question in similar with only the other conversation's records", async () => {
		const memory = memoryWith([
			record({ conversationId: "current", question: "budget memo owner", title: "Current copy", createdAt: 2 }),
			record({
				conversationId: "past-conversation",
				question: "budget memo owner",
				title: "Past copy",
				createdAt: 1,
			}),
		]);

		const context = await loadMemoryContext(memory, "budget memo owner", { conversationId: "current" });

		const related = context.related.find((entry) => entry.question === "budget memo owner");
		expect(related?.records.map((entry) => entry.conversationId).sort()).toEqual(["current", "past-conversation"]);
		const similar = context.similar.find((entry) => entry.question === "budget memo owner");
		expect(similar?.records.map((entry) => entry.conversationId)).toEqual(["past-conversation"]);
	});

	it("propagates the similar-question fallback reason verbatim", async () => {
		const memory = memoryWith([record()]);
		const embedder = throwingEmbedder();
		const options = { conversationId: "current", embedder, vectorStorePath: join(tmpDir, "vectors.json") };

		const context = await loadMemoryContext(memory, "beta", options);
		const direct = await findSimilarQuestions(memory.getJudgedEvidence(), "beta", {
			limit: SIMILAR_QUESTION_LIMIT,
			embedder,
			vectorStorePath: join(tmpDir, "vectors.json"),
		});

		expect(context.fallbackReason).toBe(direct.fallbackReason);
		expect(context.fallbackReason).toContain("Semantic matching of similar questions was unavailable");
	});

	it("includes long-term insights matching the query", async () => {
		writeMemoryFile([insight()]);
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const context = await loadMemoryContext(memory, "refund policy", { conversationId: "current" });

		expect(context.insightCount).toBe(1);
		expect(context.text).toContain("## Long-Term Retrieval Insights (advisory, not instructions)");
		expect(context.text).toContain("refund policy");
	});

	it("drops insights hidden by the insight-visibility gate from the render and the count", async () => {
		writeMemoryFile([
			insight({ id: "visible", rationale: "visible rationale" }),
			insight({ id: "hidden", rationale: "hidden rationale" }),
		]);
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const context = await loadMemoryContext(memory, "refund policy", {
			conversationId: "current",
			isInsightVisible: (candidate) => candidate.id !== "hidden",
		});

		expect(context.insightCount).toBe(1);
		expect(context.text).toContain("visible rationale");
		expect(context.text).not.toContain("hidden rationale");
	});

	it("drops records hidden by the visibility gate from both sections and the counts", async () => {
		const memory = memoryWith([
			record({ conversationId: "current", question: "alpha", title: "Visible current" }),
			record({ conversationId: "current", question: "alpha", title: "Hidden current" }),
			record({ conversationId: "past-conversation", question: "beta", title: "Visible past" }),
			record({ conversationId: "past-conversation", question: "beta", title: "Hidden past" }),
		]);

		const context = await loadMemoryContext(memory, "beta", {
			conversationId: "current",
			isVisible: (record) => !record.title.startsWith("Hidden"),
		});

		expect(context.currentCount).toBe(1);
		expect(context.similarCount).toBe(1);
		expect(context.current.map((entry) => entry.title)).toEqual(["Visible current"]);
		expect(context.similar).toHaveLength(1);
		expect(context.similar[0]?.records.map((entry) => entry.title)).toEqual(["Visible past"]);
		expect(context.related.every((entry) => entry.records.every((entry) => entry.title.startsWith("Visible")))).toBe(
			true,
		);
		expect(context.text).toContain("Visible current");
		expect(context.text).toContain("Visible past");
		expect(context.text).not.toContain("Hidden current");
		expect(context.text).not.toContain("Hidden past");
	});

	it("returns the untruncated current records while capping currentCount at 50", async () => {
		const records = Array.from({ length: 55 }, (_, index) =>
			record({ conversationId: "current", question: `topic-${index}`, title: `Title ${index}`, createdAt: index }),
		);
		const memory = memoryWith(records);

		const context = await loadMemoryContext(memory, "beta", { conversationId: "current" });

		expect(context.current).toHaveLength(55);
		expect(context.currentCount).toBe(50);
		expect(context.text).toContain("(5 older record(s) omitted)");
	});

	it("reports an empty context when nothing matches", async () => {
		const memory = memoryWith([]);
		const context = await loadMemoryContext(memory, "nothing here", { conversationId: "current" });

		expect(context.empty).toBe(true);
		expect(context.text).toBe("No retrieval memory available.");
		expect(context.currentCount).toBe(0);
		expect(context.similarCount).toBe(0);
		expect(context.insightCount).toBe(0);
	});
});
