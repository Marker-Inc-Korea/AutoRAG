import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import { type InsightExtractor, RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-judged-memory-"));
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
	vi.restoreAllMocks();
});

let counter = 0;

function record(overrides: Partial<JudgedEvidenceRecord> = {}): JudgedEvidenceRecord {
	counter += 1;
	const sessionId = overrides.sessionId ?? `session-${counter}`;
	const stableEvidenceId = overrides.stableEvidenceId ?? `method:evidence-${counter}`;
	return {
		id: `${sessionId}:${stableEvidenceId}`,
		sessionId,
		conversationId: "conversation-a",
		question: `topic${counter}`,
		searchQuery: `topic${counter}`,
		method: "search_datasource_kakao",
		source: `/kakao/default/${counter}`,
		stableEvidenceId,
		resultNumber: 1,
		title: `Title ${counter}`,
		excerpt: `Excerpt ${counter}`,
		probability: 0.9,
		createdAt: 1_000 + counter,
		...overrides,
	};
}

describe("RetrievalMemory judged evidence", () => {
	it("starts with an empty v5 schema that has no feedback or signal fields", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const schema = memory.getSchema() as unknown as Record<string, unknown>;
		expect(schema.version).toBe(5);
		expect(schema.judgedEvidence).toEqual([]);
		expect(schema.pendingInsightEntries).toEqual([]);
		expect(schema).not.toHaveProperty("feedbackSignals");
		expect(schema).not.toHaveProperty("signalDefaults");
		expect(schema).not.toHaveProperty("pendingInsightSignals");
	});

	it("persists judged evidence and reloads it", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const first = record({ question: "who approved the Q3 budget" });
		memory.recordJudgedEvidence([first]);
		memory.save();

		const reloaded = new RetrievalMemory({ storagePath: memoryPath });
		reloaded.load();
		expect(reloaded.getJudgedEvidence()).toEqual([first]);
	});

	it("ignores a judged record that is already stored", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const first = record();
		memory.recordJudgedEvidence([first]);
		memory.recordJudgedEvidence([{ ...first, probability: 0.71 }]);
		memory.save();
		expect(memory.getJudgedEvidence()).toHaveLength(1);
		expect(memory.getJudgedEvidence()[0]?.probability).toBe(0.9);
	});

	it("keeps every record: there is no 500-record cap", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence(Array.from({ length: 650 }, () => record()));
		for (let index = 0; index < 650; index++) {
			memory.recordCuratedResultsSession({
				sessionId: `curated-${index}`,
				query: `curated question ${index}`,
				results: [
					{
						number: 1,
						title: "t",
						summary: "s",
						content: "c",
						method: "m",
						source: "src",
						evidenceRefs: [
							{ method: "m", source: "src", content: `content ${index}`, stableEvidenceId: `m:${index}` },
						],
					},
				],
			});
		}
		memory.save();

		const reloaded = new RetrievalMemory({ storagePath: memoryPath });
		reloaded.load();
		expect(reloaded.getJudgedEvidence()).toHaveLength(650);
		expect(reloaded.getSchema().curatedResults).toHaveLength(650);
		expect(reloaded.getSchema().evidenceChunks).toHaveLength(650);
	});

	it("returns only the current conversation's evidence, oldest first", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const later = record({ conversationId: "conversation-a", createdAt: 5_000 });
		const earlier = record({ conversationId: "conversation-a", createdAt: 2_000 });
		const other = record({ conversationId: "conversation-b" });
		memory.recordJudgedEvidence([later, other, earlier]);
		expect(memory.getConversationEvidence("conversation-a")).toEqual([earlier, later]);
		expect(memory.getConversationEvidence("conversation-b")).toEqual([other]);
		expect(memory.getConversationEvidence("missing")).toEqual([]);
	});

	it("merges judged evidence saved by two instances without losing either", () => {
		const left = new RetrievalMemory({ storagePath: memoryPath });
		const right = new RetrievalMemory({ storagePath: memoryPath });
		left.load();
		right.load();
		const a = record();
		const b = record();
		left.recordJudgedEvidence([a]);
		right.recordJudgedEvidence([b]);
		left.save();
		right.save();

		const reloaded = new RetrievalMemory({ storagePath: memoryPath });
		reloaded.load();
		expect(reloaded.getJudgedEvidence().map((entry) => entry.id)).toEqual([a.id, b.id]);
	});
});

describe("RetrievalMemory v4 migration", () => {
	function writeV4(): void {
		writeFileSync(
			memoryPath,
			JSON.stringify({
				version: 4,
				curatedResults: [
					{
						resultId: "s1:1",
						sessionId: "s1",
						number: 1,
						query: "old question",
						title: "Old",
						summary: "old summary",
						resultHash: "h",
						evidenceIds: ["m:1"],
						createdAt: 10,
						verdict: "useful",
						remote: true,
					},
				],
				evidenceChunks: [
					{
						stableEvidenceId: "m:1",
						method: "m",
						source: "src",
						excerpt: "old evidence",
						excerptHash: "x",
						firstSeenAt: 10,
						lastSeenAt: 10,
					},
				],
				feedbackSignals: [{ id: "f", query: "old question", sentiment: "useful", weight: 1 }],
				signalDefaults: { explicitWeight: 1, followupWeight: 0.25, retryWeight: -0.25, implicitCap: 0.5 },
				warnings: [],
				insights: [
					{
						id: "insight:1",
						clusterKey: "old question",
						domain: "old question",
						recommendedSources: ["src"],
						recommendedMethods: ["m"],
						rationale: "kept",
						supportingSignalCount: 7,
						confidence: 0.8,
						createdAt: 1,
						updatedAt: 2,
					},
				],
				pendingInsightSignals: [{ signal: { query: "q" } }],
			}),
		);
	}

	it("keeps curated results, evidence, and insights and drops feedback state", () => {
		writeV4();
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const schema = memory.getSchema();
		expect(schema.version).toBe(5);
		expect(schema.curatedResults.map((result) => result.resultId)).toEqual(["s1:1"]);
		expect(schema.curatedResults[0]).not.toHaveProperty("verdict");
		expect(schema.curatedResults[0]).not.toHaveProperty("remote");
		expect(schema.evidenceChunks.map((chunk) => chunk.stableEvidenceId)).toEqual(["m:1"]);
		expect(schema.insights).toHaveLength(1);
		expect(schema.insights[0]?.supportingEvidenceCount).toBe(7);
		expect(schema.judgedEvidence).toEqual([]);
		expect(schema.pendingInsightEntries).toEqual([]);
		expect(schema).not.toHaveProperty("feedbackSignals");
		expect(schema).not.toHaveProperty("pendingInsightSignals");
	});

	it("rewrites the file as v5 on the next save", () => {
		writeV4();
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.save();
		const onDisk = JSON.parse(readFileSync(memoryPath, "utf-8")) as Record<string, unknown>;
		expect(onDisk.version).toBe(5);
		expect(onDisk).not.toHaveProperty("feedbackSignals");
	});

	it("starts fresh with a non-path warning when the file is not a known version", () => {
		writeFileSync(memoryPath, JSON.stringify({ version: 3, entries: [] }));
		vi.spyOn(console, "warn").mockImplementation(() => undefined);
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getSchema().warnings.map((warning) => warning.code)).toEqual(["memory-reset"]);
		expect(JSON.stringify(memory.getSchema().warnings)).not.toContain(tmpDir);
	});
});

describe("RetrievalMemory 100-record insight batches", () => {
	function kakaoEvidence(sessionIndex: number, evidenceIndex: number): JudgedEvidenceRecord {
		return record({
			sessionId: `kakao-session-${sessionIndex}`,
			stableEvidenceId: `kakao:${sessionIndex}-${evidenceIndex}`,
			question: "kakao announcement schedule for the hackathon",
			method: "search_datasource_kakao",
			source: "/kakao/default/room",
		});
	}

	function fullBatch(): JudgedEvidenceRecord[] {
		const clustered = Array.from({ length: 60 }, (_, index) => kakaoEvidence(Math.floor(index / 10), index % 10));
		const noise = Array.from({ length: 40 }, () => record());
		return [...clustered, ...noise];
	}

	it("waits until 100 judged records have accumulated before summarizing", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence(fullBatch().slice(0, 99));
		memory.save();
		expect(memory.getSchema().insights).toEqual([]);
		expect(memory.getSchema().pendingInsightEntries).toHaveLength(99);
	});

	it("summarizes a complete batch into a long-term insight and clears the buffer", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence(fullBatch());
		memory.save();

		const { insights, pendingInsightEntries, judgedEvidence } = memory.getSchema();
		expect(pendingInsightEntries).toEqual([]);
		expect(judgedEvidence).toHaveLength(100);
		expect(insights).toHaveLength(1);
		expect(insights[0]).toMatchObject({
			recommendedMethods: ["search_datasource_kakao"],
			recommendedSources: ["/kakao/default/room"],
			supportingEvidenceCount: 60,
		});
		expect(insights[0]?.confidence).toBeGreaterThan(0.8);
		expect(insights[0]?.rationale).toContain("advisory");
	});

	it("keeps the remainder of an over-full buffer for the next batch", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence([...fullBatch(), ...Array.from({ length: 7 }, () => record())]);
		memory.save();
		expect(memory.getSchema().pendingInsightEntries).toHaveLength(7);
	});

	it("does not create an insight from a cluster seen in a single search run", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const singleRun = Array.from({ length: 60 }, (_, index) => kakaoEvidence(0, index));
		memory.recordJudgedEvidence([...singleRun, ...Array.from({ length: 40 }, () => record())]);
		memory.save();
		expect(memory.getSchema().insights).toEqual([]);
		expect(memory.getSchema().pendingInsightEntries).toEqual([]);
	});

	it("accumulates records across saves and merges repeated batches into one insight", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence(fullBatch().slice(0, 50));
		memory.save();
		memory.recordJudgedEvidence(fullBatch().slice(50));
		memory.save();
		expect(memory.getSchema().insights).toHaveLength(1);

		const again = fullBatch().map((entry) => ({
			...entry,
			id: `again:${entry.id}`,
			sessionId: `again-${entry.sessionId}`,
		}));
		memory.recordJudgedEvidence(again);
		memory.save();
		expect(memory.getSchema().insights).toHaveLength(1);
		expect(memory.getSchema().insights[0]?.supportingEvidenceCount).toBeGreaterThan(60);
	});

	it("matches stored insights to similar questions", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordJudgedEvidence(fullBatch());
		memory.save();
		expect(memory.getInsights("kakao announcement schedule for the hackathon")).toHaveLength(1);
		expect(memory.getInsights("tax filing deadline")).toEqual([]);
	});

	it("stays fail-open when the extractor throws", () => {
		vi.spyOn(console, "warn").mockImplementation(() => undefined);
		const extractor: InsightExtractor = () => {
			throw new Error("boom");
		};
		const memory = new RetrievalMemory({ storagePath: memoryPath, insightExtractor: extractor });
		memory.load();
		memory.recordJudgedEvidence(fullBatch());
		expect(() => memory.save()).not.toThrow();
		expect(memory.getSchema().insights).toEqual([]);
		expect(memory.getSchema().warnings.map((warning) => warning.code)).toContain("insight-extraction-failed");
		expect(existsSync(memoryPath)).toBe(true);
	});

	it("creates the storage directory on first save", () => {
		const nested = join(tmpDir, "a", "b", "memory.json");
		mkdirSync(join(tmpDir, "a"), { recursive: true });
		const memory = new RetrievalMemory({ storagePath: nested });
		memory.load();
		memory.recordJudgedEvidence([record()]);
		memory.save();
		expect(existsSync(nested)).toBe(true);
	});
});
