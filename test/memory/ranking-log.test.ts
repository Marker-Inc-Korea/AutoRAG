import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { JudgedEvidenceRecord } from "../../src/memory/judged-evidence.ts";
import {
	type RankingCandidate,
	RankingLogStore,
	type RankingRunRecord,
	RankingRunRecorder,
	rankingLogPath,
} from "../../src/memory/ranking-log.ts";
import { evidenceChunkKey } from "../../src/retrieval/evidence-id.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ranking-log-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function hit(source: string, content: string, score: number, metadata: Record<string, unknown> = {}): RetrievalResult {
	return { id: `${source}#${content}`, source, content, score, metadata };
}

const QUESTION = "who approved the Q3 budget?";
const STARTED_AT = 1_000;

function recorder(): RankingRunRecorder {
	const run = new RankingRunRecorder({
		runId: "run-1",
		conversationId: "conv-1",
		question: QUESTION,
		startedAt: STARTED_AT,
	});
	run.plan({ searchQueries: [QUESTION, "Q3 budget approval"], datasources: ["slack"], web: false });
	return run;
}

function candidate(record: RankingRunRecord, source: string): RankingCandidate {
	const found = record.candidates.find((entry) => entry.source === source);
	if (found === undefined) throw new Error(`no candidate for ${source}`);
	return found;
}

function judged(source: string, createdAt: number, question = "older question"): JudgedEvidenceRecord {
	return {
		id: `${createdAt}:${source}`,
		sessionId: `s-${createdAt}`,
		conversationId: "past",
		question,
		searchQuery: question,
		method: "minsync",
		source,
		stableEvidenceId: `minsync:${createdAt}`,
		resultNumber: 1,
		title: source,
		excerpt: "x",
		probability: 0.9,
		createdAt,
	};
}

describe("RankingRunRecorder", () => {
	it("folds every channel that surfaced one chunk into a single candidate with its raw ranks and scores", () => {
		const run = recorder();
		const budget = hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 0.42, { method: "minsync" });
		const filler = hit("/docs/filler.md", "Quarterly planning notes.", 0.61, { method: "minsync" });
		run.retrieved({ surface: "minsync", query: QUESTION, results: [filler, budget] });
		run.retrieved({
			surface: "datasource:slack",
			query: "Q3 budget approval",
			results: [hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 1, { method: "slack.keyword" })],
		});
		const record = run.finish({ finalCitationKeys: [], finishedAt: 2_000 });

		const budgetCandidate = candidate(record, "/docs/budget.md");
		expect(record.candidates).toHaveLength(2);
		expect(budgetCandidate.observations).toEqual([
			{ stage: "prefetch", surface: "minsync", method: "minsync", query: QUESTION, rank: 2, score: 0.42 },
			{
				stage: "prefetch",
				surface: "datasource:slack",
				method: "slack.keyword",
				query: "Q3 budget approval",
				rank: 1,
				score: 1,
			},
		]);
		expect(candidate(record, "/docs/filler.md").observations[0]).toMatchObject({ rank: 1, score: 0.61 });
	});

	it("keeps a candidate the reranker cut as unexposed, and records rerank rank and score for the kept ones", () => {
		const run = recorder();
		const kept = hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 0.4);
		const cut = hit("/docs/filler.md", "Quarterly planning notes.", 0.6);
		run.retrieved({ surface: "minsync", query: QUESTION, results: [cut, kept] });
		run.reranked({
			provider: "openrouter",
			model: "test-rerank",
			results: [{ ...kept, score: 0.93, metadata: { rerankScore: 0.93 } }],
		});
		run.exposed({ surface: "baseline", query: QUESTION, results: [kept] });
		const record = run.finish({ finalCitationKeys: [], finishedAt: 2_000 });

		expect(record.reranker).toEqual({ provider: "openrouter", model: "test-rerank" });
		expect(candidate(record, "/docs/budget.md").rerank).toEqual({ rank: 1, score: 0.93 });
		expect(candidate(record, "/docs/budget.md").exposures).toEqual([
			{ surface: "baseline", query: QUESTION, position: 1 },
		]);
		expect(candidate(record, "/docs/filler.md").rerank).toBeUndefined();
		expect(candidate(record, "/docs/filler.md").exposures).toEqual([]);
	});

	it("counts a tool's output both as a retrieval observation and as an exposure, with the merger's corroboration", () => {
		const run = recorder();
		run.tool({
			surface: "search_all_documents",
			query: "budget approver",
			results: [
				hit("/docs/other.md", "Other.", 1, { method: "minsync-hybrid" }),
				hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 0.7, {
					method: "minsync-hybrid",
					rerankScore: 0.88,
					retrievalMethods: ["hybrid", "slack.keyword"],
					sourceChunkCount: 3,
					duplicateHitCount: 2,
				}),
			],
		});
		const record = run.finish({ finalCitationKeys: [], finishedAt: 2_000 });

		const budget = candidate(record, "/docs/budget.md");
		expect(budget.observations).toEqual([
			{
				stage: "tool",
				surface: "search_all_documents",
				method: "minsync-hybrid",
				query: "budget approver",
				rank: 2,
				score: 0.7,
				rerankScore: 0.88,
				retrievalMethodCount: 2,
				sourceChunkCount: 3,
				duplicateHitCount: 2,
			},
		]);
		expect(budget.exposures).toEqual([{ surface: "search_all_documents", query: "budget approver", position: 2 }]);
	});

	it("labels the final answer's citations, draft-only citations, and Jev's relevance and support probabilities", () => {
		const run = recorder();
		const budget = hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 0.5);
		const toc = hit("/docs/toc.md", "Table of contents", 0.4);
		const draft = hit("/docs/draft.md", "Budget draft v1", 0.3);
		const unseen = hit("/docs/unseen.md", "Never cited", 0.2);
		run.exposed({ surface: "baseline", query: QUESTION, results: [budget, toc, draft, unseen] });
		const key = (result: RetrievalResult): string => evidenceChunkKey(result.source, result.content);
		// The fast answer cited the draft; the final answer cited the budget and the TOC, and Jev rejected the TOC.
		run.cited([key(draft)], new Map([[key(draft), 0.8]]));
		run.cited(
			[key(budget), key(toc)],
			new Map([
				[key(budget), 0.95],
				[key(toc), 0.05],
			]),
		);
		run.supported(new Map([[key(budget), 0.82]]));
		const record = run.finish({ finalCitationKeys: [key(budget)], finishedAt: 2_000 });

		expect(record.judgments).toEqual({ citationRelevance: "judged", support: "judged" });
		expect(candidate(record, "/docs/budget.md").labels).toEqual({
			citedFinal: true,
			modelCited: true,
			relevance: 0.95,
			support: 0.82,
		});
		expect(candidate(record, "/docs/toc.md").labels).toEqual({
			citedFinal: false,
			modelCited: true,
			relevance: 0.05,
		});
		expect(candidate(record, "/docs/draft.md").labels).toEqual({
			citedFinal: false,
			modelCited: true,
			relevance: 0.8,
		});
		expect(candidate(record, "/docs/unseen.md").labels).toEqual({ citedFinal: false, modelCited: false });
	});

	it("records an unavailable judgment instead of guessing a label when Jev could not judge", () => {
		const run = recorder();
		const budget = hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 0.5);
		run.exposed({ surface: "baseline", query: QUESTION, results: [budget] });
		run.cited([evidenceChunkKey(budget.source, budget.content)], undefined);
		run.supported(undefined);
		const record = run.finish({
			finalCitationKeys: [evidenceChunkKey(budget.source, budget.content)],
			finishedAt: 2_000,
		});

		expect(record.judgments).toEqual({ citationRelevance: "unavailable", support: "unavailable" });
		expect(candidate(record, "/docs/budget.md").labels).toEqual({ citedFinal: true, modelCited: true });
	});

	it("takes memory priors only from evidence judged before the run started (point-in-time)", () => {
		const run = recorder();
		run.exposed({
			surface: "baseline",
			query: QUESTION,
			results: [hit("/docs/budget.md", "Mina Park approved.", 1)],
		});
		const record = run.finish({
			finalCitationKeys: [],
			finishedAt: 2_000,
			priorEvidence: [
				judged("/docs/budget.md", 10),
				judged("/docs/budget.md", 20),
				judged("/docs/other.md", 30),
				// Judged by this very run, after it started: must not leak into its own features.
				judged("/docs/budget.md", STARTED_AT + 500),
			],
			similarQuestions: [
				{ question: "who signed off the budget?", score: 1, records: [judged("/docs/budget.md", 20)] },
			],
		});

		expect(candidate(record, "/docs/budget.md").memory).toEqual({ priorSupport: 2, similarQuestionSupport: 1 });
	});

	it("leaves the similar-question prior missing, not zero, when memory context was never loaded", () => {
		const run = recorder();
		run.exposed({
			surface: "baseline",
			query: QUESTION,
			results: [hit("/docs/budget.md", "Mina Park approved.", 1)],
		});
		const record = run.finish({ finalCitationKeys: [], finishedAt: 2_000, priorEvidence: [] });

		expect(candidate(record, "/docs/budget.md").memory).toEqual({ priorSupport: 0, similarQuestionSupport: null });
	});

	it("computes text signals against the user's question at log time and never stores the chunk text", () => {
		const run = recorder();
		run.exposed({
			surface: "baseline",
			query: QUESTION,
			results: [
				hit("/docs/budget.md", "Mina Park approved the Q3 budget.", 1),
				hit("/docs/x.md", "unrelated words", 1),
			],
		});
		const record = run.finish({
			finalCitationKeys: [],
			finishedAt: 2_000,
			origin: ({ source }) => (source.startsWith("/docs/") ? "local files" : undefined),
		});

		const budget = candidate(record, "/docs/budget.md");
		expect(budget.text.questionCoverage).toBeGreaterThan(0.5);
		expect(budget.text.bm25).toBeGreaterThan(candidate(record, "/docs/x.md").text.bm25);
		expect(candidate(record, "/docs/x.md").text.questionCoverage).toBe(0);
		expect(budget.origin).toBe("local files");
		expect(budget.contentChars).toBe("Mina Park approved the Q3 budget.".length);
		expect(JSON.stringify(record)).not.toContain("Mina Park approved");
	});

	it("ignores a citation of evidence this run never retrieved (an earlier turn's chunk)", () => {
		const run = recorder();
		run.cited(["/docs/old.md\0old chunk"], new Map([["/docs/old.md\0old chunk", 0.9]]));
		const record = run.finish({ finalCitationKeys: ["/docs/old.md\0old chunk"], finishedAt: 2_000 });
		expect(record.candidates).toEqual([]);
	});
});

describe("RankingLogStore", () => {
	function sample(runId: string, startedAt: number): RankingRunRecord {
		const run = new RankingRunRecorder({ runId, conversationId: "c", question: "q", startedAt });
		run.plan({ searchQueries: ["q"], datasources: [], web: false });
		run.exposed({ surface: "baseline", query: "q", results: [hit("/docs/a.md", "a", 1)] });
		return run.finish({ finalCitationKeys: [], finishedAt: startedAt + 1 });
	}

	it("lives beside the memory file", () => {
		expect(rankingLogPath("/home/u/.autorag/memory.json")).toBe("/home/u/.autorag/memory.json.ranking.db");
	});

	it("appends runs from independent stores and lists them oldest first, each run once", () => {
		const path = join(root, "memory.json.ranking.db");
		const first = new RankingLogStore(path);
		const second = new RankingLogStore(path);
		second.append(sample("late", 300));
		first.append(sample("early", 100));
		first.append(sample("early", 100));
		first.close();
		second.close();

		const reader = new RankingLogStore(path);
		const { runs, skipped } = reader.list();
		reader.close();
		expect(runs.map((run) => run.runId)).toEqual(["early", "late"]);
		expect(skipped).toBe(0);
	});

	it("skips and counts a row that is not a valid run record instead of failing the export", () => {
		const path = join(root, "memory.json.ranking.db");
		const store = new RankingLogStore(path);
		store.append(sample("good", 100));
		store.close();
		const raw = new DatabaseSync(path);
		raw.prepare("INSERT INTO runs (run_id, started_at, record) VALUES (?, ?, ?)").run(
			"broken",
			50,
			'{"schemaVersion":1}',
		);
		raw.close();
		const reopened = new RankingLogStore(path);
		const { runs, skipped } = reopened.list();
		reopened.close();
		expect(runs.map((run) => run.runId)).toEqual(["good"]);
		expect(skipped).toBe(1);
	});

	it("never deletes a file it cannot read: training data is not a cache", () => {
		const path = join(root, "memory.json.ranking.db");
		writeFileSync(path, "not a database at all");
		expect(() => new RankingLogStore(path)).toThrow(path);
		expect(readFileSync(path, "utf-8")).toBe("not a database at all");
	});
});
