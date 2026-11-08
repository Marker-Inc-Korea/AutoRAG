import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildLtrDataset,
	LTR_FEATURE_NAMES,
	type LtrDataset,
	ltrLabel,
	writeLtrDataset,
} from "../../src/memory/ltr-export.ts";
import type { RankingCandidate, RankingLabels, RankingRunRecord } from "../../src/memory/ranking-log.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-export-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function labels(overrides: Partial<RankingLabels> = {}): RankingLabels {
	return { citedFinal: false, modelCited: false, ...overrides };
}

function candidate(chunkId: string, overrides: Partial<RankingCandidate> = {}): RankingCandidate {
	return {
		chunkId,
		source: `/docs/${chunkId}.md`,
		origin: "local files",
		contentChars: 120,
		observations: [],
		exposures: [],
		text: { bm25: 1, questionCoverage: 0.5 },
		memory: { priorSupport: 0, similarQuestionSupport: 0 },
		labels: labels(),
		...overrides,
	};
}

function run(runId: string, startedAt: number, candidates: RankingCandidate[]): RankingRunRecord {
	return {
		schemaVersion: 1,
		runId,
		conversationId: `conv-${runId}`,
		question: `question ${runId}`,
		startedAt,
		finishedAt: startedAt + 1,
		searchQueries: [`question ${runId}`],
		datasources: [],
		web: false,
		judgments: { citationRelevance: "judged", support: "judged" },
		candidates,
	};
}

function feature(dataset: LtrDataset, split: "train" | "valid", row: number, name: string): number {
	const index = LTR_FEATURE_NAMES.indexOf(name as (typeof LTR_FEATURE_NAMES)[number]);
	if (index < 0) throw new Error(`unknown feature ${name}`);
	return dataset[split].rows[row]?.features[index] ?? Number.NaN;
}

const positive = (id: string): RankingCandidate =>
	candidate(id, { labels: labels({ citedFinal: true, modelCited: true }) });

describe("ltrLabel", () => {
	it("grades evidence by how far it got: supported > cited in the final answer > cited only by a dropped draft > nothing", () => {
		expect(ltrLabel(labels({ citedFinal: true, modelCited: true, support: 0.7 }))).toBe(3);
		expect(ltrLabel(labels({ citedFinal: true, modelCited: true, support: 0.69 }))).toBe(2);
		expect(ltrLabel(labels({ citedFinal: true, modelCited: true }))).toBe(2);
		expect(ltrLabel(labels({ modelCited: true }))).toBe(1);
		expect(ltrLabel(labels({ modelCited: true, relevance: 0.6 }))).toBe(1);
		expect(ltrLabel(labels())).toBe(0);
	});

	it("treats a citation Jev found unrelated as a negative, not as a weak positive", () => {
		expect(ltrLabel(labels({ modelCited: true, relevance: 0.29 }))).toBe(0);
		expect(ltrLabel(labels({ modelCited: true, relevance: 0.3 }))).toBe(1);
	});
});

describe("buildLtrDataset", () => {
	it("turns each run into one contiguous query group and drops groups that teach no comparison", () => {
		const dataset = buildLtrDataset(
			[
				run("a", 1, [positive("a1"), candidate("a2"), candidate("a3")]),
				// Every label equal: LambdaRank gets no gradient from it.
				run("b", 2, [candidate("b1"), candidate("b2")]),
				// A single candidate is no ranking.
				run("c", 3, [positive("c1")]),
				run("d", 4, [candidate("d1"), positive("d2")]),
			],
			{ validFraction: 0 },
		);

		expect(dataset.train.groups).toEqual([3, 2]);
		expect(dataset.train.rows.map((row) => `${row.runId}/${row.chunkId}`)).toEqual([
			"a/a1",
			"a/a2",
			"a/a3",
			"d/d1",
			"d/d2",
		]);
		expect(dataset.train.rows.map((row) => row.label)).toEqual([2, 0, 0, 0, 2]);
		expect(dataset.manifest.skippedGroups).toEqual({ singleCandidate: 1, noLabelVariance: 1 });
		expect(dataset.valid.rows).toEqual([]);
	});

	it("splits by whole runs in time order, newest runs to validation, so no query is on both sides", () => {
		const runs = [5, 1, 4, 2, 3].map((startedAt) =>
			run(`r${startedAt}`, startedAt, [positive(`r${startedAt}-p`), candidate(`r${startedAt}-n`)]),
		);
		const dataset = buildLtrDataset(runs, { validFraction: 0.4 });

		expect([...new Set(dataset.train.rows.map((row) => row.runId))]).toEqual(["r1", "r2", "r3"]);
		expect([...new Set(dataset.valid.rows.map((row) => row.runId))]).toEqual(["r4", "r5"]);
		expect(dataset.train.groups).toEqual([2, 2, 2]);
		expect(dataset.valid.groups).toEqual([2, 2]);
	});

	it("puts every candidate of one run in one split even when a conversation spans the boundary", () => {
		const dataset = buildLtrDataset(
			[run("x", 1, [positive("x1"), candidate("x2")]), run("y", 2, [positive("y1"), candidate("y2")])],
			{ validFraction: 0.5 },
		);
		expect(dataset.train.groups.reduce((sum, size) => sum + size, 0)).toBe(dataset.train.rows.length);
		expect(dataset.valid.groups.reduce((sum, size) => sum + size, 0)).toBe(dataset.valid.rows.length);
	});

	it("builds per-channel rank and score features from the raw observations, missing when a channel never found the chunk", () => {
		const both = candidate("both", {
			labels: labels({ citedFinal: true, modelCited: true }),
			observations: [
				{ stage: "prefetch", surface: "minsync", method: "minsync", query: "q", rank: 3, score: 0.4 },
				{ stage: "prefetch", surface: "minsync", method: "minsync", query: "q2", rank: 1, score: 0.7 },
				{ stage: "prefetch", surface: "datasource:slack", method: "slack.keyword", query: "q", rank: 2, score: 5 },
			],
			rerank: { rank: 1, score: 0.9 },
		});
		const denseOnly = candidate("dense", {
			observations: [{ stage: "prefetch", surface: "minsync", method: "minsync", query: "q", rank: 1, score: 0.8 }],
		});
		const dataset = buildLtrDataset([run("a", 1, [both, denseOnly])], { validFraction: 0 });

		expect(feature(dataset, "train", 0, "dense_best_rank")).toBe(1);
		expect(feature(dataset, "train", 0, "dense_max_score")).toBe(0.7);
		expect(feature(dataset, "train", 0, "dense_reciprocal_rank_sum")).toBeCloseTo(1 + 1 / 3);
		expect(feature(dataset, "train", 0, "dense_hits")).toBe(2);
		expect(feature(dataset, "train", 0, "datasource_best_rank")).toBe(2);
		expect(feature(dataset, "train", 0, "channel_count")).toBe(2);
		expect(feature(dataset, "train", 0, "query_hit_count")).toBe(2);
		expect(feature(dataset, "train", 0, "rerank_rank")).toBe(1);
		expect(feature(dataset, "train", 0, "rerank_score")).toBe(0.9);
		expect(feature(dataset, "train", 1, "datasource_best_rank")).toBeNaN();
		expect(feature(dataset, "train", 1, "datasource_hits")).toBe(0);
		expect(feature(dataset, "train", 1, "rerank_score")).toBeNaN();
		expect(feature(dataset, "train", 1, "sparse_best_rank")).toBeNaN();
	});

	it("never turns how the model was shown a chunk into a feature: exposure goes to the position column", () => {
		const shownFirst = candidate("shown", {
			labels: labels({ citedFinal: true, modelCited: true }),
			exposures: [
				{ surface: "search_all_documents", query: "q", position: 4 },
				{ surface: "baseline", query: "q", position: 2 },
			],
		});
		const neverShown = candidate("hidden");
		const dataset = buildLtrDataset([run("a", 1, [shownFirst, neverShown])], { validFraction: 0 });

		expect(LTR_FEATURE_NAMES.some((name) => /expos|position/u.test(name))).toBe(false);
		expect(dataset.train.rows.map((row) => row.position)).toEqual([2, 0]);
	});

	it("keeps a memory prior that was never consulted missing instead of zero", () => {
		const consulted = candidate("p", {
			labels: labels({ citedFinal: true, modelCited: true }),
			memory: { priorSupport: 3, similarQuestionSupport: 2 },
		});
		const notConsulted = candidate("n", { memory: { priorSupport: 0, similarQuestionSupport: null } });
		const dataset = buildLtrDataset([run("a", 1, [consulted, notConsulted])], { validFraction: 0 });

		expect(feature(dataset, "train", 0, "memory_similar_question_support")).toBe(2);
		expect(feature(dataset, "train", 0, "memory_prior_support")).toBe(3);
		expect(feature(dataset, "train", 1, "memory_similar_question_support")).toBeNaN();
	});

	it("rejects a validation fraction outside [0, 1)", () => {
		expect(() => buildLtrDataset([], { validFraction: 1 })).toThrow(/validFraction/u);
		expect(() => buildLtrDataset([], { validFraction: -0.1 })).toThrow(/validFraction/u);
	});
});

describe("writeLtrDataset", () => {
	it("writes LightGBM text files: a header row, label first, nan for missing, and .query/.position sidecars", () => {
		const dataset = buildLtrDataset(
			[
				run("a", 1, [
					candidate("a1", {
						labels: labels({ citedFinal: true, modelCited: true, support: 0.9 }),
						exposures: [{ surface: "baseline", query: "q", position: 1 }],
					}),
					candidate("a2"),
				]),
			],
			{ validFraction: 0 },
		);
		const written = writeLtrDataset(dataset, root);

		const lines = readFileSync(join(root, "train.tsv"), "utf-8").trimEnd().split("\n");
		expect(lines[0]).toBe(["label", ...LTR_FEATURE_NAMES].join("\t"));
		expect(lines).toHaveLength(3);
		expect(lines[1]?.split("\t")[0]).toBe("3");
		expect(lines[2]?.split("\t")[0]).toBe("0");
		expect(lines[1]?.split("\t")).toHaveLength(LTR_FEATURE_NAMES.length + 1);
		expect(lines[2]).toContain("nan");
		expect(readFileSync(join(root, "train.tsv.query"), "utf-8")).toBe("2\n");
		expect(readFileSync(join(root, "train.tsv.position"), "utf-8")).toBe("1\n0\n");
		const rows = readFileSync(join(root, "train.rows.jsonl"), "utf-8")
			.trimEnd()
			.split("\n")
			.map((line) => JSON.parse(line));
		expect(rows.map((row) => row.chunkId)).toEqual(["a1", "a2"]);
		const manifest = JSON.parse(readFileSync(join(root, "manifest.json"), "utf-8"));
		expect(manifest).toMatchObject({
			featureNames: [...LTR_FEATURE_NAMES],
			train: { groups: 1, rows: 2 },
			valid: { groups: 0, rows: 0 },
			labelCounts: { "0": 1, "3": 1 },
		});
		expect(written.files).toContain(join(root, "train.tsv"));
		expect(written.files).not.toContain(join(root, "valid.tsv"));
	});
});
