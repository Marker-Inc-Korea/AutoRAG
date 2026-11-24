import { describe, expect, it } from "vitest";
import { buildServingDataset, LTR_SERVING_FEATURE_NAMES, servingRows } from "../../src/memory/ltr-export.ts";
import { hit, recordedRun } from "./helpers.ts";

describe("serving view of a logged run", () => {
	it("excludes every feature that does not exist at rerank time", () => {
		for (const name of [
			"text_bm25",
			"memory_prior_support",
			"memory_similar_question_support",
			"origin_prior_win_rate",
			"origin_prior_candidates",
			"tool_rerank_score_max",
			"merger_retrieval_method_count",
			"candidate_pool_size",
			"origin_id",
			"method_id",
		]) {
			expect(LTR_SERVING_FEATURE_NAMES).not.toContain(name);
		}
		expect(LTR_SERVING_FEATURE_NAMES).toContain("rerank_score");
		expect(LTR_SERVING_FEATURE_NAMES).toContain("dense_best_rank");
		expect(LTR_SERVING_FEATURE_NAMES).toContain("sparse_best_rank");
		expect(LTR_SERVING_FEATURE_NAMES).toContain("sparse_max_score");
		expect(LTR_SERVING_FEATURE_NAMES).toContain("sparse_reciprocal_rank_sum");
		expect(LTR_SERVING_FEATURE_NAMES).toContain("sparse_hits");
	});

	it("populates sparse_* serving features from minsync-sparse prefetch observations and leaves missing channels NaN", () => {
		const { recorder, pool } = recordedRun("r-sparse", 3_000, { poolSize: 4 });
		const sparseFirst = pool[2];
		const sparseSecond = pool[0];
		if (sparseFirst === undefined || sparseSecond === undefined) throw new Error("pool too small");
		recorder.retrieved({
			surface: "minsync",
			query: "sparse prefetch",
			results: [
				hit(sparseFirst.source, sparseFirst.content, 14.5, "minsync-sparse"),
				hit(sparseSecond.source, sparseSecond.content, 8.2, "minsync-sparse"),
			],
		});
		const rows = servingRows(
			recorder.finish({ finalCitationKeys: [], finishedAt: 3_010, origin: () => "local files" }),
		);
		const idx = (name: (typeof LTR_SERVING_FEATURE_NAMES)[number]) => LTR_SERVING_FEATURE_NAMES.indexOf(name);
		const rowFor = (source: string) => {
			const found = rows.find((row) => row.source === source);
			if (found === undefined) throw new Error(`missing row for ${source}`);
			return found.features;
		};
		const first = rowFor(sparseFirst.source);
		expect(first[idx("sparse_best_rank")]).toBe(1);
		expect(first[idx("sparse_max_score")]).toBeCloseTo(14.5, 6);
		expect(first[idx("sparse_reciprocal_rank_sum")]).toBeCloseTo(1, 6);
		expect(first[idx("sparse_hits")]).toBe(1);

		const absent = rowFor(pool[1]!.source);
		expect(Number.isNaN(absent[idx("sparse_best_rank")])).toBe(true);
		expect(Number.isNaN(absent[idx("sparse_max_score")])).toBe(true);
		expect(absent[idx("sparse_reciprocal_rank_sum")]).toBe(0);
		expect(absent[idx("sparse_hits")]).toBe(0);
	});

	it("scores only the candidates the reranker returned", () => {
		const { recorder, pool } = recordedRun("r1", 1_000, { poolSize: 6 });
		recorder.retrieved({ surface: "minsync", query: "q2", results: [hit("/docs/extra.md", "never reranked", 0.1)] });
		const rows = servingRows(
			recorder.finish({ finalCitationKeys: [], finishedAt: 1_010, origin: () => "local files" }),
		);
		expect(rows).toHaveLength(pool.length);
		expect(rows.every((row) => row.features.length === LTR_SERVING_FEATURE_NAMES.length)).toBe(true);
	});

	it("gives a rerank-time snapshot the same features the final log record yields after later tool results", () => {
		const { recorder, pool } = recordedRun("r2", 2_000, { poolSize: 6 });
		const known = pool[2];
		if (known === undefined) throw new Error("pool has no third chunk");
		const origin = () => "local files";
		const atRerank = servingRows(recorder.finish({ finalCitationKeys: [], finishedAt: 2_001, origin }));
		// The model then calls a retrieval tool: new observations on a known chunk, and a brand-new chunk.
		recorder.tool({
			surface: "search_all_documents",
			query: "follow up",
			results: [hit(known.source, known.content, 0.99), hit("/docs/late.md", "late chunk", 0.5)],
		});
		const final = servingRows(recorder.finish({ finalCitationKeys: [], finishedAt: 2_500, origin }));
		expect(final).toHaveLength(atRerank.length);
		const byId = new Map(final.map((row) => [row.chunkId, row.features]));
		for (const row of atRerank) expect(byId.get(row.chunkId)).toEqual(row.features);
	});
});

describe("buildServingDataset", () => {
	function runs(count: number) {
		return Array.from({ length: count }, (_, index) =>
			recordedRun(`run-${index}`, 10_000 + index * 1_000, { gold: index % 6 }).finish(true),
		);
	}

	it("splits whole runs in time order into train, validation and holdout", () => {
		const dataset = buildServingDataset(runs(40), { validFraction: 0.15, holdoutFraction: 0.15 });
		expect(dataset.train.groups.length + dataset.valid.groups.length + dataset.holdout.groups.length).toBe(40);
		expect(dataset.holdout.groups).toHaveLength(6);
		expect(dataset.valid.groups).toHaveLength(6);
		const trainRuns = new Set(dataset.train.rows.map((row) => row.runId));
		for (const row of [...dataset.valid.rows, ...dataset.holdout.rows]) expect(trainRuns.has(row.runId)).toBe(false);
		const holdoutStart = Math.min(...dataset.holdout.rows.map((row) => Number(row.runId.split("-")[1])));
		const maxTrain = Math.max(...dataset.train.rows.map((row) => Number(row.runId.split("-")[1])));
		expect(holdoutStart).toBeGreaterThan(maxTrain);
	});

	it("drops runs that teach nothing: no reranker output, or every candidate on one label", () => {
		const good = recordedRun("good", 1_000, { gold: 1 }).finish(true);
		const nothingCited = recordedRun("none", 2_000).finish(false);
		const dataset = buildServingDataset([good, nothingCited], { validFraction: 0, holdoutFraction: 0 });
		expect(dataset.train.groups).toHaveLength(1);
		expect(dataset.skipped.noLabelVariance).toBe(1);
	});

	it("labels the cited chunk above the rest", () => {
		const dataset = buildServingDataset([recordedRun("l", 1_000, { gold: 3 }).finish(true)], {
			validFraction: 0,
			holdoutFraction: 0,
		});
		expect(dataset.train.rows.filter((row) => row.label > 0)).toHaveLength(1);
	});
});
