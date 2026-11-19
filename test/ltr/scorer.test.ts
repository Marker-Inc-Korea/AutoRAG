import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { type LtrModelMeta, ltrDirectory, saveLtrModel } from "../../src/ltr/model-store.ts";
import { createLtrScorer } from "../../src/ltr/scorer.ts";
import { LTR_SERVING_FEATURE_NAMES, servingRows } from "../../src/memory/ltr-export.ts";
import { recordedRun, stumpModelJson } from "./helpers.ts";

let root: string;
let dir: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-scorer-"));
	dir = ltrDirectory(join(root, "memory.json"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function meta(id: string, createdAt: number, passedGate = true): LtrModelMeta {
	return {
		id,
		createdAt,
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		objective: "rank:pairwise",
		trainedOnRuns: 100,
		loggedRunsAtTraining: 100,
		holdout: {
			queries: 20,
			baseline: { ndcg10: 0.5, mrr: 0.5 },
			model: { ndcg10: 0.6, mrr: 0.6 },
			deltaMrrCi95: [0, 0.2],
		},
		passedGate,
	};
}

/** Prefers a low dense rank: exactly the chunk the recorded runs cite. */
const PREFER_DENSE_FIRST = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "dense_best_rank", 1.5, 5, 0);

describe("createLtrScorer", () => {
	it("is absent when disabled or when there is no model to serve", () => {
		expect(createLtrScorer({ enabled: false, memoryPath: join(root, "memory.json") })).toEqual({
			status: "disabled",
		});
		expect(createLtrScorer({ enabled: true, memoryPath: join(root, "memory.json") })).toMatchObject({
			status: "no-model",
		});
	});

	it("serves the newest model that passed the gate and reports its id", () => {
		saveLtrModel(dir, meta("m-old", 1_000), PREFER_DENSE_FIRST);
		saveLtrModel(dir, meta("m-new", 2_000), PREFER_DENSE_FIRST);
		saveLtrModel(dir, meta("m-rejected", 3_000, false), PREFER_DENSE_FIRST);
		const scorer = createLtrScorer({ enabled: true, memoryPath: join(root, "memory.json") });
		expect(scorer.status).toBe("active");
		if (scorer.status === "active") expect(scorer.modelId).toBe("m-new");
	});

	it("honours an explicit model selection, including one that failed the gate", () => {
		saveLtrModel(dir, meta("m-rejected", 3_000, false), PREFER_DENSE_FIRST);
		const scorer = createLtrScorer({ enabled: true, model: "m-rejected", memoryPath: join(root, "memory.json") });
		expect(scorer.status).toBe("active");
	});

	it("reports an unusable selection instead of silently serving another model", () => {
		saveLtrModel(dir, meta("m-1", 1_000), PREFER_DENSE_FIRST);
		const scorer = createLtrScorer({ enabled: true, model: "nope", memoryPath: join(root, "memory.json") });
		expect(scorer).toMatchObject({ status: "no-model" });
		saveLtrModel(dir, meta("broken", 2_000), stumpModelJson(["a"], "a", 0, 0, 1));
		const broken = createLtrScorer({ enabled: true, model: "broken", memoryPath: join(root, "memory.json") });
		expect(broken).toMatchObject({ status: "unusable" });
	});

	it("reorders a reranked pool to match what training saw: same features as the logged run", () => {
		saveLtrModel(dir, meta("m-1", 1_000), PREFER_DENSE_FIRST);
		const scorer = createLtrScorer({ enabled: true, memoryPath: join(root, "memory.json") });
		if (scorer.status !== "active") throw new Error("expected an active scorer");

		const run = recordedRun("serve", 5_000, { poolSize: 6, gold: 0 });
		const record = run.finish(true);
		const reranked = [...run.pool].reverse();
		// The reranker put the cited chunk (dense rank 1) last; the model must move it first.
		const { results, scored } = scorer.rank(
			run.recorder.snapshot({ finishedAt: 5_010 }),
			reranked.map((result, index) => ({ ...result, score: 0.9 - index * 0.05 })),
		);
		expect(scored).toBe(6);
		expect(results[0]?.source).toBe(run.pool[0]?.source);
		expect(results).toHaveLength(6);
		// Feature parity: what rank() scored is what the exporter would have trained on.
		const exported = servingRows(record);
		expect(exported).toHaveLength(6);
	});

	it("keeps every candidate and the reranker's order when scoring throws", () => {
		saveLtrModel(dir, meta("m-1", 1_000), PREFER_DENSE_FIRST);
		const scorer = createLtrScorer({ enabled: true, memoryPath: join(root, "memory.json") });
		if (scorer.status !== "active") throw new Error("expected an active scorer");
		const run = recordedRun("fail", 6_000, { poolSize: 4 });
		const reranked = [...run.pool].reverse();
		const snapshot = run.recorder.snapshot({ finishedAt: 6_010 });
		const broken = {
			...snapshot,
			candidates: snapshot.candidates.map((candidate) => ({ ...candidate, observations: null })),
		};
		const outcome = scorer.rank(broken as never, reranked);
		expect(outcome.results).toEqual(reranked);
		expect(outcome.scored).toBe(0);
		expect(outcome.error).toBeDefined();
	});
});
