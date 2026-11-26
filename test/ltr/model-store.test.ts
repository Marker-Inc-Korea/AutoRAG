import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	deleteLtrModel,
	isLtrModelSelection,
	type LtrModelMeta,
	listLtrModels,
	loadLtrModel,
	ltrDirectory,
	readLtrState,
	resolveLtrModel,
	saveLtrModel,
	writeLtrState,
} from "../../src/ltr/model-store.ts";
import { LTR_SERVING_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { stumpModelJson } from "./helpers.ts";

let root: string;
let dir: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ltr-store-"));
	dir = ltrDirectory(join(root, "memory.json"));
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function meta(id: string, createdAt: number, passed: boolean): LtrModelMeta {
	return {
		id,
		createdAt,
		featureNames: [...LTR_SERVING_FEATURE_NAMES],
		objective: "rank:pairwise",
		trainedOnRuns: 100,
		loggedRunsAtTraining: 120,
		holdout: {
			queries: 30,
			baseline: { ndcg10: 0.7, mrr: 0.6 },
			model: { ndcg10: passed ? 0.75 : 0.65, mrr: passed ? 0.66 : 0.55 },
			deltaMrrCi95: [-0.01, 0.1],
		},
		passedGate: passed,
	};
}
const MODEL = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "rerank_score", 0.5, 0, 1);

describe("ltr model store", () => {
	it("keeps models beside the memory file and lists them newest first", () => {
		expect(dir).toBe(`${join(root, "memory.json")}.ltr`);
		saveLtrModel(dir, meta("m-1", 1_000, true), MODEL);
		saveLtrModel(dir, meta("m-2", 2_000, false), MODEL);
		expect(listLtrModels(dir).map((entry) => entry.id)).toEqual(["m-2", "m-1"]);
	});

	it("loads a model that can score a feature row", () => {
		saveLtrModel(dir, meta("m-1", 1_000, true), MODEL);
		const loaded = loadLtrModel(dir, "m-1");
		const row = LTR_SERVING_FEATURE_NAMES.map((name) => (name === "rerank_score" ? 0.9 : 0));
		expect(loaded.model.predict(row)).toBe(1);
		expect(loaded.meta.passedGate).toBe(true);
	});

	it("refuses a model whose features differ from what the pipeline computes", () => {
		saveLtrModel(dir, meta("old", 1_000, true), stumpModelJson(["a", "b"], "a", 0, 0, 1));
		expect(() => loadLtrModel(dir, "old")).toThrow(/features/u);
	});

	it("rejects ids that could escape the model directory", () => {
		expect(() => saveLtrModel(dir, meta("../evil", 1, true), MODEL)).toThrow(/invalid model id/u);
		expect(() => loadLtrModel(dir, "../evil")).toThrow(/invalid model id/u);
	});

	it("resolves `latest` to the newest model that passed the gate, and an exact id to itself", () => {
		saveLtrModel(dir, meta("m-1", 1_000, true), MODEL);
		saveLtrModel(dir, meta("m-2", 2_000, false), MODEL);
		expect(resolveLtrModel(dir, "latest")?.id).toBe("m-1");
		expect(resolveLtrModel(dir, undefined)?.id).toBe("m-1");
		expect(resolveLtrModel(dir, "m-2")?.id).toBe("m-2");
		expect(resolveLtrModel(dir, "missing")).toBeUndefined();
	});

	it("resolves to nothing when no model passed the gate", () => {
		saveLtrModel(dir, meta("m-2", 2_000, false), MODEL);
		expect(resolveLtrModel(dir, "latest")).toBeUndefined();
	});

	it("deletes a model and survives a corrupt entry when listing", () => {
		saveLtrModel(dir, meta("m-1", 1_000, true), MODEL);
		saveLtrModel(dir, meta("m-2", 2_000, true), MODEL);
		writeFileSync(join(dir, "models", "m-2", "meta.json"), "{broken");
		expect(listLtrModels(dir).map((entry) => entry.id)).toEqual(["m-1"]);
		deleteLtrModel(dir, "m-1");
		expect(listLtrModels(dir)).toHaveLength(0);
		expect(() => deleteLtrModel(dir, "m-1")).toThrow(/not found/u);
	});

	it("accepts `latest` and plain ids as a selection, nothing path-like", () => {
		for (const ok of ["latest", "xgb-20261119-ab12cd", "m_1.2"]) expect(isLtrModelSelection(ok)).toBe(true);
		for (const bad of ["", "../x", "a/b", ".hidden", "x".repeat(65)]) expect(isLtrModelSelection(bad)).toBe(false);
	});

	it("persists retrain bookkeeping and reads an empty default", () => {
		expect(readLtrState(dir)).toEqual({});
		writeLtrState(dir, { lastAttemptRuns: 200, lastAttemptAt: 5 });
		expect(readLtrState(dir)).toEqual({ lastAttemptRuns: 200, lastAttemptAt: 5 });
		expect(JSON.parse(readFileSync(join(dir, "state.json"), "utf-8")).lastAttemptRuns).toBe(200);
	});
});
