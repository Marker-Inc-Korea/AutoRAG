import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { LTR_SERVING_FEATURE_NAMES } from "../memory/ltr-export.ts";
import { parseXgboostModel, type XgboostModel } from "./xgboost-model.ts";

/**
 * Trained learning-to-rank models, kept beside the memory file in `<memoryPath>.ltr/`:
 *
 *   models/<id>/model.json   the XGBoost model (served by {@link parseXgboostModel})
 *   models/<id>/meta.json    what it was trained on and how it did on the held-out newest runs
 *   state.json               retrain bookkeeping (see {@link LtrState})
 *
 * Nothing here is derived data: a model is the product of a training run on the user's own log, so a file that
 * cannot be read is reported and skipped, never silently replaced.
 */

/** Ranking quality on the holdout runs: the model against the baseline order it would replace. */
export interface LtrHoldoutMetrics {
	readonly queries: number;
	readonly baseline: { readonly ndcg10: number; readonly mrr: number };
	readonly model: { readonly ndcg10: number; readonly mrr: number };
	/** 95% bootstrap interval of the per-query MRR difference (model minus baseline). */
	readonly deltaMrrCi95: readonly [number, number];
}

export interface LtrModelMeta {
	readonly id: string;
	readonly createdAt: number;
	readonly featureNames: readonly string[];
	readonly objective: string;
	/** Labelled runs the model was fitted on (train + validation). */
	readonly trainedOnRuns: number;
	/** Runs in the log when training started. */
	readonly loggedRunsAtTraining: number;
	readonly holdout: LtrHoldoutMetrics;
	/** True when the model beat the baseline on the holdout by the training gate; only these are picked by `latest`. */
	readonly passedGate: boolean;
	readonly trainer?: { readonly xgboost: string };
}

export interface LtrState {
	/** Log size at the last training attempt, so a failed or rejected attempt is not retried every search. */
	readonly lastAttemptRuns?: number;
	readonly lastAttemptAt?: number;
	/** The attempt's outcome, for `autorag ltr status`. */
	readonly lastAttemptResult?: string;
}

export interface LoadedLtrModel {
	readonly meta: LtrModelMeta;
	readonly model: XgboostModel;
}

const MODEL_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/u;

/** Directory holding the models of the memory file at `memoryPath`. */
export function ltrDirectory(memoryPath: string): string {
	return `${memoryPath}.ltr`;
}

function assertModelId(id: string): void {
	if (!MODEL_ID_PATTERN.test(id) || id.includes("..")) throw new Error(`invalid model id: ${JSON.stringify(id)}`);
}

/** True for `latest` or a plain model id: the only values a config or the agent may select. */
export function isLtrModelSelection(value: string): boolean {
	return value === "latest" || (MODEL_ID_PATTERN.test(value) && !value.includes(".."));
}

function modelDirectory(directory: string, id: string): string {
	assertModelId(id);
	return join(directory, "models", id);
}

function writeAtomically(path: string, contents: string): void {
	const temporary = `${path}.${process.pid}.tmp`;
	writeFileSync(temporary, contents, "utf-8");
	renameSync(temporary, path);
}

export function saveLtrModel(directory: string, meta: LtrModelMeta, modelJson: string): void {
	const target = modelDirectory(directory, meta.id);
	mkdirSync(target, { recursive: true });
	writeAtomically(join(target, "model.json"), modelJson);
	writeAtomically(join(target, "meta.json"), `${JSON.stringify(meta, null, 2)}\n`);
}

function readMeta(directory: string, id: string): LtrModelMeta {
	const parsed: unknown = JSON.parse(readFileSync(join(modelDirectory(directory, id), "meta.json"), "utf-8"));
	const meta = parsed as Partial<LtrModelMeta>;
	if (
		typeof meta.id !== "string" ||
		meta.id !== id ||
		typeof meta.createdAt !== "number" ||
		!Array.isArray(meta.featureNames) ||
		typeof meta.passedGate !== "boolean" ||
		typeof meta.holdout !== "object"
	) {
		throw new Error(`model ${id} has an unreadable meta.json`);
	}
	return meta as LtrModelMeta;
}

/** Every readable model, newest first. An unreadable entry is skipped. */
export function listLtrModels(directory: string): LtrModelMeta[] {
	const root = join(directory, "models");
	if (!existsSync(root)) return [];
	const models: LtrModelMeta[] = [];
	for (const entry of readdirSync(root, { withFileTypes: true })) {
		if (!entry.isDirectory()) continue;
		try {
			models.push(readMeta(directory, entry.name));
		} catch {
			// A half-written or hand-edited entry never blocks the others.
		}
	}
	return models.sort((a, b) => b.createdAt - a.createdAt || b.id.localeCompare(a.id));
}

/**
 * Load a model for serving. The model's feature list must equal the one the pipeline computes: a model trained
 * against another feature layout would score the wrong columns, so it is refused with that reason.
 */
export function loadLtrModel(directory: string, id: string): LoadedLtrModel {
	const meta = readMeta(directory, id);
	const model = parseXgboostModel(readFileSync(join(modelDirectory(directory, id), "model.json"), "utf-8"));
	const expected = [...LTR_SERVING_FEATURE_NAMES];
	const same =
		model.featureNames.length === expected.length &&
		model.featureNames.every((name, index) => name === expected[index]);
	if (!same) {
		throw new Error(
			`model ${id} was trained on different features (${model.featureNames.join(", ")}) than this AutoRAG computes (${expected.join(", ")}); retrain it`,
		);
	}
	return { meta, model };
}

/**
 * The model to serve for a configured selection. `undefined` or `latest` is the newest model that passed the
 * training gate; an explicit id is that model whether or not it passed (the operator chose it). `undefined` is
 * returned when nothing matches.
 */
export function resolveLtrModel(directory: string, selection: string | undefined): LtrModelMeta | undefined {
	const models = listLtrModels(directory);
	if (selection === undefined || selection === "latest") return models.find((entry) => entry.passedGate);
	return models.find((entry) => entry.id === selection);
}

export function deleteLtrModel(directory: string, id: string): void {
	const target = modelDirectory(directory, id);
	if (!existsSync(target)) throw new Error(`model ${id} not found`);
	rmSync(target, { recursive: true, force: true });
}

export function readLtrState(directory: string): LtrState {
	const path = join(directory, "state.json");
	if (!existsSync(path)) return {};
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf-8"));
		return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed) ? (parsed as LtrState) : {};
	} catch {
		return {};
	}
}

export function writeLtrState(directory: string, state: LtrState): void {
	mkdirSync(directory, { recursive: true });
	writeAtomically(join(directory, "state.json"), `${JSON.stringify(state, null, 2)}\n`);
}
