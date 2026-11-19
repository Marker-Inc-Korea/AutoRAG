import { DMatrix, XGBoost } from "@bubblyworld/xgboost-ts";

/**
 * Trains the learning-to-rank model with XGBoost compiled to WebAssembly (`@bubblyworld/xgboost-ts`): no Python, no
 * native addon, nothing to install at run time. The ranker is `rank:pairwise` with early stopping on the validation
 * queries' nDCG@10, and the result is the same XGBoost JSON model `parseXgboostModel` serves.
 *
 * The package does not document ranking groups. It exposes the C API's `XGDMatrixSetFloatInfo`, and XGBoost accepts a
 * per-row `qid` through it, which is how a row is assigned to its query here. `test/ltr/wasm-trainer.test.ts` pins
 * this against numbers produced by the Python xgboost library, so an upgrade that changes the behavior fails there.
 */

/** One split of the data: feature rows (NaN = missing), a graded label per row, and the rows per query in row order. */
export interface RankingMatrix {
	readonly rows: readonly (readonly number[])[];
	readonly labels: readonly number[];
	readonly groups: readonly number[];
}

export interface TrainRankerInput {
	readonly train: RankingMatrix;
	readonly valid: RankingMatrix;
	readonly featureNames: readonly string[];
	readonly seed: number;
}

export interface TrainedRanker {
	/** XGBoost JSON model holding exactly the first {@link rounds} trees. */
	readonly modelJson: string;
	/** Boosting rounds kept: the best validation round. */
	readonly rounds: number;
	/** Rounds actually run, including the ones that waited for early stopping to give up. */
	readonly roundsRun: number;
	/** Validation nDCG@10 at the best round. */
	readonly validNdcg10: number;
	/** xgboost version compiled into the WASM binary. */
	readonly xgboostVersion: string;
}

const MAX_ROUNDS = 600;
const EARLY_STOPPING_ROUNDS = 50;

function trainParams(seed: number): Readonly<Record<string, string>> {
	return {
		objective: "rank:pairwise",
		eval_metric: "ndcg@10",
		eta: "0.05",
		max_depth: "4",
		min_child_weight: "5",
		subsample: "0.8",
		colsample_bytree: "0.8",
		lambda: "1",
		seed: String(seed),
		tree_method: "hist",
		// One thread: the result must not depend on how many cores the machine has.
		nthread: "1",
	};
}

function check(name: string, data: RankingMatrix, featureCount: number): void {
	if (data.rows.length === 0) throw new Error(`no ${name} rows to train on`);
	if (data.labels.length !== data.rows.length)
		throw new Error(`${name} has ${data.labels.length} labels for ${data.rows.length} rows`);
	if (data.groups.reduce((sum, size) => sum + size, 0) !== data.rows.length) {
		throw new Error(`${name} groups do not add up to its ${data.rows.length} rows`);
	}
	if (data.rows.some((row) => row.length !== featureCount)) {
		throw new Error(`${name} rows do not all have ${featureCount} features`);
	}
}

function toMatrix(xgb: XGBoost, data: RankingMatrix, featureNames: readonly string[]): DMatrix {
	const matrix = DMatrix.create(xgb, {
		data: data.rows.map((row) => [...row]),
		label: data.labels.slice(),
		featureNames: [...featureNames],
	});
	const qid = new Float32Array(data.rows.length);
	let at = 0;
	for (const [query, size] of data.groups.entries()) {
		qid.fill(query, at, at + size);
		at += size;
	}
	xgb.setDMatrixFloatInfo(matrix.handle, "qid", qid);
	return matrix;
}

function validNdcg10(report: string): number {
	const value = Number(/ndcg@10:([0-9.eE+-]+)/u.exec(report)?.[1]);
	if (!Number.isFinite(value)) throw new Error(`XGBoost reported no nDCG@10: ${report}`);
	return value;
}

/** Fit a pairwise ranker with early stopping and return it as a servable XGBoost JSON model. */
export async function trainRanker(input: TrainRankerInput): Promise<TrainedRanker> {
	const { train, valid, featureNames, seed } = input;
	check("training", train, featureNames.length);
	check("validation", valid, featureNames.length);

	// An isolated module per run: boosters and matrices are freed below, and the instance itself is reclaimed with it.
	const xgb = await XGBoost.create();
	const matrices: DMatrix[] = [];
	const boosters: number[] = [];
	try {
		const dtrain = toMatrix(xgb, train, featureNames);
		matrices.push(dtrain);
		const dvalid = toMatrix(xgb, valid, featureNames);
		matrices.push(dvalid);
		const newBooster = (cache: readonly DMatrix[]): number => {
			const booster = xgb.createBooster(cache.map((matrix) => matrix.handle));
			boosters.push(booster);
			for (const [name, value] of Object.entries(trainParams(seed))) xgb.setParam(booster, name, value);
			return booster;
		};

		const searching = newBooster([dtrain, dvalid]);
		let best = Number.NEGATIVE_INFINITY;
		let bestRound = 0;
		let waited = 0;
		let roundsRun = 0;
		for (let round = 0; round < MAX_ROUNDS; round += 1) {
			xgb.updateOneIter(searching, round, dtrain.handle);
			roundsRun = round + 1;
			const score = validNdcg10(xgb.evalOneIter(searching, round, [dvalid.handle], ["valid"]));
			if (score > best) {
				best = score;
				bestRound = round + 1;
				waited = 0;
			} else {
				waited += 1;
				if (waited >= EARLY_STOPPING_ROUNDS) break;
			}
		}

		// The same seed grows the same trees, so growing exactly `bestRound` of them reproduces the early-stopped model
		// without the trailing rounds that only waited for validation to stop improving.
		const kept = newBooster([dtrain]);
		for (let round = 0; round < bestRound; round += 1) xgb.updateOneIter(kept, round, dtrain.handle);
		xgb.setBoosterStrFeatureInfo(kept, "feature_name", [...featureNames]);
		const modelJson = new TextDecoder().decode(xgb.saveModelToBuffer(kept, "json"));
		const { major, minor, patch } = xgb.version();
		return {
			modelJson,
			rounds: bestRound,
			roundsRun,
			validNdcg10: best,
			xgboostVersion: `${major}.${minor}.${patch}`,
		};
	} finally {
		for (const booster of boosters) xgb.freeBooster(booster);
		for (const matrix of matrices) matrix.dispose();
		xgb.free();
	}
}

interface JsonTree {
	left_children: number[];
	split_conditions: number[];
	base_weights: number[];
	id: number;
	[field: string]: unknown;
}

function asRecord(value: unknown, what: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value))
		throw new Error(`XGBoost model has no ${what}`);
	return value as Record<string, unknown>;
}

/**
 * Merge several XGBoost JSON models of the same features into one whose margin is the mean of theirs. Leaves are
 * scaled by 1/k and the trees concatenated, so a seed ensemble is still one ordinary model that
 * `parseXgboostModel` serves; nothing outside this file knows it is an ensemble.
 */
export function averageRankers(modelJsons: readonly string[]): string {
	const [first] = modelJsons;
	if (first === undefined) throw new Error("no models to average");
	const parsed = modelJsons.map((json) => JSON.parse(json) as unknown);
	const learners = parsed.map((model, index) =>
		asRecord(asRecord(model, `learner in model ${index}`).learner, `learner in model ${index}`),
	);
	const names = JSON.stringify(learners[0]?.feature_names);
	for (const [index, learner] of learners.entries()) {
		if (JSON.stringify(learner.feature_names) !== names)
			throw new Error(`model ${index} was trained on different features`);
	}
	const scale = 1 / modelJsons.length;
	const merged: JsonTree[] = [];
	let baseScore = 0;
	for (const learner of learners) {
		const param = asRecord(learner.learner_model_param, "learner_model_param");
		baseScore += Number(String(param.base_score).replace(/[[\]]/gu, "")) * scale;
		const model = asRecord(asRecord(asRecord(learner.gradient_booster, "gradient_booster").model, "model"), "model");
		for (const tree of model.trees as JsonTree[]) {
			const leafValue = (values: number[]): number[] =>
				values.map((value, node) => (tree.left_children[node] === -1 ? value * scale : value));
			merged.push({
				...tree,
				id: merged.length,
				split_conditions: leafValue(tree.split_conditions),
				base_weights: tree.base_weights.map((value) => value * scale),
			});
		}
	}
	const out = structuredClone(parsed[0]) as Record<string, unknown>;
	const outLearner = asRecord(out.learner, "learner");
	const booster = asRecord(outLearner.gradient_booster, "gradient_booster");
	const model = asRecord(booster.model, "model");
	model.trees = merged;
	model.tree_info = merged.map(() => 0);
	model.iteration_indptr = Array.from({ length: merged.length + 1 }, (_, index) => index);
	model.gbtree_model_param = {
		...asRecord(model.gbtree_model_param, "gbtree_model_param"),
		num_trees: String(merged.length),
	};
	outLearner.learner_model_param = {
		...asRecord(outLearner.learner_model_param, "learner_model_param"),
		base_score: `[${baseScore.toExponential()}]`,
	};
	return JSON.stringify(out);
}
