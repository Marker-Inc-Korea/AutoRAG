/**
 * Pure-TypeScript scorer for XGBoost ranking models saved as JSON (`Booster.save_model("x.json")`).
 *
 * Training runs XGBoost compiled to WebAssembly (`wasm-trainer.ts`); serving does not need it. A trained model is a few
 * hundred small decision trees that this file walks directly, so the search path stays in-process: no WASM, no native
 * addon, no network.
 *
 * Only what AutoRAG trains is accepted: the `gbtree` booster with a `rank:*` objective (the margin is the ranking
 * score, there is no link function) and numerical splits. Anything else is rejected at load time with the reason,
 * because a model that cannot be evaluated exactly must never be used to reorder results.
 */

const RANK_OBJECTIVES: Readonly<Record<string, true>> = { "rank:pairwise": true, "rank:ndcg": true, "rank:map": true };
const LEAF = -1;

interface Tree {
	readonly left: Int32Array;
	readonly right: Int32Array;
	readonly feature: Int32Array;
	/** Split threshold for internal nodes, leaf value for leaves. float32 like the library stores it. */
	readonly value: Float32Array;
	readonly defaultLeft: Uint8Array;
}

export interface XgboostModel {
	readonly featureNames: readonly string[];
	readonly treeCount: number;
	/** The ranking margin for one feature row; NaN marks a missing value. Higher ranks first. */
	predict(features: ArrayLike<number>): number;
}

function record(value: unknown, what: string): Record<string, unknown> {
	if (typeof value !== "object" || value === null || Array.isArray(value)) {
		throw new Error(`XGBoost model is missing ${what}`);
	}
	return value as Record<string, unknown>;
}

function numberArray(value: unknown, what: string): number[] {
	if (!Array.isArray(value) || value.some((entry) => typeof entry !== "number")) {
		throw new Error(`XGBoost model has an invalid ${what}`);
	}
	return value as number[];
}

function parseBaseScore(raw: unknown): number {
	// XGBoost 3 writes "[3.08E-9]"; 1.x/2.x write "0.5".
	const text = String(raw ?? "").replace(/[[\]]/gu, "");
	const value = Number(text);
	if (!Number.isFinite(value)) throw new Error(`XGBoost model has an invalid base_score: ${String(raw)}`);
	return value;
}

function parseTree(raw: unknown, index: number): Tree {
	const tree = record(raw, `tree ${index}`);
	const left = numberArray(tree.left_children, `left_children in tree ${index}`);
	const right = numberArray(tree.right_children, `right_children in tree ${index}`);
	const feature = numberArray(tree.split_indices, `split_indices in tree ${index}`);
	const value = numberArray(tree.split_conditions, `split_conditions in tree ${index}`);
	const defaultLeft = numberArray(tree.default_left, `default_left in tree ${index}`);
	const splitType = numberArray(tree.split_type, `split_type in tree ${index}`);
	const size = left.length;
	if ([right, feature, value, defaultLeft, splitType].some((column) => column.length !== size) || size === 0) {
		throw new Error(`XGBoost model tree ${index} has columns of different lengths`);
	}
	for (let node = 0; node < size; node += 1) {
		if (left[node] !== LEAF && splitType[node] !== 0) {
			throw new Error(`XGBoost model tree ${index} has a categorical split, which AutoRAG does not evaluate`);
		}
	}
	return {
		left: Int32Array.from(left),
		right: Int32Array.from(right),
		feature: Int32Array.from(feature),
		value: Float32Array.from(value),
		defaultLeft: Uint8Array.from(defaultLeft),
	};
}

/** Parse an XGBoost JSON model; throws with the reason when AutoRAG cannot evaluate it exactly. */
export function parseXgboostModel(json: string): XgboostModel {
	let parsed: unknown;
	try {
		parsed = JSON.parse(json);
	} catch {
		throw new Error("XGBoost model is not valid JSON");
	}
	const learner = record(record(parsed, "the learner").learner, "the learner");
	const booster = record(learner.gradient_booster, "gradient_booster");
	if (booster.name !== "gbtree") {
		throw new Error(`XGBoost booster "${String(booster.name)}" is not supported; only gbtree models can be served`);
	}
	const objective = String(record(learner.objective, "objective").name);
	if (!Object.hasOwn(RANK_OBJECTIVES, objective)) {
		throw new Error(
			`XGBoost objective "${objective}" is not a ranking objective (${Object.keys(RANK_OBJECTIVES).join(", ")})`,
		);
	}
	const modelParam = record(learner.learner_model_param, "learner_model_param");
	const featureCount = Number(modelParam.num_feature);
	if (!Number.isInteger(featureCount) || featureCount <= 0)
		throw new Error("XGBoost model has an invalid num_feature");
	const rawNames = learner.feature_names;
	const featureNames =
		Array.isArray(rawNames) && rawNames.length === featureCount
			? rawNames.map(String)
			: Array.from({ length: featureCount }, (_, index) => `f${index}`);
	const rawTrees = record(booster.model, "model").trees;
	if (!Array.isArray(rawTrees)) throw new Error("XGBoost model has no trees");
	const trees = rawTrees.map(parseTree);
	const baseScore = parseBaseScore(modelParam.base_score);

	return {
		featureNames,
		treeCount: trees.length,
		predict(features) {
			if (features.length !== featureCount) {
				throw new Error(`XGBoost model expects ${featureCount} features, got ${features.length}`);
			}
			let margin = baseScore;
			for (const tree of trees) {
				let node = 0;
				while (tree.left[node] !== LEAF) {
					const input = features[tree.feature[node] as number] as number;
					const goLeft = Number.isNaN(input)
						? tree.defaultLeft[node] === 1
						: Math.fround(input) < (tree.value[node] as number);
					node = (goLeft ? tree.left[node] : tree.right[node]) as number;
				}
				margin += tree.value[node] as number;
			}
			return margin;
		},
	};
}
