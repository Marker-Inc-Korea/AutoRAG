import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { acquireFileLock, type FileLockHandle } from "../filesystem/file-lock.ts";
import { countLoggedRuns, rankingLogPath } from "../memory/ranking-log.ts";
import {
	isLtrModelSelection,
	type LtrModelMeta,
	type LtrState,
	listLtrModels,
	ltrDirectory,
	readLtrState,
} from "./model-store.ts";
import { createLtrScorer } from "./scorer.ts";
import { shouldRetrain, type TrainResult, trainLtrModel } from "./trainer.ts";

/** New logged runs since the last attempt before auto-training runs again, when the config does not say. */
export const DEFAULT_AUTO_TRAIN_MIN_NEW_RUNS = 200;

/** The `ltr` config section as the management layer reads it. */
export interface LtrSettings {
	readonly enabled?: boolean;
	readonly model?: string;
	readonly autoTrain?: { readonly enabled?: boolean; readonly minNewRuns?: number };
}

export interface LtrStatus {
	readonly enabled: boolean;
	/** The configured model selection: an id or `latest`. */
	readonly selection: string;
	/** What a search would do right now. */
	readonly serving:
		| { readonly status: "disabled" }
		| { readonly status: "active"; readonly modelId: string }
		| { readonly status: "no-model"; readonly reason: string }
		| { readonly status: "unusable"; readonly modelId: string; readonly reason: string };
	readonly loggedRuns: number;
	/** Every readable model, newest first. */
	readonly models: readonly LtrModelMeta[];
	readonly state: LtrState;
	readonly autoTrain: { readonly enabled: boolean; readonly minNewRuns: number; readonly due: boolean };
}

/** Everything `autorag ltr status` and the agent's `manage_ltr` report; read-only and never creates a file. */
export function getLtrStatus(input: {
	readonly memoryPath: string;
	readonly config: LtrSettings | undefined;
}): LtrStatus {
	const { memoryPath, config } = input;
	const directory = ltrDirectory(memoryPath);
	const enabled = config?.enabled === true;
	const scorer = createLtrScorer({
		enabled,
		...(config?.model !== undefined ? { model: config.model } : {}),
		memoryPath,
	});
	const serving: LtrStatus["serving"] =
		scorer.status === "active" ? { status: "active", modelId: scorer.modelId } : scorer;
	const loggedRuns = countLoggedRuns(rankingLogPath(memoryPath));
	const state = readLtrState(directory);
	const autoEnabled = config?.autoTrain?.enabled === true;
	const minNewRuns = config?.autoTrain?.minNewRuns ?? DEFAULT_AUTO_TRAIN_MIN_NEW_RUNS;
	return {
		enabled,
		selection: config?.model ?? "latest",
		serving,
		loggedRuns,
		models: listLtrModels(directory),
		state,
		autoTrain: {
			enabled: autoEnabled,
			minNewRuns,
			due: autoEnabled && shouldRetrain({ loggedRuns, lastAttemptRuns: state.lastAttemptRuns, minNewRuns }),
		},
	};
}

export interface LtrPatch {
	readonly enabled?: boolean;
	/** A model id, or `latest` to drop any pin. */
	readonly model?: string;
	readonly autoTrain?: { readonly enabled?: boolean; readonly minNewRuns?: number };
}

/**
 * The `ltr` config section after applying `patch`. Only the fields the patch names change; selecting `latest` removes
 * the model pin (that is the default). Validates what it is given, so a caller cannot write an invalid section.
 */
export function applyLtrPatch(current: LtrSettings | undefined, patch: LtrPatch): LtrSettings {
	let next: LtrSettings = { ...current };
	if (patch.enabled !== undefined) next = { ...next, enabled: patch.enabled };
	if (patch.model !== undefined) {
		if (!isLtrModelSelection(patch.model)) {
			throw new Error(
				`model must be \`latest\` or a model id from \`autorag ltr list\`, got ${JSON.stringify(patch.model)}`,
			);
		}
		const { model: _drop, ...withoutModel } = next;
		next = patch.model === "latest" ? withoutModel : { ...withoutModel, model: patch.model };
	}
	if (patch.autoTrain !== undefined) {
		const { minNewRuns } = patch.autoTrain;
		if (minNewRuns !== undefined && (!Number.isInteger(minNewRuns) || minNewRuns < 1)) {
			throw new Error(`autoTrain.minNewRuns must be a positive integer, got ${String(minNewRuns)}`);
		}
		next = { ...next, autoTrain: { ...next.autoTrain, ...patch.autoTrain } };
	}
	return next;
}

/** A training claim older than this is a crashed process's and may be taken over (training times out well before). */
export const AUTO_TRAIN_CLAIM_MS = 15 * 60 * 1000;

export interface AutoTrainOptions {
	readonly memoryPath: string;
	readonly settings: LtrSettings | undefined;
	readonly signal?: AbortSignal;
}

/**
 * Retrain when the agent's own settings ask for it and the log has grown enough since the last attempt. Returns
 * `undefined` when nothing was due or another process is already training; otherwise the training result. A claim
 * lock beside the models keeps several agent processes sharing one memory directory from training at once, and every
 * attempt (including a rejected or failed one) is recorded so the next search does not start another straight away.
 */
export async function runAutoTrain(options: AutoTrainOptions): Promise<TrainResult | undefined> {
	const { memoryPath, settings } = options;
	if (settings?.autoTrain?.enabled !== true) return undefined;
	const status = getLtrStatus({ memoryPath, config: settings });
	if (!status.autoTrain.due) return undefined;
	const directory = ltrDirectory(memoryPath);
	mkdirSync(directory, { recursive: true });
	let claim: FileLockHandle;
	try {
		claim = acquireFileLock(join(directory, "train.lock"), {
			timeoutMs: 0,
			staleMs: AUTO_TRAIN_CLAIM_MS,
			timeoutError: () => new Error("another process is training"),
		});
	} catch {
		return undefined;
	}
	try {
		return await trainLtrModel({
			memoryPath,
			...(options.signal !== undefined ? { signal: options.signal } : {}),
		});
	} finally {
		claim.release();
	}
}
