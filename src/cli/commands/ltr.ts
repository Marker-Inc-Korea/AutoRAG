import { applyLtrPatch, getLtrStatus, type LtrPatch, type LtrSettings, type LtrStatus } from "../../ltr/manager.ts";
import { deleteLtrModel, type LtrModelMeta, listLtrModels, ltrDirectory } from "../../ltr/model-store.ts";
import { trainLtrModel } from "../../ltr/trainer.ts";
import { normalizeLtrConfig, readRawConfigObject, resolveConfig, writeConfigObject } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

const USAGE =
	"Usage: autorag ltr <status|list|train|enable|disable|use <model|latest>|auto-train <on|off> [--min-new-runs <n>]|delete <model>>";

/**
 * `autorag ltr` — manage the opt-in learning-to-rank reorder of search results: train a model from the ranking log,
 * pick which model serves, and turn serving and automatic retraining on or off. The `ltr` block of the active config
 * file is the only thing a setting subcommand writes, and only the fields it changes.
 */
export async function runLtr(ctx: CommandContext): Promise<number> {
	const [subcommand, argument] = ctx.positionals;
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd, readOnly: true });
		const emit = (value: unknown, human: string): void => ctx.stdout(ctx.json ? JSON.stringify(value) : human);
		const directory = ltrDirectory(config.memoryPath);

		switch (subcommand) {
			case "status": {
				const status = getLtrStatus({ memoryPath: config.memoryPath, config: config.ltr });
				emit(status, formatStatus(status));
				return 0;
			}
			case "list": {
				const models = listLtrModels(directory);
				emit({ models }, models.length === 0 ? "No trained models." : models.map(formatModel).join("\n"));
				return 0;
			}
			case "train": {
				const result = await trainLtrModel({ memoryPath: config.memoryPath });
				const summary = {
					status: result.status,
					...(result.reason !== undefined ? { reason: result.reason } : {}),
					...(result.model !== undefined ? { model: result.model } : {}),
					eligibleRuns: result.eligibleRuns,
					loggedRuns: result.loggedRuns,
				};
				emit(
					summary,
					`${result.status}: ${result.reason ?? ""}${result.model ? `\nmodel ${result.model.id}` : ""}`.trim(),
				);
				return result.status === "trained" ? 0 : 1;
			}
			case "enable":
			case "disable":
				return patchConfig(ctx, config.configPath, { enabled: subcommand === "enable" });
			case "use": {
				if (argument === undefined) return usageError(ctx, "use requires a model id or `latest`");
				if (argument !== "latest" && !listLtrModels(directory).some((model) => model.id === argument)) {
					throw new Error(`model ${argument} does not exist; run \`autorag ltr list\``);
				}
				return patchConfig(ctx, config.configPath, { model: argument });
			}
			case "auto-train": {
				if (argument !== "on" && argument !== "off") return usageError(ctx, "auto-train requires `on` or `off`");
				const raw = ctx.flags["min-new-runs"];
				const minNewRuns = typeof raw === "string" ? Number(raw) : undefined;
				return patchConfig(ctx, config.configPath, {
					autoTrain: { enabled: argument === "on", ...(minNewRuns !== undefined ? { minNewRuns } : {}) },
				});
			}
			case "delete": {
				if (argument === undefined) return usageError(ctx, "delete requires a model id");
				if (config.ltr?.model === argument) {
					throw new Error(`model ${argument} is pinned in the config; run \`autorag ltr use latest\` first`);
				}
				deleteLtrModel(directory, argument);
				emit({ deleted: argument }, `Deleted model ${argument}.`);
				return 0;
			}
			default:
				return usageError(ctx, USAGE);
		}
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

function usageError(ctx: CommandContext, message: string): number {
	ctx.stderr(
		renderError(new Error(message === USAGE ? USAGE : `${message}. ${USAGE}`), { json: ctx.json, debug: ctx.debug }),
	);
	return 2;
}

/** Apply `patch` to the `ltr` block of the config file on disk, leaving every other field as the user wrote it. */
function patchConfig(ctx: CommandContext, configPath: string | undefined, patch: LtrPatch): number {
	if (configPath === undefined) throw new Error("no config file to update");
	const raw = readRawConfigObject(configPath);
	const current = raw.ltr === undefined ? undefined : (normalizeLtrConfig(raw.ltr, "ltr") as LtrSettings);
	const next = applyLtrPatch(current, patch);
	writeConfigObject(configPath, { ...raw, ltr: next });
	ctx.stdout(ctx.json ? JSON.stringify({ ltr: next }) : `ltr: ${JSON.stringify(next)}`);
	return 0;
}

function formatModel(model: LtrModelMeta): string {
	const { holdout } = model;
	return `${model.id}  ${model.passedGate ? "passed" : "rejected"}  holdout ${holdout.queries} queries  MRR ${holdout.baseline.mrr.toFixed(3)} -> ${holdout.model.mrr.toFixed(3)}  trained on ${model.trainedOnRuns} runs`;
}

function formatStatus(status: LtrStatus): string {
	const serving =
		status.serving.status === "active"
			? `serving model ${status.serving.modelId}`
			: status.serving.status === "disabled"
				? "off"
				: `not serving: ${status.serving.reason}`;
	return [
		`ltr: ${status.enabled ? "enabled" : "disabled"} (${serving})`,
		`model selection: ${status.selection}`,
		`logged search runs: ${status.loggedRuns}`,
		`models: ${status.models.length}`,
		`auto-train: ${status.autoTrain.enabled ? `on, every ${status.autoTrain.minNewRuns} new runs${status.autoTrain.due ? " (retrain due)" : ""}` : "off"}`,
	].join("\n");
}
