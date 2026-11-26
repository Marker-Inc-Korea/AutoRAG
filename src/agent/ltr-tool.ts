import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { getLtrStatus, type LtrPatch, type LtrSettings, type LtrStatus } from "../ltr/manager.ts";
import { isLtrModelSelection, type LtrModelMeta } from "../ltr/model-store.ts";
import type { TrainResult } from "../ltr/trainer.ts";

export const MANAGE_LTR_TOOL_NAME = "manage_ltr";

const manageLtrSchema = Type.Object({
	action: Type.Union([Type.Literal("status"), Type.Literal("train"), Type.Literal("set")], {
		description:
			"status: show whether learning-to-rank is on, which model would serve, the models, and whether retraining is due. train: train a new model now from the logged searches. set: change settings (enabled, model, auto_train, min_new_runs).",
	}),
	enabled: Type.Optional(
		Type.Boolean({ description: "set: reorder search results with the selected model (true) or stop (false)." }),
	),
	model: Type.Optional(
		Type.String({
			description:
				"set: a model id from status, or `latest` for the newest model that passed its holdout check (the default).",
		}),
	),
	auto_train: Type.Optional(
		Type.Boolean({ description: "set: retrain automatically once enough new searches were logged." }),
	),
	min_new_runs: Type.Optional(
		Type.Integer({ minimum: 1, description: "set: new logged searches between automatic retrains. Default 200." }),
	),
});

/** What the tool needs from the agent's host: where state lives, how to save settings, how to train. */
export interface LtrToolHost {
	readonly memoryPath: string;
	/** The `ltr` settings in effect. */
	settings(): LtrSettings | undefined;
	/** Persist a settings change and return the new `ltr` settings; throws when the session has no config file. */
	update(patch: LtrPatch): LtrSettings;
	train(): Promise<TrainResult>;
}

export interface ManageLtrDetails {
	readonly action: string;
}

function describeStatus(status: LtrStatus): string {
	const serving =
		status.serving.status === "active"
			? `serving model ${status.serving.modelId}`
			: status.serving.status === "disabled"
				? "off"
				: `enabled but not serving: ${status.serving.reason}`;
	return [
		`Learning-to-rank: ${status.enabled ? "enabled" : "disabled"} (${serving}).`,
		`Model selection: ${status.selection}.`,
		`Logged search runs: ${status.loggedRuns}.`,
		`Auto-train: ${status.autoTrain.enabled ? `on, every ${status.autoTrain.minNewRuns} new runs${status.autoTrain.due ? " (retrain is due)" : ""}` : "off"}.`,
		...(status.state.lastAttemptResult !== undefined
			? [`Last training attempt: ${status.state.lastAttemptResult}.`]
			: []),
		...(status.models.length === 0
			? ["No trained models."]
			: ["Models (newest first):", ...status.models.map((model) => `- ${describeModel(model)}`)]),
	].join("\n");
}

function describeModel(model: LtrModelMeta): string {
	const { holdout } = model;
	return `${model.id}: ${model.passedGate ? "passed" : "rejected"} its holdout check (${holdout.queries} queries, MRR ${holdout.baseline.mrr.toFixed(3)} -> ${holdout.model.mrr.toFixed(3)}), trained on ${model.trainedOnRuns} runs`;
}

function text(value: string, details: ManageLtrDetails): AgentToolResult<ManageLtrDetails> {
	return { content: [{ type: "text", text: value }], details };
}

export function createManageLtrTool(host: LtrToolHost): AgentTool<typeof manageLtrSchema, ManageLtrDetails> {
	return {
		name: MANAGE_LTR_TOOL_NAME,
		label: "Manage Learning-to-Rank",
		description:
			"Manage AutoRAG's opt-in learning-to-rank reordering of search results: check status, train from logged searches, enable/disable, pick a model, or configure auto-retraining. It only reorders reranker output (never adds/drops evidence). Never enable, switch models, or train without being asked, and never recommend serving a model marked rejected.",
		parameters: manageLtrSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<ManageLtrDetails>> {
			const { action } = params;
			switch (action) {
				case "status":
					return text(describeStatus(getLtrStatus({ memoryPath: host.memoryPath, config: host.settings() })), {
						action,
					});
				case "train": {
					const result = await host.train();
					const lines = [`Training ${result.status}${result.reason !== undefined ? `: ${result.reason}` : ""}.`];
					if (result.model !== undefined) lines.push(describeModel(result.model));
					if (result.status === "trained") {
						lines.push(
							"It serves once learning-to-rank is enabled (set enabled=true) and it is the selected or latest model.",
						);
					}
					return text(lines.join("\n"), { action });
				}
				case "set": {
					const autoTrain =
						params.auto_train !== undefined || params.min_new_runs !== undefined
							? {
									...(params.auto_train !== undefined ? { enabled: params.auto_train } : {}),
									...(params.min_new_runs !== undefined ? { minNewRuns: params.min_new_runs } : {}),
								}
							: undefined;
					const patch: LtrPatch = {
						...(params.enabled !== undefined ? { enabled: params.enabled } : {}),
						...(params.model !== undefined ? { model: params.model } : {}),
						...(autoTrain !== undefined ? { autoTrain } : {}),
					};
					if (patch.model !== undefined && !isLtrModelSelection(patch.model)) {
						throw new Error(
							`model must be \`latest\` or a model id from status, got ${JSON.stringify(patch.model)}`,
						);
					}
					if (Object.keys(patch).length === 0) {
						throw new Error(
							"manage_ltr set has nothing to change: pass enabled, model, auto_train, or min_new_runs",
						);
					}
					const next = host.update(patch);
					return text(
						`Saved. Learning-to-rank settings: ${JSON.stringify(next)}. They apply from the next search.`,
						{ action },
					);
				}
			}
		},
	};
}
