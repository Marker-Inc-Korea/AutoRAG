import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { TSchema } from "typebox";
import { describe, expect, it } from "vitest";
import {
	createManageLtrTool,
	type LtrToolHost,
	MANAGE_LTR_TOOL_NAME,
	type ManageLtrDetails,
} from "../../src/agent/ltr-tool.ts";
import type { LtrSettings } from "../../src/ltr/manager.ts";
import type { TrainResult } from "../../src/ltr/trainer.ts";

function host(overrides: Partial<LtrToolHost> = {}) {
	let settings: LtrSettings = {};
	const calls: string[] = [];
	const value: LtrToolHost = {
		memoryPath: "/nonexistent/memory.json",
		settings: () => settings,
		update: (patch) => {
			calls.push(JSON.stringify(patch));
			settings = { ...settings, ...patch, ...(patch.model === "latest" ? { model: undefined } : {}) };
			return settings;
		},
		train: async (): Promise<TrainResult> => ({
			status: "insufficient-data",
			reason: "0 usable runs",
			eligibleRuns: 0,
			loggedRuns: 0,
		}),
		...overrides,
	};
	return { value, calls };
}

async function call(tool: AgentTool<TSchema, ManageLtrDetails>, params: Record<string, unknown>) {
	const result = await tool.execute("id", params as never);
	return result.content.map((part) => (part.type === "text" ? part.text : "")).join("\n");
}

describe("manage_ltr tool", () => {
	it("is named manage_ltr", () => {
		expect(createManageLtrTool(host().value).name).toBe(MANAGE_LTR_TOOL_NAME);
		expect(MANAGE_LTR_TOOL_NAME).toBe("manage_ltr");
	});

	it("reports status without side effects", async () => {
		const { value, calls } = host();
		const text = await call(createManageLtrTool(value), { action: "status" });
		expect(text).toContain("disabled");
		expect(text).toContain("Logged search runs: 0");
		expect(calls).toEqual([]);
	});

	it("turns serving on and off, selects a model and configures auto-train through the host", async () => {
		const { value, calls } = host();
		const tool = createManageLtrTool(value);
		expect(await call(tool, { action: "set", enabled: true })).toContain("enabled");
		expect(await call(tool, { action: "set", model: "xgb-1" })).toContain("xgb-1");
		await call(tool, { action: "set", auto_train: true, min_new_runs: 300 });
		expect(calls).toEqual([
			JSON.stringify({ enabled: true }),
			JSON.stringify({ model: "xgb-1" }),
			JSON.stringify({ autoTrain: { enabled: true, minNewRuns: 300 } }),
		]);
	});

	it("refuses a set with nothing to change and an invalid model id", async () => {
		const tool = createManageLtrTool(host().value);
		await expect(call(tool, { action: "set" })).rejects.toThrow(/nothing to change/u);
		await expect(call(tool, { action: "set", model: "../x" })).rejects.toThrow(/model/u);
	});

	it("explains that settings cannot be saved when the host has no config file", async () => {
		const { value } = host({
			update: () => {
				throw new Error("this session has no config file to save settings to");
			},
		});
		await expect(call(createManageLtrTool(value), { action: "set", enabled: true })).rejects.toThrow(
			/no config file/u,
		);
	});

	it("runs training and reports the verdict", async () => {
		const { value } = host({
			train: async () => ({
				status: "trained",
				reason: "holdout 30 queries: MRR 0.500 -> 0.600, nDCG@10 0.600 -> 0.650",
				eligibleRuns: 150,
				loggedRuns: 160,
				model: {
					id: "xgb-1",
					createdAt: 1,
					featureNames: [],
					objective: "rank:pairwise",
					trainedOnRuns: 120,
					loggedRunsAtTraining: 160,
					holdout: {
						queries: 30,
						baseline: { ndcg10: 0.6, mrr: 0.5 },
						model: { ndcg10: 0.65, mrr: 0.6 },
						deltaMrrCi95: [0.01, 0.2],
					},
					passedGate: true,
				},
			}),
		});
		const text = await call(createManageLtrTool(value), { action: "train" });
		expect(text).toContain("trained");
		expect(text).toContain("xgb-1");
		expect(text).toContain("MRR 0.500 -> 0.600");
	});
});
