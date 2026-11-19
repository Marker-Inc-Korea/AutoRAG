import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runLtr } from "../../src/cli/commands/ltr.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { type LtrModelMeta, ltrDirectory, saveLtrModel } from "../../src/ltr/model-store.ts";
import { MIN_TRAINING_RUNS } from "../../src/ltr/trainer.ts";
import { LTR_SERVING_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { recordedRun, stumpModelJson } from "../ltr/helpers.ts";

let root: string;
let memoryPath: string;
let configPath: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-ltr-"));
	memoryPath = join(root, "memory.json");
	configPath = join(root, "config.json");
	writeConfig({});
});
afterEach(() => rmSync(root, { recursive: true, force: true }));

function writeConfig(extra: Record<string, unknown>): void {
	writeFileSync(
		configPath,
		`${JSON.stringify({ searchPaths: [root], workspacePath: root, memoryPath, ...extra }, null, 2)}\n`,
	);
}
function config(): Record<string, unknown> {
	return JSON.parse(readFileSync(configPath, "utf-8"));
}

async function run(positionals: string[], flags: CommandContext["flags"] = {}) {
	const out: string[] = [];
	const err: string[] = [];
	const code = await runLtr({
		positionals,
		flags: { config: configPath, ...flags },
		json: true,
		debug: false,
		cwd: root,
		stdout: (line) => out.push(line),
		stderr: (line) => err.push(line),
	});
	return { code, out: out.join("\n"), err: err.join("\n") };
}

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
const MODEL = stumpModelJson([...LTR_SERVING_FEATURE_NAMES], "dense_best_rank", 1.5, 5, 0);

describe("autorag ltr", () => {
	it("status on a fresh install reports everything off and creates nothing", async () => {
		const { code, out } = await run(["status"]);
		expect(code).toBe(0);
		const status = JSON.parse(out);
		expect(status).toMatchObject({
			enabled: false,
			selection: "latest",
			loggedRuns: 0,
			serving: { status: "disabled" },
		});
		expect(existsSync(ltrDirectory(memoryPath))).toBe(false);
		expect(existsSync(rankingLogPath(memoryPath))).toBe(false);
	});

	it("list shows models with their holdout verdict", async () => {
		saveLtrModel(ltrDirectory(memoryPath), meta("m-1", 1_000), MODEL);
		saveLtrModel(ltrDirectory(memoryPath), meta("m-2", 2_000, false), MODEL);
		const { out } = await run(["list"]);
		const models = JSON.parse(out).models as { id: string; passedGate: boolean }[];
		expect(models.map((model) => [model.id, model.passedGate])).toEqual([
			["m-2", false],
			["m-1", true],
		]);
	});

	it("enable and disable write only the ltr block and keep every other config field", async () => {
		writeConfig({ rerank: { model: "x" }, datasources: { kakao: { instanceId: "default" } } });
		expect((await run(["enable"])).code).toBe(0);
		expect(config().ltr).toEqual({ enabled: true });
		expect(config().rerank).toEqual({ model: "x" });
		expect(config().datasources).toEqual({ kakao: { instanceId: "default" } });
		expect((await run(["disable"])).code).toBe(0);
		expect(config().ltr).toEqual({ enabled: false });
	});

	it("use selects an existing model and `latest` clears the pin; an unknown id is refused", async () => {
		saveLtrModel(ltrDirectory(memoryPath), meta("m-1", 1_000), MODEL);
		expect((await run(["use", "m-1"])).code).toBe(0);
		expect(config().ltr).toEqual({ model: "m-1" });
		expect((await run(["use", "latest"])).code).toBe(0);
		expect(config().ltr).toEqual({});
		const missing = await run(["use", "nope"]);
		expect(missing.code).toBe(1);
		expect(missing.err).toContain("nope");
		expect(config().ltr).toEqual({});
	});

	it("auto-train toggles the retrain trigger and takes a threshold", async () => {
		expect((await run(["auto-train", "on"], { "min-new-runs": "300" })).code).toBe(0);
		expect(config().ltr).toEqual({ autoTrain: { enabled: true, minNewRuns: 300 } });
		expect((await run(["auto-train", "off"])).code).toBe(0);
		expect(config().ltr).toEqual({ autoTrain: { enabled: false, minNewRuns: 300 } });
		expect((await run(["auto-train", "maybe"])).code).toBe(2);
	});

	it("delete removes a model, and refuses to delete the one that is pinned in the config", async () => {
		saveLtrModel(ltrDirectory(memoryPath), meta("m-1", 1_000), MODEL);
		saveLtrModel(ltrDirectory(memoryPath), meta("m-2", 2_000), MODEL);
		await run(["use", "m-2"]);
		const pinned = await run(["delete", "m-2"]);
		expect(pinned.code).toBe(1);
		expect(pinned.err).toContain("pinned");
		expect((await run(["delete", "m-1"])).code).toBe(0);
		expect(JSON.parse((await run(["list"])).out).models).toHaveLength(1);
	});

	it("train reports insufficient data and exits non-zero without writing a model", async () => {
		const store = new RankingLogStore(rankingLogPath(memoryPath));
		for (let i = 0; i < 5; i += 1) store.append(recordedRun(`r-${i}`, 1_000 + i).finish(true));
		store.close();
		const { code, out } = await run(["train"]);
		expect(code).toBe(1);
		expect(JSON.parse(out)).toMatchObject({ status: "insufficient-data" });
	});

	it("train with a working trainer stores a model that status then shows as servable once enabled", async () => {
		const store = new RankingLogStore(rankingLogPath(memoryPath));
		for (let i = 0; i < MIN_TRAINING_RUNS + 20; i += 1) {
			store.append(recordedRun(`r-${String(i).padStart(3, "0")}`, 1_000 + i, { gold: 0 }).finish(true));
		}
		store.close();
		const python = join(root, "fake-python.sh");
		writeFileSync(python, `#!/bin/sh\ncat > "$4" <<'JSON'\n${MODEL}\nJSON\necho '{"xgboost":"fake"}'\n`);
		chmodSync(python, 0o755);
		const trained = await run(["train"], { python });
		expect(trained.code).toBe(0);
		expect(JSON.parse(trained.out)).toMatchObject({ status: "trained" });
		await run(["enable"]);
		expect(JSON.parse((await run(["status"])).out).serving.status).toBe("active");
	});

	it("rejects an unknown subcommand with usage", async () => {
		const { code, err } = await run(["frobnicate"]);
		expect(code).toBe(2);
		expect(err).toContain("Usage: autorag ltr");
	});
});
