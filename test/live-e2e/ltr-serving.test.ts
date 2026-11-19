/**
 * Live E2E: real searches fill the ranking log, real XGBoost trains a model from it, and the model serves a real
 * search through the opt-in `ltr` setting.
 *
 * Gated on AUTORAG_LTR_LIVE=1 plus an OpenRouter credential; the offline suite never calls a model. The librarian, a
 * real MinSync index (dense + SPLADE), the real hosted Jev judge, a real OpenRouter reranker and a real 2026+ chat
 * model run every search. Training runs the shipped WASM XGBoost trainer, in process. To keep a live run
 * small, the log is seeded with `SEED_RUNS` replayed runs on top of the real searches (same record shape, written
 * through the real recorder), because a trainer needs ~100 runs and a real search costs several model calls.
 *
 *   AUTORAG_LTR_LIVE=1 [AUTORAG_LTR_LIVE_MODEL=qwen/qwen3.7-flash] bun run vitest run test/live-e2e/ltr-serving.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { runLtr } from "../../src/cli/commands/ltr.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { listLtrModels, ltrDirectory } from "../../src/ltr/model-store.ts";
import { MIN_TRAINING_RUNS } from "../../src/ltr/trainer.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { realMinSync } from "../helpers/real-minsync.ts";
import { recordedRun } from "../ltr/helpers.ts";

const LIVE = process.env.AUTORAG_LTR_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_LTR_LIVE_MODEL ?? "qwen/qwen3.7-flash";
const RERANK_MODEL = process.env.AUTORAG_LTR_LIVE_RERANK ?? "qwen/qwen3-reranker-8b";
const SEED_RUNS = MIN_TRAINING_RUNS + 40;

const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR"] as const;
let root: string;
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "arag-ltr-serving-live-"));
	savedEnv = Object.fromEntries(RESTORED_VARIABLES.map((name) => [name, process.env[name]]));
});

afterEach(() => {
	for (const name of RESTORED_VARIABLES) {
		const value = savedEnv[name];
		if (value === undefined) delete process.env[name];
		else process.env[name] = value;
	}
	rmSync(root, { recursive: true, force: true });
});

const DOCS: Readonly<Record<string, string>> = {
	"lease.md": "# Seoul office lease\n\nThe Seoul office lease renews on 2027-03-31.\nMonthly rent is 4,800,000 KRW.\n",
	"refund.md":
		"# Refund policy\n\nRefund exceptions require director approval before payout.\nStandard refunds are paid within 14 days.\n",
	"api.md": "# API limits\n\nThe public API allows 100 requests per minute per key.\nExceeding it returns HTTP 429.\n",
	"lunch.md": "# Cafeteria\n\nToday's lunch: bibimbap 9,000 KRW, kimchi stew 8,000 KRW.\n",
	"handbook.md": "# Handbook\n\nTable of contents\n1. Overview\n2. Badges and parking\n3. Contacts\n",
	"onboarding.md": "# Onboarding\n\nNew hires get a laptop on day one. Badges are issued by facilities.\n",
};

const QUESTIONS = [
	"When does the Seoul office lease renew?",
	"Who must approve refund exceptions?",
	"What is the API rate limit per key?",
];

function ctx(
	configPath: string,
	positionals: string[],
	out: string[],
	flags: CommandContext["flags"] = {},
): CommandContext {
	return {
		positionals,
		flags: { config: configPath, ...flags },
		json: true,
		debug: false,
		cwd: root,
		stdout: (line) => out.push(line),
		stderr: (line) => out.push(line),
	};
}

describe.skipIf(!LIVE)("learning-to-rank training and serving, live stack", () => {
	it("trains an XGBoost model from the log with real xgboost, then serves it on a real search", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		for (const [name, body] of Object.entries(DOCS)) writeFileSync(join(docs, name), body);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		const memoryPath = join(root, "memory.json");
		const configPath = join(root, "config.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				searchPaths: [docs],
				workspacePath: root,
				memoryPath,
				rerank: { provider: "openrouter", model: RERANK_MODEL },
			}),
		);
		const resolveFor = () =>
			resolveConfig({
				flags: { config: configPath, "model-provider": "openrouter", "model-id": MODEL_ID },
				env: { ...process.env },
				cwd: root,
				readOnly: true,
			});
		const buildAgent = async (ltr?: { enabled: boolean; model: string }) => {
			const config = resolveFor();
			const resolved = await resolveAgentModel(config);
			return {
				config,
				agent: new AutoRAGAgent({
					model: resolved.model,
					...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
					...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
					searchPaths: [docs],
					workspacePath: root,
					memoryPath,
					minSync: { autoInstall: false, binaryPath: realMinSync() },
					jev: { backend: "openrouter" },
					rerank: { provider: "openrouter", model: RERANK_MODEL },
					jikji: false,
					everything: false,
					fsearch: false,
					webSearch: false,
					piAgentDir: agentDir,
					piSessionDir: join(root, "sessions"),
					...(ltr !== undefined ? { ltr } : {}),
				}),
			};
		};

		// 1. Real searches, LTR off: they fill the log with real candidates, real rerank output and real labels.
		const { agent: first } = await buildAgent();
		const refreshed = await first.refresh(true, { methods: ["parsed", "minsync"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);
		for (const question of QUESTIONS) {
			const response = await first.searchDocuments(question);
			const codes = response.diagnostics?.map((diagnostic) => diagnostic.code) ?? [];
			console.log(`--- ${question}\n${response.answer}\ndiagnostics: ${codes.join(", ")}`);
			expect(codes).not.toContain("ltr-unavailable");
			expect(codes).not.toContain("rerank-failed");
		}
		const store = new RankingLogStore(rankingLogPath(memoryPath));
		const real = store.list().runs;
		expect(real.length).toBe(QUESTIONS.length);
		expect(real.every((run) => run.reranker?.model === RERANK_MODEL)).toBe(true);
		expect(real.every((run) => run.ltr === undefined)).toBe(true);

		// 2. Top the log up to a trainable size with runs of the same shape (labelled, reranked, exposed).
		for (let index = 0; index < SEED_RUNS; index += 1) {
			store.append(
				recordedRun(`seed-${String(index).padStart(3, "0")}`, 1_000 + index, { gold: index % 6 }).finish(true),
			);
		}
		store.close();

		// 3. Train through the CLI with the real trainer (XGBoost WASM, in process). Not enabled yet: searches must be unchanged.
		const trainOut: string[] = [];
		const trainCode = await runLtr(ctx(configPath, ["train"], trainOut));
		console.log(`train exit ${trainCode}: ${trainOut.join("\n").slice(0, 1200)}`);
		const trained = JSON.parse(trainOut.join("\n")) as {
			status: string;
			model?: { id: string; trainer?: { xgboost: string } };
		};
		expect(["trained", "rejected"]).toContain(trained.status);
		expect(trained.model?.trainer?.xgboost).toMatch(/^\d/u);
		const models = listLtrModels(ltrDirectory(memoryPath));
		expect(models).toHaveLength(1);
		expect(existsSync(join(ltrDirectory(memoryPath), "models", models[0]?.id ?? "", "model.json"))).toBe(true);

		// 4. Serve it on a real search. A model that did not pass its gate is served by explicit id (the operator's
		// choice), so this step proves the serving path on real model output whatever the synthetic holdout said.
		const modelId = models[0]?.id ?? "";
		const useOut: string[] = [];
		expect(await runLtr(ctx(configPath, ["use", modelId], useOut))).toBe(0);
		expect(await runLtr(ctx(configPath, ["enable"], useOut))).toBe(0);
		const fileConfig = JSON.parse(readFileSync(configPath, "utf-8")) as { ltr?: unknown; rerank?: unknown };
		expect(fileConfig.ltr).toEqual({ enabled: true, model: modelId });
		expect(fileConfig.rerank).toEqual({ provider: "openrouter", model: RERANK_MODEL });

		const { agent: servingAgent, config } = await buildAgent({ enabled: true, model: modelId });
		expect(config.ltr).toEqual({ enabled: true, model: modelId });
		const response = await servingAgent.searchDocuments("What is the monthly rent for the Seoul office?");
		const codes = response.diagnostics?.map((diagnostic) => diagnostic.code) ?? [];
		console.log(`--- served\n${response.answer}\ndiagnostics: ${codes.join(", ")}`);
		expect(codes).not.toContain("ltr-unavailable");
		expect(codes).not.toContain("ltr-failed");
		expect(response.answer.length).toBeGreaterThan(0);

		const after = new RankingLogStore(rankingLogPath(memoryPath));
		const logged = after.list().runs.find((run) => run.runId === response.sessionId);
		after.close();
		expect(logged?.ltr).toEqual({ modelId });
		// The log keeps the reranker's own order for training, and the order the model was shown differs only by the model.
		expect(logged?.candidates.some((candidate) => candidate.rerank !== undefined)).toBe(true);
		expect(
			logged?.candidates.some((candidate) =>
				candidate.exposures.some((exposure) => exposure.surface === "baseline"),
			),
		).toBe(true);
	}, 1_500_000);
});
