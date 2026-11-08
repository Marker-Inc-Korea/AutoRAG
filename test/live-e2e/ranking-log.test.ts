/**
 * Live E2E: real searches fill the learning-to-rank log, and the export trains a real LightGBM LambdaRank model.
 *
 * Gated on AUTORAG_LTR_LIVE=1 plus an OpenRouter credential; the offline suite never calls a model. The librarian,
 * a real MinSync index (dense + SPLADE), the real hosted Jev judge, and a real 2026+ chat model run every search.
 * When `uv` is on PATH the exported files are also fed to LightGBM (`uv run --with lightgbm`), which must load the
 * `.query`/`.position` sidecars and train without error.
 *
 *   AUTORAG_LTR_LIVE=1 [AUTORAG_LTR_LIVE_MODEL=qwen/qwen3.7-flash] bun run vitest run test/live-e2e/ranking-log.test.ts
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";
import { buildLtrDataset, LTR_FEATURE_NAMES, writeLtrDataset } from "../../src/memory/ltr-export.ts";
import { RankingLogStore, rankingLogPath } from "../../src/memory/ranking-log.ts";
import { realMinSync } from "../helpers/real-minsync.ts";

const LIVE = process.env.AUTORAG_LTR_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_LTR_LIVE_MODEL ?? "qwen/qwen3.7-flash";

const RESTORED_VARIABLES = ["PI_CODING_AGENT_DIR"] as const;
let root: string;
let savedEnv: Record<string, string | undefined>;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "arag-ltr-live-"));
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
	"How much is the monthly rent for the Seoul office?",
];

describe.skipIf(!LIVE)("learning-to-rank log, live stack", () => {
	it("logs every real search's candidate pool with labels and exports files LightGBM trains on", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		for (const [name, body] of Object.entries(DOCS)) writeFileSync(join(docs, name), body);
		const agentDir = join(root, "agent");
		mkdirSync(agentDir, { recursive: true });
		process.env.PI_CODING_AGENT_DIR = agentDir;
		writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
		const config = resolveConfig({
			flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
			env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
			cwd: root,
			readOnly: true,
		});
		const resolved = await resolveAgentModel(config);
		const memoryPath = join(root, "memory.json");
		const agent = new AutoRAGAgent({
			model: resolved.model,
			...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
			...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
			searchPaths: [docs],
			workspacePath: root,
			memoryPath,
			minSync: { autoInstall: false, binaryPath: realMinSync() },
			jev: { backend: "openrouter" },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			piAgentDir: agentDir,
			piSessionDir: join(root, "sessions"),
		});
		const refreshed = await agent.refresh(true, { methods: ["parsed", "minsync"] });
		expect(refreshed.minsync?.ok, JSON.stringify(refreshed.minsync)).toBe(true);

		const sessions: string[] = [];
		for (const question of QUESTIONS) {
			const response = await agent.searchDocuments(question);
			sessions.push(response.sessionId);
			const codes = response.diagnostics?.map((diagnostic) => diagnostic.code) ?? [];
			console.log(`--- ${question}\n${response.answer}\ndiagnostics: ${codes.join(", ")}`);
			expect(codes).not.toContain("ranking-log-failed");
		}

		const store = new RankingLogStore(rankingLogPath(memoryPath));
		const { runs, skipped } = store.list();
		store.close();
		expect(skipped).toBe(0);
		expect(runs.map((run) => run.runId)).toEqual(sessions);
		for (const run of runs) {
			// The whole pool is logged, not only what the answer cited.
			expect(run.candidates.length, run.question).toBeGreaterThan(1);
			expect(run.judgments.citationRelevance === "judged" || run.judgments.citationRelevance === "not-run").toBe(
				true,
			);
			for (const candidate of run.candidates) {
				expect(candidate.observations.length + candidate.exposures.length).toBeGreaterThan(0);
			}
		}
		// A real model sometimes answers without citing; such a run has no positive and the export must skip it.
		const withPositive = runs.filter((run) => run.candidates.some((candidate) => candidate.labels.citedFinal));
		const withoutPositive = runs.length - withPositive.length;
		expect(withPositive.length).toBeGreaterThanOrEqual(2);
		for (const run of withPositive) {
			expect(
				run.candidates.some((candidate) => !candidate.labels.citedFinal),
				run.question,
			).toBe(true);
			expect(run.judgments.citationRelevance, run.question).toBe("judged");
		}
		// Chunk text never reaches the log.
		expect(readFileSync(rankingLogPath(memoryPath)).includes("bibimbap")).toBe(false);

		const dataset = buildLtrDataset(runs, { validFraction: 0.25 });
		const output = join(root, "ltr");
		writeLtrDataset(dataset, output);
		console.log(`manifest: ${readFileSync(join(output, "manifest.json"), "utf-8")}`);
		expect(dataset.train.groups.length + dataset.valid.groups.length).toBe(withPositive.length);
		expect(dataset.manifest.skippedGroups.noLabelVariance).toBe(withoutPositive);
		expect(dataset.train.rows.some((row) => row.label >= 2)).toBe(true);
		expect(dataset.train.rows.some((row) => row.label === 0)).toBe(true);

		const uv = spawnSync("uv", ["--version"], { encoding: "utf-8" });
		if (uv.status !== 0) {
			console.log("uv not on PATH: LightGBM training skipped");
			return;
		}
		const train = spawnSync(
			"uv",
			[
				"run",
				"--quiet",
				"--with",
				"lightgbm",
				"--with",
				"numpy",
				"python",
				"-c",
				[
					"import json, sys, lightgbm as lgb",
					"d = sys.argv[1]",
					"m = json.load(open(f'{d}/manifest.json'))",
					"ds = lgb.Dataset(f'{d}/train.tsv', params={'header': True, 'label_column': 0})",
					"booster = lgb.train({'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [1, 3], 'min_data_in_leaf': 1, 'min_data_in_bin': 1, 'verbose': -1}, ds, num_boost_round=5)",
					"assert booster.num_feature() == len(m['featureNames']), (booster.num_feature(), len(m['featureNames']))",
					"print(json.dumps({'features': booster.num_feature(), 'trees': booster.num_trees(), 'groups': len(ds.get_group())}))",
				].join("\n"),
				output,
			],
			{ encoding: "utf-8", timeout: 600_000 },
		);
		console.log(`lightgbm: ${train.stdout}${train.stderr}`);
		expect(train.status, train.stderr).toBe(0);
		const trained = JSON.parse(train.stdout.trim().split("\n").at(-1) ?? "{}");
		expect(trained.features).toBe(LTR_FEATURE_NAMES.length);
		expect(trained.groups).toBe(dataset.train.groups.length);
		expect(existsSync(join(output, "train.tsv.position"))).toBe(true);
	}, 900_000);
});
