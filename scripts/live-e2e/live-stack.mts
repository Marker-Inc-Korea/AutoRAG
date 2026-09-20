import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	assertLiteRetrieveHealthy,
	assertLiteRetrieveReportsSkippedSurface,
	buildLiveStackOptions,
	REPO_ROOT,
} from "./workflow.mjs";

const args = process.argv.slice(2);
const value = (name: string): string => {
	const index = args.indexOf(name);
	const result = index >= 0 ? args[index + 1] : undefined;
	if (!result) throw new Error(`missing ${name}`);
	return resolve(result);
};
const root = value("--root");
const workspace = value("--workspace");
const mode = args.includes("--mode") ? args[args.indexOf("--mode") + 1] : "warm";
if (mode !== "cold" && mode !== "warm") throw new Error("invalid mode");

const source = realpathSync(join(root, "corpus", "sample.txt"));
const cursorPath = join(workspace, ".minsync", "cursor.json");
const cursorExistedBefore = existsSync(cursorPath);
const agent = new AutoRAGAgent(buildLiveStackOptions(root, workspace));
const refresh = await agent.refresh(mode === "cold", { methods: ["parsed", "minsync"] });
if (refresh.minsync?.ok !== true) throw new Error(`minsync-${refresh.minsync?.reason ?? "unready"}`);
if (!existsSync(cursorPath)) throw new Error("minsync-cursor-missing");
const hits = await agent.retrieve("semantic question about the sample corpus", { topK: 5 });
const hit = hits.find((result) => result.source === resolve(source));
if (!hit) throw new Error("semantic-hit-source-mismatch");
if (!existsSync(hit.source)) throw new Error("source-missing");
const content = readFileSync(hit.source, "utf8");
if (content.length === 0) throw new Error("source-empty");
// ---------------------------------------------------------------------------
// `autorag lite retrieve --json` — the machine contract skills consume.
// ---------------------------------------------------------------------------

const QUERY = "semantic question about the sample corpus";

function writeConfig(path: string, embedderOverride: Record<string, unknown> = {}): string {
	const base = buildLiveStackOptions(root, workspace);
	const minSync = base.minSync === false || base.minSync === undefined ? {} : base.minSync;
	const config = {
		...base,
		minSync: { ...minSync, embedder: { ...minSync.embedder, ...embedderOverride } },
	};
	writeFileSync(path, `${JSON.stringify(config, null, 2)}\n`);
	return path;
}

function runLiteRetrieve(configPath: string, extraArgs: readonly string[] = []) {
	const run = spawnSync(
		"bun",
		["src/cli/index.ts", "lite", "retrieve", QUERY, "--config", configPath, ...extraArgs],
		{ cwd: REPO_ROOT, env: process.env, encoding: "utf8", timeout: 5 * 60 * 1000 },
	);
	return { exitCode: run.status ?? 1, stdout: run.stdout ?? "", stderr: run.stderr ?? "" };
}

const healthyConfig = writeConfig(join(workspace, "lite-retrieve.json"));
const healthyRun = runLiteRetrieve(healthyConfig, ["--json"]);
if (healthyRun.exitCode !== 0) throw new Error(`lite-retrieve-exit-${healthyRun.exitCode}`);
const healthyEnvelope = JSON.parse(healthyRun.stdout);
assertLiteRetrieveHealthy(healthyEnvelope, resolve(source));

// Point MinSync at a closed loopback port so the embedder is genuinely down for
// this call. Retrieval must still answer, name the skipped surface, and hand
// back the real error instead of a generic sentence.
const brokenConfig = writeConfig(join(workspace, "lite-retrieve-embedder-down.json"), {
	baseUrl: "http://127.0.0.1:9",
	timeoutMs: 5_000,
});
const degradedRun = runLiteRetrieve(brokenConfig, ["--json"]);
if (degradedRun.exitCode !== 0) throw new Error(`lite-retrieve-degraded-exit-${degradedRun.exitCode}`);
const degradedEnvelope = JSON.parse(degradedRun.stdout);
const skipped = assertLiteRetrieveReportsSkippedSurface(degradedEnvelope, "minsync");

// The same skip must reach human output without --debug.
const degradedHuman = runLiteRetrieve(brokenConfig);
if (!degradedHuman.stdout.includes("warning: not searched: minsync")) {
	throw new Error("lite-retrieve-human-skip-missing");
}

const status = await agent.getRefreshStatus();
const result = {
	mode,
	refresh: { parsed: { scanned: refresh.scanned, written: refresh.written, skipped: refresh.skipped }, minsync: refresh.minsync },
	cursorPath: resolve(cursorPath),
	cursorExists: true,
	hit: { source: hit.source, sourceAbsolute: true, sourceExists: true, sourceReadable: true, score: hit.score },
	status: { state: status.state, stale: status.stale, components: status.components },
	liteRetrieve: {
		healthy: {
			exitCode: healthyRun.exitCode,
			ok: healthyEnvelope.ok,
			stale: healthyEnvelope.stale,
			resultCount: healthyEnvelope.results.length,
			unsearched: healthyEnvelope.unsearched,
			sourceMatched: resolve(source),
		},
		embedderDown: {
			exitCode: degradedRun.exitCode,
			ok: degradedEnvelope.ok,
			unsearched: degradedEnvelope.unsearched,
			diagnostics: degradedEnvelope.diagnostics,
			skippedSurface: skipped.surface,
			skippedMethods: skipped.methods,
			reasonVerbatim: skipped.reason,
			humanWarningWithoutDebug: degradedHuman.stdout
				.split("\n")
				.filter((line: string) => line.startsWith("warning: not searched:")),
		},
	},
	embedding: { endpoint: process.env.AUTORAG_GATEWAY_ENDPOINT, model: "Qwen3-Embedding-0.6B-Q8_0.gguf", profileId: "qwen3-embedding-0.6b", dimension: 1024, queryPrefix: "", passagePrefix: "", loopbackOnly: true },
	openaiKeyPresent: Boolean(process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY),
	incremental: mode === "warm" && cursorExistedBefore && refresh.written === 0,
	fullSync: !cursorExistedBefore,
};
writeFileSync(join(workspace, "live-stack-result.json"), `${JSON.stringify(result, null, 2)}\n`);
