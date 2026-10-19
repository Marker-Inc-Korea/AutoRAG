import { spawn, spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	assertKoreanLexicalHit,
	assertLiteRetrieveHealthy,
	assertLiteRetrieveReportsSkippedSurface,
	buildLiveStackOptions,
	REPO_ROOT,
	resolveKoreanLexicalLane,
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
// The Korean fixture holds only suffixed forms of 인증서 (인증서를, 인증서는). Its bare stem reaches the fixture through
// BM25 only when the agent built the index with the multilingual tokenizer, which AutoRAG always selects and which
// needs Kiwi on the host. Without Kiwi the sync falls back to whitespace tokenization and the lane is a SKIP.
const koreanLane = resolveKoreanLexicalLane(
	refresh.minsync.lexicalFallback,
	process.env.AUTORAG_LIVE_E2E_KIWI === "required",
);
if (koreanLane === "assert") {
	const minsyncExecutable = process.platform === "win32" ? "minsync.exe" : "minsync";
	const cachedMinsync = join(workspace, ".autorag", "bin", minsyncExecutable);
	const koreanQuery = spawnSync(
		existsSync(cachedMinsync) ? cachedMinsync : minsyncExecutable,
		["query", "--mode", "bm25", "--format", "json", "-k", "5", "인증서"],
		{ cwd: workspace, encoding: "utf8", timeout: 60_000 },
	);
	if (koreanQuery.status !== 0) throw new Error(`korean-lexical-query-exit-${koreanQuery.status}`);
	assertKoreanLexicalHit(koreanQuery.stdout, "korean-policy.txt");
}
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

/** Starts the stub in its own process (see openai-stub.mjs) and resolves once it is listening. */
async function startOpenAiStub(dimension: number) {
	const child = spawn("node", [join(REPO_ROOT, "scripts", "live-e2e", "openai-stub.mjs"), String(dimension)], {
		stdio: ["ignore", "pipe", "inherit"],
	});
	let output = "";
	const port = await new Promise<number>((resolvePort, reject) => {
		child.stdout.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			output += chunk;
			const line = output.split("\n").find((candidate) => candidate.startsWith("{") && candidate.includes("port"));
			if (line !== undefined) resolvePort((JSON.parse(line) as { port: number }).port);
		});
		child.once("error", reject);
		child.once("exit", (code) => reject(new Error(`openai-stub-exited-${code}`)));
	});
	return {
		baseUrl: `http://127.0.0.1:${port}`,
		/** Stops the stub and returns how many embedding requests it served. */
		stop: () =>
			new Promise<number>((resolveStop) => {
				child.removeAllListeners("exit");
				child.stdout.on("data", () => {
					const done = output.split("\n").find((candidate) => candidate.includes("requests"));
					if (done !== undefined) resolveStop((JSON.parse(done) as { requests: number }).requests);
				});
				child.once("exit", () => {
					const done = output.split("\n").find((candidate) => candidate.includes("requests"));
					resolveStop(done === undefined ? 0 : (JSON.parse(done) as { requests: number }).requests);
				});
				child.kill("SIGTERM");
			}),
	};
}

function runLiteRetrieve(configPath: string, extraArgs: readonly string[] = [], env: NodeJS.ProcessEnv = process.env) {
	const run = spawnSync(
		"bun",
		["src/cli/index.ts", "lite", "retrieve", QUERY, "--config", configPath, ...extraArgs],
		{ cwd: REPO_ROOT, env, encoding: "utf8", timeout: 5 * 60 * 1000 },
	);
	return { exitCode: run.status ?? 1, stdout: run.stdout ?? "", stderr: run.stderr ?? "" };
}

const healthyConfig = writeConfig(join(workspace, "lite-retrieve.json"));
const healthyRun = runLiteRetrieve(healthyConfig, ["--json"]);
if (healthyRun.exitCode !== 0) throw new Error(`lite-retrieve-exit-${healthyRun.exitCode}`);
const healthyEnvelope = JSON.parse(healthyRun.stdout);
assertLiteRetrieveHealthy(healthyEnvelope, resolve(source));

let embedderDown:
	| {
			exitCode: number;
			ok: boolean;
			unsearched: unknown;
			diagnostics: unknown;
			skippedSurface: string;
			skippedMethods: readonly string[];
			reasonVerbatim: string;
			humanWarningWithoutDebug: readonly string[];
	  }
	| undefined;

// A remote endpoint is a first-class backend, and `minsync query` reads the embedder recorded in the
// workspace's own config, so "the embedder is down" has to be tested on a workspace that was SYNCED through
// that endpoint. Build one against a loopback OpenAI-compatible stub, then stop the stub: retrieval must
// still answer, name the skipped surface, and hand back the real error instead of a generic sentence.
const remoteWorkspace = join(workspace, "remote-embedder");
const remoteCorpus = join(remoteWorkspace, "corpus");
// The fixture corpus is read-only, and a copy keeps that mode, so a warm run could not overwrite the previous
// run's copy. Write the content instead of the mode.
mkdirSync(remoteCorpus, { recursive: true });
rmSync(join(remoteCorpus, "sample.txt"), { force: true });
writeFileSync(join(remoteCorpus, "sample.txt"), readFileSync(source));
const stub = await startOpenAiStub(1024);
// The key is a placeholder under an operator-chosen variable name: the stub ignores it, and the run proves the
// name is mapped to the OPENAI_API_KEY that `minsync` reads.
process.env.AUTORAG_LIVE_E2E_REMOTE_KEY = "live-e2e-placeholder";
const remoteEmbedder = {
	id: "openai:stub-embedding",
	baseUrl: stub.baseUrl,
	apiKeyEnv: "AUTORAG_LIVE_E2E_REMOTE_KEY",
	dimension: 1024,
	timeoutMs: 5_000,
};
const remoteOptions = {
	...buildLiveStackOptions(root, remoteWorkspace),
	searchPaths: [remoteCorpus],
	minSync: { workspacePath: remoteWorkspace, autoInstall: true, embedder: remoteEmbedder },
};
const remoteConfig = join(remoteWorkspace, "config.json");
// The runner points this process at the main workspace through AUTORAG_* variables, and the environment wins
// over a config file. This lane has its own workspace and corpus, so those variables are dropped for its calls.
const remoteEnv: NodeJS.ProcessEnv = { ...process.env };
for (const name of ["AUTORAG_WORKSPACE", "AUTORAG_SEARCH_PATHS", "AUTORAG_MEMORY_PATH", "AUTORAG_CONFIG"]) delete remoteEnv[name];
writeFileSync(remoteConfig, `${JSON.stringify(remoteOptions, null, 2)}\n`);
const remoteAgent = new AutoRAGAgent(remoteOptions);
const remoteRefresh = await remoteAgent.refresh(true, { methods: ["parsed", "minsync"] });
if (remoteRefresh.minsync?.ok !== true) throw new Error(`remote-embedder-sync-${remoteRefresh.minsync?.reason ?? "unready"}`);
const remoteHealthy = runLiteRetrieve(remoteConfig, ["--json"], remoteEnv);
if (remoteHealthy.exitCode !== 0) throw new Error(`remote-retrieve-exit-${remoteHealthy.exitCode}`);
assertLiteRetrieveHealthy(JSON.parse(remoteHealthy.stdout), realpathSync(join(remoteCorpus, "sample.txt")));
const stubRequests = await stub.stop();
if (stubRequests === 0) throw new Error("remote-embedder-never-called");

const degradedRun = runLiteRetrieve(remoteConfig, ["--json"], remoteEnv);
if (degradedRun.exitCode !== 0) throw new Error(`lite-retrieve-degraded-exit-${degradedRun.exitCode}`);
const degradedEnvelope = JSON.parse(degradedRun.stdout);
const skipped = assertLiteRetrieveReportsSkippedSurface(degradedEnvelope, "minsync");

// The same skip must reach human output without --debug.
const degradedHuman = runLiteRetrieve(remoteConfig, [], remoteEnv);
if (!degradedHuman.stdout.includes("warning: not searched: minsync")) {
	throw new Error("lite-retrieve-human-skip-missing");
}
embedderDown = {
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
};

const status = await agent.getRefreshStatus();
const result = {
	mode,
	refresh: { parsed: { scanned: refresh.scanned, written: refresh.written, skipped: refresh.skipped }, minsync: refresh.minsync },
	cursorPath: resolve(cursorPath),
	cursorExists: true,
	koreanLexical:
		koreanLane === "assert"
			? { status: "PASS", query: "인증서", fixture: "korean-policy.txt" }
			: { status: "SKIP", reason: "kiwi-unavailable", lexicalFallback: refresh.minsync.lexicalFallback },
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
		...(embedderDown !== undefined ? { embedderDown } : {}),
	},
	embedding: {
		endpoint: process.env.AUTORAG_EMBED_ENDPOINT,
		model: "Qwen/Qwen3-Embedding-0.6B",
		dimension: 1024,
		queryPrefix: "",
		passagePrefix: "",
		loopbackOnly: true,
	},
	openaiKeyPresent: Boolean(process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY),
	incremental: mode === "warm" && cursorExistedBefore && refresh.written === 0,
	fullSync: !cursorExistedBefore,
};
writeFileSync(join(workspace, "live-stack-result.json"), `${JSON.stringify(result, null, 2)}\n`);
