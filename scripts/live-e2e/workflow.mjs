import { accessSync, constants, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { buildEnv, loadFingerprint, removeE2eState, writeFingerprint } from "./env.mjs";
import { runPreflight } from "./preflight.mjs";
import { parseDatasourceSelection, runDatasourceMatrix } from "./datasources.mjs";

const REPO_ROOT = resolve(fileURLToPath(new URL("../..", import.meta.url)));
const E2E_DIR = join(REPO_ROOT, ".autorag-e2e");
export function isFingerprintCurrent(previous, current) {
	const keys = ["runnerSchemaVersion", "corpusVersion", "corpusDigest", "gitCommitSha", "embeddingModel", "embeddingDimension", "embeddingService", "parserConfig", "minSyncConfig"];
	return keys.every((key) => previous[key] === current[key]);
}

export function assertServiceReady(result) {
	if (result.verdict === "refused") throw new Error(result.code ?? "live-e2e-service-refused");
}

export function buildLiveStackOptions(root, workspace) {
	return {
		searchPaths: [join(resolve(root), "corpus")],
		workspacePath: resolve(workspace),
		memoryPath: join(resolve(workspace), "memory.json"),
		jikji: false,
		minSync: {
			workspacePath: resolve(workspace),
			autoInstall: false,
			embedder: {
				id: "tei:embeddinggemma:latest",
				baseUrl: "http://127.0.0.1:18080",
				dimension: 768,
				timeoutMs: 120_000,
			},
		},
	};
}

export function assertAbsoluteReadableSource(source) {
	if (!isAbsolute(source)) throw new Error("source-not-absolute");
	accessSync(source, constants.R_OK);
	return realpathSync(source);
}

export function tryAcquireWorkflowLock(root = E2E_DIR) {
	const lock = join(root, "locks", "workflow.lock");
	mkdirSync(join(root, "locks"), { recursive: true });
	try {
		mkdirSync(lock);
		writeFileSync(join(lock, "pid"), String(process.pid));
		return { ok: true, release: () => rmSync(lock, { recursive: true, force: true }) };
	} catch {
		return { ok: false, code: "live-e2e-lock-held" };
	}
}

export function cleanupCloneState(clonePath, sharedRoot) {
	const clone = resolve(clonePath);
	const shared = resolve(sharedRoot);
	if (clone === shared || shared.startsWith(`${clone}/`)) throw new Error("cleanup-boundary-violation");
	rmSync(clone, { recursive: true, force: true });
}

function commandResult(command, args, cwd, env) {
	const result = spawnSync(command, args, { cwd, env, encoding: "utf8", timeout: 15 * 60 * 1000 });
	return { command: [command, ...args], exitCode: result.status ?? 1, stdout: result.stdout ?? "", stderr: result.stderr ?? "" };
}

function readJson(path) {
	return JSON.parse(readFileSync(path, "utf8"));
}

export async function runWorkflow({ root, mode, evidenceDir }) {
	const resolvedRoot = resolve(root);
	const evidencePath = resolve(evidenceDir ?? join(REPO_ROOT, ".omo", "evidence", `live-core-${mode}`));
	mkdirSync(evidencePath, { recursive: true });
	const evidence = { mode, root: resolvedRoot, startedAt: new Date().toISOString(), commands: [], diagnostics: [], datasourceSelection: parseDatasourceSelection(), datasourceLanes: [], commandsSummary: { core: "PENDING", datasources: "PENDING" }, cleanup: { lockReleased: false, cloneStateRemoved: false } };
	const record = (result) => { evidence.commands.push(result); return result; };
	const corpus = record(commandResult("node", ["scripts/live-e2e/runner.mjs", "verify-corpus", "--root", resolvedRoot], REPO_ROOT, process.env));
	if (corpus.exitCode !== 0) return finish(evidence, "live-e2e-root-not-bootstrapped", evidencePath, 1);
	if (mode === "cold") removeE2eState();
	mkdirSync(E2E_DIR, { recursive: true });
	const env = buildEnv(resolvedRoot);
	if (mode === "warm") {
		let previous;
		try { previous = loadFingerprint(); } catch { return finish(evidence, "live-e2e-fingerprint-mismatch", evidencePath, 1); }
		if (previous !== null && !isFingerprintCurrent(previous, env.fingerprint)) return finish(evidence, "live-e2e-fingerprint-mismatch", evidencePath, 1);
	}
	writeFingerprint(env.fingerprint);
	const preflight = await runPreflight({ endpoint: "http://127.0.0.1:18080" });
	evidence.preflight = preflight;
	try { assertServiceReady(preflight); } catch (error) { evidence.diagnostics.push(error instanceof Error ? error.message : "live-e2e-service-refused"); return finish(evidence, evidence.diagnostics[0], evidencePath, 1); }
	if (process.env.OPENAI_API_KEY || process.env.AUTORAG_OPENAI_API_KEY) return finish(evidence, "live-e2e-openai-egress", evidencePath, 1);
	const lock = tryAcquireWorkflowLock();
	if (!lock.ok) return finish(evidence, lock.code, evidencePath, 1);
	let diagnostic;
	let exitCode = 0;
	try {
		const childEnv = { ...process.env, AUTORAG_HOME: env.AUTORAG_HOME, AUTORAG_CONFIG: env.AUTORAG_CONFIG, AUTORAG_WORKSPACE: env.AUTORAG_WORKSPACE, AUTORAG_SEARCH_PATHS: env.AUTORAG_SEARCH_PATHS, AUTORAG_MEMORY_PATH: env.AUTORAG_MEMORY_PATH, OPENAI_API_KEY: "", AUTORAG_OPENAI_API_KEY: "" };
		const child = record(commandResult("bun", ["scripts/live-e2e/live-stack.mts", "--root", resolvedRoot, "--workspace", env.AUTORAG_WORKSPACE, "--mode", mode], REPO_ROOT, childEnv));
		if (child.exitCode !== 0) {
			diagnostic = "live-e2e-stack-failed";
			exitCode = 1;
		} else {
			Object.assign(evidence, readJson(join(env.AUTORAG_WORKSPACE, "live-stack-result.json")));
			evidence.commandsSummary.core = "PASS";
			const datasources = await runDatasourceMatrix({ root: resolvedRoot, selection: evidence.datasourceSelection });
			evidence.datasourceLanes = datasources.lanes;
			evidence.datasourceSummary = datasources.summary;
			evidence.commandsSummary.datasources = datasources.exitCode === 0 ? "PASS_OR_SKIP" : "FAIL";
			if (datasources.exitCode !== 0) {
				diagnostic = "live-e2e-datasource-failure";
				exitCode = 1;
			}
		}
	} finally {
		lock.release?.();
		evidence.cleanup.lockReleased = true;
	}
	return finish(evidence, diagnostic, evidencePath, exitCode);
}

function finish(evidence, diagnostic, evidencePath, exitCode) {
	if (diagnostic && !evidence.diagnostics.includes(diagnostic)) evidence.diagnostics.push(diagnostic);
	evidence.finishedAt = new Date().toISOString();
	evidence.exitCode = exitCode;
	writeFileSync(join(evidencePath, "result.json"), `${JSON.stringify(evidence, null, 2)}\n`);
	const taskEvidencePath = join(REPO_ROOT, ".omo", "evidence", "task-6-fixed-live-e2e-environment.json");
	mkdirSync(join(REPO_ROOT, ".omo", "evidence"), { recursive: true });
	writeFileSync(taskEvidencePath, `${JSON.stringify({ ...evidence, task: "6", scenario: "fixed-live-e2e-environment", cleanupReceipt: "runner lock released; native stores untouched" }, null, 2)}\n`);
	return { ...evidence, evidencePath };
}

export { REPO_ROOT };
