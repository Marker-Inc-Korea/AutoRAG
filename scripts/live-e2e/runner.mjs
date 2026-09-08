/**
 * scripts/live-e2e/runner.mjs — Fixed corpus bootstrap & verification,
 * environment, fingerprint, lock, and cleanup commands.
 *
 * Commands:
 *   bootstrap --root <path>
 *   verify-corpus --root <path>
 *   print-env --mode <warm|cold> --json [--root <path>]
 *   lock-probe [--hold-ms <ms>]
 *
 * This is a plain JS ESM module (.mjs).  All type annotations use JSDoc.
 */

import { existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
	buildEnv,
	loadFingerprint,
	removeE2eState,
	tryLock,
	writeFingerprint,
} from "./env.mjs";

// ── Re-export sub-module functions for direct test imports ────────────

export { bootstrap, verifyCorpus } from "./bootstrap.mjs";

// ── Paths ────────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const E2E_DIR = join(REPO_ROOT, ".autorag-e2e");

// ── Root resolution ──────────────────────────────────────────────────

/**
 * Resolve the shared corpus root from CLI args or env.
 * Priority: --root flag > AUTORAG_LIVE_E2E_ROOT env var > repo root.
 * @param {readonly string[]} args
 * @returns {string} resolved absolute path
 */
function resolveRoot(args) {
	const rootIndex = args.indexOf("--root");
	if (rootIndex !== -1 && args[rootIndex + 1]) {
		return resolve(args[rootIndex + 1]);
	}
	// Fallback to env var
	const envRoot = process.env.AUTORAG_LIVE_E2E_ROOT;
	if (envRoot) {
		return resolve(envRoot);
	}
	// Default: use repo root itself
	return REPO_ROOT;
}

// ── Print-env command ────────────────────────────────────────────────

/**
 * @param {readonly string[]} args
 */
async function cmdPrintEnv(args) {
	const modeIndex = args.indexOf("--mode");
	const mode = modeIndex !== -1 ? args[modeIndex + 1] : "warm";

	if (mode !== "warm" && mode !== "cold") {
		console.error(`ERROR: --mode must be "warm" or "cold", got "${mode}"`);
		process.exit(1);
	}

	const root = resolveRoot(args);

	// Guard: corpus must be bootstrapped
	const corpusManifest = join(root, "corpus", "MANIFEST.json");
	if (!existsSync(corpusManifest)) {
		console.error(
			"ERROR: live-e2e-root-not-bootstrapped: " + root + " is missing corpus/MANIFEST.json",
		);
		process.exit(1);
	}

	// Cold mode: blow away .autorag-e2e and rebuild
	if (mode === "cold") {
		removeE2eState();
	}

	// Ensure .autorag-e2e exists
	mkdirSync(E2E_DIR, { recursive: true });

	// Build the environment
	const env = buildEnv(root);

	// Warm-mode fingerprint check
	if (mode === "warm") {
		let prevFp = null;
		try {
			prevFp = loadFingerprint();
		} catch (/** @type {unknown} */ e) {
			// File exists but is malformed — treat as mismatch
			const msg = e instanceof Error ? e.message : String(e);
			console.error("FATAL: " + msg);
			console.error(
				"ERROR: live-e2e-fingerprint-mismatch — stale state detected. Run with --mode cold to rebuild.",
			);
			process.exit(1);
		}
		if (prevFp !== null && !fingerprintsMatch(prevFp, env.fingerprint)) {
			console.error(
				"ERROR: live-e2e-fingerprint-mismatch — stale state detected. Run with --mode cold to rebuild.",
			);
			process.exit(1);
		}
	}

	// Persist the fingerprint (always)
	writeFingerprint(env.fingerprint);

	// Output
	const hasJson = args.includes("--json");
	if (hasJson) {
		console.log(JSON.stringify(env, null, 2));
	} else {
		console.log("AUTORAG_HOME=" + env.AUTORAG_HOME);
		console.log("AUTORAG_CONFIG=" + env.AUTORAG_CONFIG);
		console.log("AUTORAG_WORKSPACE=" + env.AUTORAG_WORKSPACE);
		console.log("AUTORAG_SEARCH_PATHS=" + env.AUTORAG_SEARCH_PATHS);
		console.log("AUTORAG_MEMORY_PATH=" + env.AUTORAG_MEMORY_PATH);
		console.log("globalHomeFallback=" + env.globalHomeFallback);
		console.log("runnerSchemaVersion=" + env.runnerSchemaVersion);
		console.log("corpusVersion=" + env.fingerprint.corpusVersion);
		console.log("corpusDigest=" + env.fingerprint.corpusDigest);
		console.log("gitCommitSha=" + env.fingerprint.gitCommitSha);
		console.log("gitDirty=" + env.fingerprint.gitDirty);
	}
}

// ── Fingerprint comparison ───────────────────────────────────────────

/**
 * @param {Record<string, unknown>} a
 * @param {Record<string, unknown>} b
 * @returns {boolean}
 */
function fingerprintsMatch(a, b) {
	const keys = [
		"runnerSchemaVersion",
		"corpusVersion",
		"corpusDigest",
		"gitCommitSha",
		"embeddingModel",
		"embeddingDimension",
		"embeddingService",
		"parserConfig",
		"minSyncConfig",
	];
	for (const key of keys) {
		if (a[key] !== b[key]) return false;
	}
	return true;
}

// ── Lock-probe command ───────────────────────────────────────────────

/**
 * @param {readonly string[]} args
 */
function cmdLockProbe(args) {
	const holdIndex = args.indexOf("--hold-ms");
	let holdMs = 0;
	if (holdIndex !== -1 && args[holdIndex + 1]) {
		const raw = args[holdIndex + 1];
		const parsed = Number(raw);
		if (!Number.isFinite(parsed) || parsed < 0) {
			console.error("ERROR: --hold-ms must be a non-negative integer, got \"" + raw + "\"");
			process.exit(1);
		}
		holdMs = parsed;
	}

	// Ensure the e2e directory exists so lock dir can be created
	mkdirSync(E2E_DIR, { recursive: true });

	const result = tryLock(holdMs);

	switch (result.kind) {
		case "acquired":
			console.log("live-e2e-lock-acquired");
			process.exit(0);
		case "held":
			console.log(result.reason);
			process.exit(1);
	}
}

// ── CLI entrypoint ───────────────────────────────────────────────────

async function main() {
	const args = process.argv.slice(2);
	if (args.length === 0) {
		console.error("usage: node scripts/live-e2e/runner.mjs <command> [options]");
		console.error("  commands: bootstrap, verify-corpus, print-env, lock-probe");
		process.exit(2);
	}

	const command = args[0];

	// print-env and lock-probe handle their own flag parsing
	if (command === "print-env") {
		return cmdPrintEnv(args);
	}

	if (command === "lock-probe") {
		return cmdLockProbe(args);
	}

	// bootstrap and verify-corpus need --root
	const rootIndex = args.indexOf("--root");
	const root = rootIndex !== -1 ? args[rootIndex + 1] : undefined;

	if (!root) {
		console.error("ERROR: --root <path> is required");
		process.exit(2);
	}

	switch (command) {
		case "bootstrap": {
			const { bootstrap: doBootstrap } = await import("./bootstrap.mjs");
			await doBootstrap(root);
			console.log("BOOTSTRAP_OK: " + resolve(root));
			process.exit(0);
		}
		case "verify-corpus": {
			const { verifyCorpus: doVerify } = await import("./bootstrap.mjs");
			const result = await doVerify(root);
			if (result.ok) {
				console.log("VERIFY_OK: corpus integrity confirmed");
				process.exit(0);
			} else {
				for (const err of result.errors) {
					console.error("VERIFY_FAIL: " + err);
				}
				process.exit(1);
			}
		}
		default: {
			console.error("ERROR: unknown command \"" + command + "\"");
			console.error("  valid commands: bootstrap, verify-corpus, print-env, lock-probe");
			process.exit(2);
		}
	}
}

// Guard: only run main() when invoked directly, not when imported by tests
const isDirectRun = process.argv[1] && fileURLToPath(import.meta.url) === resolve(process.argv[1]);
if (isDirectRun) {
	// no-excuse-ok: catch — CLI top-level boundary
	main().catch((e) => {
		console.error("FATAL:", e);
		process.exit(1);
	});
}