/**
 * scripts/live-e2e/env.mjs — Environment builder for live-e2e.
 *
 * Builds the five AutoRAG override env vars pinned to .autorag-e2e under the
 * repository root, along with fingerprint and schema version metadata.
 * Exported so tests can inspect values directly.
 *
 * This is a plain JS ESM module (.mjs).  Type annotations use JSDoc.
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

export { tryLock } from "./lock.mjs";

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const E2E_DIR = join(REPO_ROOT, ".autorag-e2e");

// ── Schema version — bump when the fingerprint contract changes ──────

/** @type {number} */
export const RUNNER_SCHEMA_VERSION = 1;

// ── Environment descriptor (JSDoc types for docs, not enforced at runtime) ──

/**
 * @typedef {Object} LiveE2eEnv
 * @property {string} root - Path to corpus root (shared tree, not under .autorag-e2e)
 * @property {string} AUTORAG_HOME
 * @property {string} AUTORAG_CONFIG
 * @property {string} AUTORAG_WORKSPACE
 * @property {string} AUTORAG_SEARCH_PATHS
 * @property {string} AUTORAG_MEMORY_PATH
 * @property {false} globalHomeFallback
 * @property {number} runnerSchemaVersion
 * @property {LiveE2eFingerprint|null} fingerprint
 *
 * @typedef {Object} LiveE2eFingerprint
 * @property {number} runnerSchemaVersion
 * @property {number} corpusVersion
 * @property {string} corpusDigest
 * @property {string} gitCommitSha
 * @property {boolean} gitDirty
 * @property {string} embeddingModel
 * @property {number} embeddingDimension
 * @property {string} embeddingService
 * @property {string} parserConfig
 * @property {string} minSyncConfig
 */


// ── Git helpers ──────────────────────────────────────────────────────

/**
 * @returns {{ sha: string, dirty: boolean }}
 */
function gitHead() {
	try {
		const sha = spawnSync("git", ["rev-parse", "HEAD"], {
			encoding: "utf-8",
			cwd: REPO_ROOT,
		}).stdout?.trim() ?? "";
		const porcelain = spawnSync("git", ["status", "--porcelain"], {
			encoding: "utf-8",
			cwd: REPO_ROOT,
		}).stdout?.trim() ?? "";
		return { sha, dirty: porcelain.length > 0 };
	} catch {
		return { sha: "unknown", dirty: false };
	}
}

// ── Corpus digest ────────────────────────────────────────────────────

/**
 * Compute a combined hash of the corpus manifest version and all entry
 * SHA-256 values.  This lets us detect any corpus-level change.
 * @param {string} root
 * @returns {string}
 */
function computeCorpusDigest(root) {
	const manifestPath = join(root, "corpus", "MANIFEST.json");
	if (!existsSync(manifestPath)) {
		return "no-manifest";
	}
	const raw = readFileSync(manifestPath, "utf-8");
	const manifest = JSON.parse(raw);

	const h = createHash("sha256");
	h.update(String(manifest.version ?? 0));
	if (Array.isArray(manifest.entries)) {
		for (const entry of manifest.entries) {
			h.update(entry.sha256 ?? "");
		}
	}
	return h.digest("hex");
}

// ── Build env ───────────────────────────────────────────────────────

/**
 * Build the full environment descriptor for a given corpus root.
 * @param {string} root - Path to the bootstrapped shared corpus root.
 * @returns {LiveE2eEnv}
 */
export function buildEnv(root) {
	const rootReal = resolve(root);

	// All managed paths are under .autorag-e2e/
	const autoragHome = join(E2E_DIR, "home");
	const autoragConfig = join(E2E_DIR, "config");
	const autoragWorkspace = join(E2E_DIR, "workspace");
	// AUTORAG_SEARCH_PATHS points to the immutable bootstrapped corpus,
	// never a mutable dir under .autorag-e2e.
	const autoragSearchPaths = join(rootReal, "corpus");
	const autoragMemory = join(E2E_DIR, "memory");

	const { sha: gitCommitSha, dirty: gitDirty } = gitHead();

	/** @type {LiveE2eFingerprint} */
	const fingerprint = {
		runnerSchemaVersion: RUNNER_SCHEMA_VERSION,
		corpusVersion: corpusVersion(rootReal),
		corpusDigest: computeCorpusDigest(rootReal),
		gitCommitSha,
		gitDirty,
		embeddingModel: "text-embedding-3-small",
		embeddingDimension: 1536,
		embeddingService: "openai",
		parserConfig: "default",
		minSyncConfig: "default",
	};

	return {
		root: rootReal,
		AUTORAG_HOME: autoragHome,
		AUTORAG_CONFIG: autoragConfig,
		AUTORAG_WORKSPACE: autoragWorkspace,
		AUTORAG_SEARCH_PATHS: autoragSearchPaths,
		AUTORAG_MEMORY_PATH: autoragMemory,
		globalHomeFallback: false,
		runnerSchemaVersion: RUNNER_SCHEMA_VERSION,
		fingerprint,
	};
}

// ── Corpus version helper ────────────────────────────────────────────

/**
 * @param {string} root
 * @returns {number}
 */
function corpusVersion(root) {
	const manifestPath = join(root, "corpus", "MANIFEST.json");
	if (!existsSync(manifestPath)) {
		return 0;
	}
	const raw = readFileSync(manifestPath, "utf-8");
	const parsed = JSON.parse(raw);
	return parsed.version ?? 0;
}

// ── Persisting and loading fingerprint ───────────────────────────────

const FINGERPRINT_DIR = join(E2E_DIR, "fingerprint");
const FINGERPRINT_STATE = join(FINGERPRINT_DIR, "state.json");

/**
 * Persist a fingerprint (and thus .autorag-e2e state) to disk.
 * @param {LiveE2eFingerprint} fp
 */
export function writeFingerprint(fp) {
	mkdirSync(FINGERPRINT_DIR, { recursive: true });
	writeFileSync(FINGERPRINT_STATE, JSON.stringify(fp, null, 2), "utf-8");
}

/**
 * Load a previously-persisted fingerprint, or null if none exists.
 * Returns null when the file is absent.
 * Throws when the file exists but is malformed or unparseable.
 * @returns {LiveE2eFingerprint | null}
 */
export function loadFingerprint() {
	if (!existsSync(FINGERPRINT_STATE)) {
		return null;
	}
	const raw = readFileSync(FINGERPRINT_STATE, "utf-8");
	if (raw.trim().length === 0) {
		throw new Error("live-e2e-fingerprint-mismatch: fingerprint/state.json is empty");
	}
	const parsed = JSON.parse(raw);
	if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
		throw new Error("live-e2e-fingerprint-mismatch: fingerprint/state.json is not a valid fingerprint object");
	}
	return parsed;
}


// ── Cleanup ──────────────────────────────────────────────────────────

/**
 * Delete only the runner-owned clone state (.autorag-e2e directory).
 * Never touches the shared corpus root.
 */
export function removeE2eState() {
	if (existsSync(E2E_DIR)) {
		try {
			rmSync(E2E_DIR, { recursive: true, force: true });
		} catch (/** @type {unknown} */ e) {
			// On macOS APFS, rmSync may throw ENOTEMPTY if a subdirectory
			// was just created/deleted by a concurrent process.  Retry once.
			try {
				rmSync(E2E_DIR, { recursive: true, force: true });
			} catch { /* give up */ }
		}
	}
}