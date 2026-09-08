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
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

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

/**
 * @typedef {'acquired' | { kind: 'held', reason: string }} LockResult
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
	const autoragSearchPaths = join(E2E_DIR, "search-paths");
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
 * @returns {LiveE2eFingerprint | null}
 */
export function loadFingerprint() {
	if (!existsSync(FINGERPRINT_STATE)) {
		return null;
	}
	try {
		const raw = readFileSync(FINGERPRINT_STATE, "utf-8");
		return JSON.parse(raw);
	} catch {
		return null;
	}
}

// ── Lock helpers ─────────────────────────────────────────────────────

const LOCK_DIR = join(E2E_DIR, "locks");
const LOCK_FILE = join(LOCK_DIR, "live-e2e.lock");

/**
 * Filesystem-based exclusive lock using mkdir atomicity.
 *
 * mkdirSync on an existing path throws EEXIST — only one process can create a
 * given directory.  The lock is a fixed-name directory; the first mkdirSync wins,
 * all others lose.  The winner writes its PID inside for diagnostics.
 *
 * @param {number} [holdMs=0] - How long (ms) to hold the lock before releasing.
 * @returns {{ kind: 'acquired' } | { kind: 'held', reason: string }}
 */
export function tryLock(holdMs = 0) {
	// Ensure lock root exists
	mkdirSync(LOCK_DIR, { recursive: true });

	try {
		// Atomic test-and-set on a SINGLE fixed name.
		mkdirSync(LOCK_FILE);

		// Write PID for diagnostics (best-effort — reader may see this before write)
		try {
			writeFileSync(join(LOCK_FILE, "pid"), String(process.pid));
		} catch { /* ignore — lock is ours regardless */ }

		// Hold if requested
		if (holdMs > 0) {
			spawnSync("sleep", [String(Math.ceil(holdMs / 1000))], {
				stdio: "inherit",
				timeout: holdMs + 5000,
			});
		}

		// Release: remove the lock directory
		rmSync(LOCK_FILE, { recursive: true, force: true });

		return { kind: "acquired" };
	} catch (/** @type {unknown} */ e) {
		// mkdir failed — lock held by another process
		let heldBy = "unknown";
		try {
			const pidPath = join(LOCK_FILE, "pid");
			if (existsSync(pidPath)) {
				heldBy = readFileSync(pidPath, "utf-8").trim();
			}
		} catch { /* ignore */ }
		return { kind: "held", reason: "live-e2e-lock-held: lock held by pid " + heldBy };
	}
}

// ── Cleanup ──────────────────────────────────────────────────────────

/**
 * Delete only the runner-owned clone state (.autorag-e2e directory).
 * Never touches the shared corpus root.
 */
export function removeE2eState() {
	if (existsSync(E2E_DIR)) {
		rmSync(E2E_DIR, { recursive: true, force: true });
	}
}