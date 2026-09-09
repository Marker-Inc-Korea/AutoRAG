/**
 * scripts/live-e2e/bootstrap.mjs — Corpus bootstrap & verification.
 *
 * Extracted from runner.mjs so both the runner CLI and tests can import it.
 */

import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// ── Paths ────────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const MANIFEST_PATH = join(__dirname, "corpus", "MANIFEST.json");

// ── Manifest loader ──────────────────────────────────────────────────

function loadManifest() {
	if (!existsSync(MANIFEST_PATH)) {
		throw new Error("MANIFEST_MISSING: " + MANIFEST_PATH);
	}
	const raw = readFileSync(MANIFEST_PATH, "utf-8");
	const parsed = JSON.parse(raw);

	if (!parsed.version || !Array.isArray(parsed.entries)) {
		throw new Error("MANIFEST_INVALID: MANIFEST.json must have `version` (number) and `entries` (array)");
	}
	if (parsed.version !== 1) {
		throw new Error("MANIFEST_VERSION_UNSUPPORTED: version " + parsed.version);
	}
	for (const entry of parsed.entries) {
		if (!entry.path || !entry.sha256) {
			throw new Error("MANIFEST_INVALID_ENTRY: each entry must have `path` and `sha256`");
		}
	}
	return parsed;
}

// ── SHA-256 helper ───────────────────────────────────────────────────

function toLf(bytes) {
	if (!bytes.includes(0x0d)) {
		return bytes;
	}
	return Buffer.from(bytes.toString("utf8").replace(/\r\n/g, "\n").replace(/\r/g, "\n"), "utf8");
}

function sha256OfBytes(bytes) {
	return createHash("sha256").update(bytes).digest("hex");
}

function sha256Of(filePath) {
	return sha256OfBytes(readFileSync(filePath));
}

// ── Bootstrap ────────────────────────────────────────────────────────

/**
 * Copy manifest + all fixture files to `root`.
 * Creates `root` if it doesn't exist. Idempotent — overwrites only if files
 * differ. All bootstrapped files are set to read-only (0444).
 * @param {string} root
 */
export async function bootstrap(root) {
	const manifest = loadManifest();

	const rootReal = resolve(root);
	mkdirSync(rootReal, { recursive: true });

	// Copy MANIFEST.json to root/corpus/MANIFEST.json
	const manifestDest = join(rootReal, "corpus", "MANIFEST.json");
	const manifestParent = resolve(manifestDest, "..");
	mkdirSync(manifestParent, { recursive: true });
	copyReadOnly(MANIFEST_PATH, manifestDest);

	// Copy each entry
	for (const entry of manifest.entries) {
		const src = join(__dirname, entry.path);
		const dest = join(rootReal, entry.path);

		// Ensure parent directory exists
		const parentDir = resolve(dest, "..");
		mkdirSync(parentDir, { recursive: true });

		copyReadOnly(src, dest);
	}
}

/**
 * Copy `src` to `dest`, making `dest` writable first if needed, then
 * set `dest` to read-only (0444). Skips copy if content is already identical.
 * @param {string} src
 * @param {string} dest
 */
function copyReadOnly(src, dest) {
	// Normalize CRLF so Windows checkouts still match the LF SHA-256 contract.
	const srcBytes = toLf(readFileSync(src));
	if (existsSync(dest) && sha256Of(dest) === sha256OfBytes(srcBytes)) {
		chmodSync(dest, 0o444);
		return;
	}

	if (existsSync(dest)) {
		chmodSync(dest, 0o644);
	}

	writeFileSync(dest, srcBytes);
	chmodSync(dest, 0o444);
}

// ── Verify corpus ────────────────────────────────────────────────────

/**
 * Verify that `root` contains a faithful copy of the fixture corpus.
 * Returns { ok: true } on success, or { ok: false, errors: [...] } on
 * any mismatch or missing file.
 * @param {string} root
 * @returns {Promise<{ok: boolean, errors: string[]}>}
 */
export async function verifyCorpus(root) {
	const errors = [];

	// Guard: root must exist
	const rootReal = resolve(root);
	if (!existsSync(rootReal)) {
		return { ok: false, errors: ["MISSING_ROOT: " + rootReal] };
	}

	let manifest;
	try {
		manifest = loadManifest();
	} catch (e) {
		return { ok: false, errors: [e.message] };
	}

	// Verify each entry
	for (const entry of manifest.entries) {
		const fixturePath = join(rootReal, entry.path);

		if (!existsSync(fixturePath)) {
			errors.push("MISSING_FILE: " + entry.path);
			continue;
		}

		const actualHash = sha256Of(fixturePath);
		if (actualHash !== entry.sha256) {
			errors.push(
				"DRIFT_DETECTED: " + entry.path
				+ " — expected " + entry.sha256
				+ ", got " + actualHash,
			);
		}
	}

	return errors.length === 0 ? { ok: true, errors: [] } : { ok: false, errors };
}