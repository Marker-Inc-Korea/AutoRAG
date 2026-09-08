/**
 * scripts/live-e2e/runner.mjs — Fixed corpus bootstrap & verification.
 *
 * Commands:
 *   node scripts/live-e2e/runner.mjs bootstrap --root <path>
 *   node scripts/live-e2e/runner.mjs verify-corpus --root <path>
 */

import { createHash } from "node:crypto";
import { copyFileSync, existsSync, mkdirSync, readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// ── Paths ────────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const MANIFEST_PATH = join(__dirname, "manifest.json");

// ── Manifest loader ──────────────────────────────────────────────────

function loadManifest() {
	if (!existsSync(MANIFEST_PATH)) {
		throw new Error("MANIFEST_MISSING: " + MANIFEST_PATH);
	}
	const raw = readFileSync(MANIFEST_PATH, "utf-8");
	const parsed = JSON.parse(raw);

	if (!parsed.version || !Array.isArray(parsed.entries)) {
		throw new Error("MANIFEST_INVALID: manifest.json must have `version` (number) and `entries` (array)");
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

function sha256Of(filePath) {
	const content = readFileSync(filePath);
	return createHash("sha256").update(content).digest("hex");
}

// ── Bootstrap ────────────────────────────────────────────────────────

/**
 * Copy manifest + all fixture files to `root`.
 * Creates `root` if it doesn't exist. Idempotent — overwrites only if files
 * differ.
 * @param {string} root
 */
export async function bootstrap(root) {
	const manifest = loadManifest();

	const rootReal = resolve(root);
	mkdirSync(rootReal, { recursive: true });

	// Copy manifest.json to root
	const manifestDest = join(rootReal, "manifest.json");
	copyManifestFile(MANIFEST_PATH, manifestDest);

	// Copy each entry
	for (const entry of manifest.entries) {
		const src = join(__dirname, entry.path);
		const dest = join(rootReal, entry.path);

		// Ensure parent directory exists
		const parentDir = resolve(dest, "..");
		mkdirSync(parentDir, { recursive: true });

		copyManifestFile(src, dest);
	}
}

function copyManifestFile(src, dest) {
	if (existsSync(dest)) {
		const existingHash = sha256Of(dest);
		const srcHash = sha256Of(src);
		if (existingHash === srcHash) {
			return; // already identical — idempotent skip
		}
	}
	copyFileSync(src, dest);
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

// ── CLI entrypoint ───────────────────────────────────────────────────

async function main() {
	const args = process.argv.slice(2);
	if (args.length === 0) {
		console.error("usage: node scripts/live-e2e/runner.mjs <command> [--root <path>]");
		console.error("  commands: bootstrap, verify-corpus");
		process.exit(2);
	}

	const command = args[0];
	const rootIndex = args.indexOf("--root");
	const root = rootIndex !== -1 ? args[rootIndex + 1] : undefined;

	if (!root) {
		console.error("ERROR: --root <path> is required");
		process.exit(2);
	}

	switch (command) {
		case "bootstrap": {
			await bootstrap(root);
			console.log("BOOTSTRAP_OK: " + resolve(root));
			process.exit(0);
			break;
		}
		case "verify-corpus": {
			const result = await verifyCorpus(root);
			if (result.ok) {
				console.log("VERIFY_OK: corpus integrity confirmed");
				process.exit(0);
			} else {
				for (const err of result.errors) {
					console.error("VERIFY_FAIL: " + err);
				}
				process.exit(1);
			}
			break;
		}
		default: {
			console.error("ERROR: unknown command \"" + command + "\"");
			console.error("  valid commands: bootstrap, verify-corpus");
			process.exit(2);
			break;
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