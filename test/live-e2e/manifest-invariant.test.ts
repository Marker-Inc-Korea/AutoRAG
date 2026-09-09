import { createHash } from "node:crypto";
import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, test } from "vitest";

// ── The manifest contract ─────────────────────────────────────────────

interface ManifestEntry {
	path: string;
	sha256: string;
}

interface LiveE2eManifest {
	version: number;
	entries: readonly ManifestEntry[];
}

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const FIXTURE_ROOT = resolve(__dirname, "..", "..", "scripts", "live-e2e");
const MANIFEST_PATH = join(FIXTURE_ROOT, "corpus", "MANIFEST.json");
const LEGACY_MANIFEST_PATH = join(FIXTURE_ROOT, "manifest.json");
const RUNNER_PATH = join(FIXTURE_ROOT, "runner.mjs");

/** True when the file's mode has no write bits set (0444 or stricter). */
function isReadOnly(path: string): boolean {
	return (statSync(path).mode & 0o222) === 0;
}

function loadManifest(): LiveE2eManifest {
	return JSON.parse(readFileSync(MANIFEST_PATH, "utf-8")) as LiveE2eManifest;
}

function fixtureRoot(): string {
	return FIXTURE_ROOT;
}

// ── Tests ────────────────────────────────────────────────────────────

describe("LiveE2eManifest", () => {
	test("manifest lives at corpus/MANIFEST.json", () => {
		expect(existsSync(MANIFEST_PATH), `missing ${MANIFEST_PATH}`).toBe(true);

		const manifest = loadManifest();
		expect(manifest.version).toBe(1);
	});

	test("no legacy manifest.json remains at fixture root", () => {
		expect(existsSync(LEGACY_MANIFEST_PATH)).toBe(false);
	});

	test("has version 1", () => {
		const manifest = loadManifest();
		expect(manifest.version).toBe(1);
	});

	test("every entry path is relative and points to an existing file", () => {
		const manifest = loadManifest();
		const root = fixtureRoot();

		for (const entry of manifest.entries) {
			expect(entry.path).not.toMatch(/^\//);
			expect(entry.path).not.toMatch(/\.\./);

			const fullPath = join(root, entry.path);
			expect(existsSync(fullPath), `missing fixture: ${entry.path}`).toBe(true);

			const stat = readFileSync(fullPath);
			expect(stat.length).toBeGreaterThan(0);
		}
	});

	test("every entry SHA-256 matches the file content", () => {
		const manifest = loadManifest();
		const root = fixtureRoot();

		for (const entry of manifest.entries) {
			const fullPath = join(root, entry.path);
			const content = readFileSync(fullPath);
			const normalized = Buffer.from(
				content.toString("utf8").replace(/\r\n/g, "\n").replace(/\r/g, "\n"),
				"utf8",
			);
			const digest = createHash("sha256").update(normalized).digest("hex");
			expect(digest, `SHA-256 mismatch for ${entry.path}`).toBe(entry.sha256);
		}
	});

	test("bootstrap creates root with all fixture files and bootstrapped manifest", async () => {
		const { bootstrap } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-bootstrap-"));
		try {
			await bootstrap(root);

			expect(existsSync(join(root, "corpus", "MANIFEST.json")), "bootstrapped corpus/MANIFEST.json missing").toBe(
				true,
			);

			const manifest = loadManifest();
			for (const entry of manifest.entries) {
				expect(existsSync(join(root, entry.path)), `missing bootstrapped file: ${entry.path}`).toBe(true);
			}
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("bootstrap produces read-only bootstrapped files (no write bits)", async () => {
		const { bootstrap } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-dreadonly-"));
		try {
			await bootstrap(root);

			expect(isReadOnly(join(root, "corpus", "MANIFEST.json")), "manifest must be read-only").toBe(true);

			const manifest = loadManifest();
			for (const entry of manifest.entries) {
				expect(isReadOnly(join(root, entry.path)), `bootstrapped file ${entry.path} must be read-only`).toBe(true);
			}
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("re-bootstrap is idempotent and preserves read-only mode", async () => {
		const { bootstrap } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-idempotent-"));
		try {
			await bootstrap(root);

			// Make one file writable, as if externally modified, then re-bootstrap.
			const manifest = loadManifest();
			const firstEntry = manifest.entries[0]!;
			const filePath = join(root, firstEntry.path);
			chmodSync(filePath, 0o644);

			await bootstrap(root);

			expect(isReadOnly(filePath), "re-bootstrap must restore read-only mode").toBe(true);

			const result = await verifyCorpus(root);
			expect(result.ok).toBe(true);
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("verify-corpus passes on bootstrapped root", async () => {
		const { bootstrap, verifyCorpus } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-verify-pass-"));
		try {
			await bootstrap(root);

			const result = await verifyCorpus(root);
			expect(result.ok).toBe(true);
			expect(result.errors).toHaveLength(0);

			const manifest = loadManifest();
			for (const entry of manifest.entries) {
				const bytes = readFileSync(join(root, entry.path));
				expect(bytes.includes(0x0d), `bootstrapped ${entry.path} must be LF`).toBe(false);
			}
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("verify-corpus refuses missing root", async () => {
		const { verifyCorpus } = await import(RUNNER_PATH);

		const missingRoot = join(tmpdir(), `live-e2e-nonexistent-${Date.now()}`);
		const result = await verifyCorpus(missingRoot);
		expect(result.ok).toBe(false);
		expect(
			result.errors.some((e: string) => e.includes("missing") || e.includes("MISSING") || e.includes("root")),
		).toBe(true);
	});

	test("verify-corpus refuses drifted file (modified byte)", async () => {
		const { bootstrap, verifyCorpus } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-drift-"));
		try {
			await bootstrap(root);

			const manifest = loadManifest();
			const firstEntry = manifest.entries[0]!;
			const filePath = join(root, firstEntry.path);
			chmodSync(filePath, 0o644); // restore write access to simulate external drift
			const content = readFileSync(filePath);
			content[0] = content[0]! ^ 0xff;
			writeFileSync(filePath, content);

			const result = await verifyCorpus(root);
			expect(result.ok).toBe(false);
			expect(result.errors.some((e: string) => e.includes(firstEntry.path))).toBe(true);
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});
});

// Local helper to avoid importing runner at module scope.
async function verifyCorpus(root: string) {
	const { verifyCorpus: verify } = await import(RUNNER_PATH);
	return verify(root) as Promise<{ ok: boolean; errors: string[] }>;
}
