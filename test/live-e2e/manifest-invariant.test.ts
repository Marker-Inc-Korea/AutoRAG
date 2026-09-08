import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { tmpdir } from "node:os";
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
const MANIFEST_PATH = resolve(__dirname, "..", "..", "scripts", "live-e2e", "manifest.json");
const FIXTURE_ROOT = resolve(__dirname, "..", "..", "scripts", "live-e2e");
const RUNNER_PATH = join(FIXTURE_ROOT, "runner.mjs");

function loadManifest(): LiveE2eManifest {
	return JSON.parse(readFileSync(MANIFEST_PATH, "utf-8")) as LiveE2eManifest;
}

function fixtureRoot(): string {
	return FIXTURE_ROOT;
}

// ── Tests ────────────────────────────────────────────────────────────

describe("LiveE2eManifest", () => {
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
			const digest = createHash("sha256").update(content).digest("hex");
			expect(digest, `SHA-256 mismatch for ${entry.path}`).toBe(entry.sha256);
		}
	});

	test("bootstrap creates root with all fixture files", async () => {
		const { bootstrap } = await import(RUNNER_PATH);
		const root = mkdtempSync(join(tmpdir(), "live-e2e-test-bootstrap-"));
		try {
			await bootstrap(root);

			const manifest = loadManifest();
			for (const entry of manifest.entries) {
				expect(
					existsSync(join(root, entry.path)),
					`missing bootstrapped file: ${entry.path}`,
				).toBe(true);
			}
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
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});

	test("verify-corpus refuses missing root", async () => {
		const { verifyCorpus } = await import(RUNNER_PATH);

		const missingRoot = join(tmpdir(), "live-e2e-nonexistent-" + Date.now());
		const result = await verifyCorpus(missingRoot);
		expect(result.ok).toBe(false);
		expect(
			result.errors.some(
				(e: string) => e.includes("missing") || e.includes("MISSING") || e.includes("root"),
			),
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