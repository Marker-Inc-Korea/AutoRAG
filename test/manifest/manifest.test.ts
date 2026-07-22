import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadManifest, loadManifests } from "../../src/manifest/loader.ts";

const FIXTURES_DIR = "test/fixtures/manifests";

describe("loadManifest", () => {
	it("loads a valid YAML manifest", () => {
		const manifest = loadManifest(join(FIXTURES_DIR, "vector-codebase.yaml"));
		expect(manifest.name).toBe("codebase-vectors");
		expect(manifest.type).toBe("vector");
		expect(manifest.description).toBeTruthy();
	});

	it("loads a valid JSON manifest", () => {
		const manifest = loadManifest(join(FIXTURES_DIR, "bm25-docs.json"));
		expect(manifest.name).toBe("documentation-index");
		expect(manifest.type).toBe("bm25");
	});
});

describe("loadManifests", () => {
	it("loads all valid manifests from a directory", () => {
		const manifests = loadManifests(FIXTURES_DIR);
		// Should load vector-codebase.yaml and bm25-docs.json, skip invalid.yaml
		const names = manifests.map((m) => m.name);
		expect(names).toContain("codebase-vectors");
		expect(names).toContain("documentation-index");
	});

	it("skips invalid manifests without crashing", () => {
		const manifests = loadManifests(FIXTURES_DIR);
		// invalid.yaml is missing required fields, should be skipped
		const names = manifests.map((m) => m.name);
		expect(names).not.toContain(undefined);
		// All returned manifests have required fields
		for (const m of manifests) {
			expect(m.name).toBeTruthy();
			expect(m.description).toBeTruthy();
			expect(m.type).toBeTruthy();
		}
	});

	it("returns empty array for missing directory", () => {
		const manifests = loadManifests("/nonexistent/path/that/does/not/exist");
		expect(manifests).toEqual([]);
	});

	it("returns empty array for empty directory", () => {
		const tmpDir = join(tmpdir(), `autorag-test-${Date.now()}`);
		mkdirSync(tmpDir, { recursive: true });
		try {
			const manifests = loadManifests(tmpDir);
			expect(manifests).toEqual([]);
		} finally {
			rmSync(tmpDir, { recursive: true });
		}
	});
});
