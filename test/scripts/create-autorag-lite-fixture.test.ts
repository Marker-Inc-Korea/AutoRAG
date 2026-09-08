/**
 * Tests for create-autorag-lite-fixture.mjs
 *
 * Verifies the deterministic fixture generator creates the expected tree,
 * rejects pre-existing roots, handles unknown flags and missing arguments,
 * produces identical output on repeated runs, and cleans up correctly.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";

const FIXTURE_SCRIPT = fileURLToPath(
	new URL("../../scripts/manual-qa/create-autorag-lite-fixture.mjs", import.meta.url),
);

let baseDir: string;
let TEST_ROOT: string;

beforeEach(() => {
	baseDir = mkdtempSync(join(tmpdir(), "autorag-lite-fixture-test-"));
	TEST_ROOT = join(baseDir, "runner");
});

afterEach(() => {
	try {
		rmSync(baseDir, { recursive: true, force: true });
	} catch {
		// ok
	}
});

function runFixture(...args: string[]): string {
	return execFileSync(process.execPath, [FIXTURE_SCRIPT, ...args], { encoding: "utf8" });
}

/** Deterministic sha256 manifest of every file under root: "hash  relativePath" lines, sorted. */
function hashTree(root: string): string {
	const lines: string[] = [];
	const walk = (dir: string): void => {
		for (const entry of readdirSync(dir)) {
			const full = join(dir, entry);
			if (statSync(full).isDirectory()) walk(full);
			else
				lines.push(`${createHash("sha256").update(readFileSync(full)).digest("hex")}  ${full.slice(root.length)}`);
		}
	};
	walk(root);
	return lines.sort().join("\n");
}

describe("create-autorag-lite-fixture", () => {
	it("creates the expected fixture tree and prints JSON", () => {
		const out = runFixture("--root", TEST_ROOT, "--print-json");
		const result = JSON.parse(out.trim());

		expect(result.ok).toBe(true);
		expect(result.root).toBe(TEST_ROOT);
		expect(result.datasources).toEqual(["missing-cli"]);
		expect(result.missingBinaryPath).toBe("/definitely/missing/autorag-lite-cli");
		expect(result.contents["docs/README.md"].size).toBeGreaterThan(100);
		expect(result.contents["workspace/data/notes.txt"].size).toBeGreaterThan(100);
		expect(result.contents["workspace/src/main.ts"].size).toBeGreaterThan(50);

		// Verify files exist on disk
		expect(existsSync(join(TEST_ROOT, "docs", "README.md"))).toBe(true);
		expect(existsSync(join(TEST_ROOT, "workspace", "data", "notes.txt"))).toBe(true);
		expect(existsSync(join(TEST_ROOT, "workspace", "src", "main.ts"))).toBe(true);
		expect(existsSync(join(TEST_ROOT, "config.json"))).toBe(true);
		expect(existsSync(join(TEST_ROOT, "unrefreshed-config.json"))).toBe(true);

		// config.json has the expected structure
		const config = JSON.parse(readFileSync(join(TEST_ROOT, "config.json"), "utf8"));
		expect(config.searchPaths).toBeInstanceOf(Array);
		expect(config.searchPaths).toHaveLength(2);
		expect(config.workspacePath).toContain(TEST_ROOT);
		expect(config.memoryPath).toContain(TEST_ROOT);
		expect(config.minSync.autoInstall).toBe(false);
		expect(config.jikji).toBe(false);
		expect(config.datasources["missing-cli"].type).toBe("obsidian");
		expect(config.datasources["missing-cli"].connector.vaultPath).toBe(join(TEST_ROOT, "workspace"));
		expect(config.datasources["missing-cli"].connector.binaryPath).toBe("/definitely/missing/autorag-lite-cli");
		const datasourceResult = buildDatasourceSkills(config.datasources, TEST_ROOT);
		expect(datasourceResult.unknown).toEqual([]);
		expect(datasourceResult.skills).toHaveLength(1);
	});

	it("rejects pre-existing root", () => {
		runFixture("--root", TEST_ROOT);
		try {
			runFixture("--root", TEST_ROOT, "--print-json");
			expect.unreachable("should have thrown on pre-existing root");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			const stderr = err.stderr ?? "";
			const stdout = err.stdout ?? "";
			const text = stderr + stdout;
			expect(text).toContain("already exists");
		}
	});

	it("exits 2 on unknown flag", () => {
		try {
			runFixture("--root", TEST_ROOT, "--bogus");
			expect.unreachable("should have thrown on unknown flag");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("unknown flag");
		}
	});

	it("exits 2 on missing root argument", () => {
		try {
			runFixture("--root");
			expect.unreachable("should have thrown on missing arg");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("requires a path argument");
		}
	});

	it("rejects empty root string", () => {
		try {
			runFixture("--root", "");
			expect.unreachable("should have thrown on empty root");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("must not be empty");
		}
		// verify no files were written to a bogus path
		expect(existsSync(TEST_ROOT)).toBe(false);
	});

	it("rejects flag-looking root value", () => {
		try {
			runFixture("--root", "--print-json");
			expect.unreachable("should have thrown on flag-looking root");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("looks like a flag");
		}
	});

	it("produces identical output on repeated runs", () => {
		runFixture("--root", TEST_ROOT);
		const files1 = hashTree(TEST_ROOT);

		rmSync(TEST_ROOT, { recursive: true, force: true });
		runFixture("--root", TEST_ROOT);

		const files2 = hashTree(TEST_ROOT);
		expect(files1).toBe(files2);
	});

	it("cleans up via --cleanup flag", () => {
		runFixture("--root", TEST_ROOT);
		expect(existsSync(TEST_ROOT)).toBe(true);

		const out = runFixture("--root", TEST_ROOT, "--cleanup", "--print-json");
		const result = JSON.parse(out.trim());
		expect(result.ok).toBe(true);
		expect(result.cleaned).toBe(true);

		expect(existsSync(TEST_ROOT)).toBe(false);
	});

	it("cleanup of non-existent root reports not-cleaned", () => {
		const missing = join(baseDir, "does-not-exist");
		const out = runFixture("--root", missing, "--cleanup", "--print-json");
		const result = JSON.parse(out.trim());
		expect(result.ok).toBe(true);
		expect(result.cleaned).toBe(false);
	});

	it("human-mode output prints expected lines", () => {
		const out = runFixture("--root", TEST_ROOT);
		expect(out).toContain("fixture created at:");
		expect(out).toContain("docs/README.md");
		expect(out).toContain("config.json");
		expect(out).toContain("datasource: missing-cli");
		expect(out).toContain("cleanup");
		expect(out).toContain("definitely/missing/autorag-lite-cli");
	});
});
