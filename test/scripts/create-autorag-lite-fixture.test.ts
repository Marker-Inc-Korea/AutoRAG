/**
 * Tests for create-autorag-lite-fixture.mjs
 *
 * Verifies the deterministic fixture generator creates the expected tree,
 * rejects pre-existing roots, handles unknown flags and missing arguments,
 * produces identical output on repeated runs, and cleans up correctly.
 */

import { execSync } from "node:child_process";
import { existsSync, readFileSync, rmSync } from "node:fs";
import { afterEach, describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";

const FIXTURE_SCRIPT = new URL("../../scripts/manual-qa/create-autorag-lite-fixture.mjs", import.meta.url).pathname;
const TEST_ROOT = "/tmp/autorag-lite-fixture-test-runner";

afterEach(() => {
	try {
		rmSync(TEST_ROOT, { recursive: true, force: true });
	} catch {
		// ok
	}
});

describe("create-autorag-lite-fixture", () => {
	it("creates the expected fixture tree and prints JSON", () => {
		const out = execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT} --print-json`, {
			encoding: "utf8",
		});
		const result = JSON.parse(out.trim());

		expect(result.ok).toBe(true);
		expect(result.root).toBe(TEST_ROOT);
		expect(result.datasources).toEqual(["missing-cli"]);
		expect(result.missingBinaryPath).toBe("/definitely/missing/autorag-lite-cli");
		expect(result.contents["docs/README.md"].size).toBeGreaterThan(100);
		expect(result.contents["workspace/data/notes.txt"].size).toBeGreaterThan(100);
		expect(result.contents["workspace/src/main.ts"].size).toBeGreaterThan(50);

		// Verify files exist on disk
		expect(existsSync(`${TEST_ROOT}/docs/README.md`)).toBe(true);
		expect(existsSync(`${TEST_ROOT}/workspace/data/notes.txt`)).toBe(true);
		expect(existsSync(`${TEST_ROOT}/workspace/src/main.ts`)).toBe(true);
		expect(existsSync(`${TEST_ROOT}/config.json`)).toBe(true);
		expect(existsSync(`${TEST_ROOT}/unrefreshed-config.json`)).toBe(true);

		// config.json has the expected structure
		const config = JSON.parse(readFileSync(`${TEST_ROOT}/config.json`, "utf8"));
		expect(config.searchPaths).toBeInstanceOf(Array);
		expect(config.searchPaths).toHaveLength(2);
		expect(config.workspacePath).toContain(TEST_ROOT);
		expect(config.memoryPath).toContain(TEST_ROOT);
		expect(config.minSync.autoInstall).toBe(false);
		expect(config.jikji).toBe(false);
		expect(config.datasources["missing-cli"].type).toBe("obsidian");
		expect(config.datasources["missing-cli"].connector.vaultPath).toBe(`${TEST_ROOT}/workspace`);
		expect(config.datasources["missing-cli"].connector.binaryPath).toBe("/definitely/missing/autorag-lite-cli");
		const datasourceResult = buildDatasourceSkills(config.datasources, TEST_ROOT);
		expect(datasourceResult.unknown).toEqual([]);
		expect(datasourceResult.skills).toHaveLength(1);
	});

	it("rejects pre-existing root", () => {
		execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT}`, { encoding: "utf8" });
		try {
			execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT} --print-json`, {
				encoding: "utf8",
			});
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
			execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT} --bogus`, {
				encoding: "utf8",
			});
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
			execSync(`node ${FIXTURE_SCRIPT} --root`, { encoding: "utf8" });
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
			execSync(`node ${FIXTURE_SCRIPT} --root ''`, { encoding: "utf8" });
			expect.unreachable("should have thrown on empty root");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("must not be empty");
		}
		// verify no files were written to a bogus path
		expect(existsSync("/")).toBe(true);
	});

	it("rejects flag-looking root value", () => {
		try {
			execSync(`node ${FIXTURE_SCRIPT} --root --print-json`, { encoding: "utf8" });
			expect.unreachable("should have thrown on flag-looking root");
		} catch (e) {
			const err = e as { stderr?: string; stdout?: string; status?: number };
			expect(err.status).toBe(2);
			const text = (err.stderr ?? "") + (err.stdout ?? "");
			expect(text).toContain("looks like a flag");
		}
	});

	it("produces identical output on repeated runs", () => {
		execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT}`, { encoding: "utf8" });

		const files1 = execSync(`find ${TEST_ROOT} -type f -exec shasum -a 256 {} \\; | sort`, {
			encoding: "utf8",
		});

		rmSync(TEST_ROOT, { recursive: true, force: true });
		execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT}`, { encoding: "utf8" });

		const files2 = execSync(`find ${TEST_ROOT} -type f -exec shasum -a 256 {} \\; | sort`, {
			encoding: "utf8",
		});

		expect(files1).toBe(files2);
	});

	it("cleans up via --cleanup flag", () => {
		execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT}`, { encoding: "utf8" });
		expect(existsSync(TEST_ROOT)).toBe(true);

		const out = execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT} --cleanup --print-json`, {
			encoding: "utf8",
		});
		const result = JSON.parse(out.trim());
		expect(result.ok).toBe(true);
		expect(result.cleaned).toBe(true);

		expect(existsSync(TEST_ROOT)).toBe(false);
	});

	it("cleanup of non-existent root reports not-cleaned", () => {
		const out = execSync(
			`node ${FIXTURE_SCRIPT} --root /tmp/autorag-lite-does-not-exist-xxxx --cleanup --print-json`,
			{
				encoding: "utf8",
			},
		);
		const result = JSON.parse(out.trim());
		expect(result.ok).toBe(true);
		expect(result.cleaned).toBe(false);
	});

	it("human-mode output prints expected lines", () => {
		const out = execSync(`node ${FIXTURE_SCRIPT} --root ${TEST_ROOT}`, { encoding: "utf8" });
		expect(out).toContain("fixture created at:");
		expect(out).toContain("docs/README.md");
		expect(out).toContain("config.json");
		expect(out).toContain("datasource: missing-cli");
		expect(out).toContain("cleanup");
		expect(out).toContain("definitely/missing/autorag-lite-cli");
	});
});
