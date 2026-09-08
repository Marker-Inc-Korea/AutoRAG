/**
 * test/live-e2e/env-fingerprint-lock.test.ts — Todo 2 tests for
 * print-env path isolation, fingerprint mismatch blocking warm mode,
 * lock contention, and cold-mode cleanup boundary.
 */

import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from "vitest";

// ── Paths ───────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const RUNNER_PATH = join(REPO_ROOT, "scripts", "live-e2e", "runner.mjs");
const BOOTSTRAP_PATH = join(REPO_ROOT, "scripts", "live-e2e", "bootstrap.mjs");

// ── Helpers ──────────────────────────────────────────────────────────

/** Run `node runner.mjs <args>` synchronously and return result. */
function runRunnerSync(
	args: readonly string[],
	env?: Record<string, string>,
): { exitCode: number; stdout: string; stderr: string } {
	const { spawnSync } = require("node:child_process") as typeof import("node:child_process");

	const result = spawnSync(process.execPath, [RUNNER_PATH, ...args], {
		encoding: "utf-8",
		env: { ...process.env, ...env },
		timeout: 30_000,
	});

	return {
		exitCode: result.status ?? 1,
		stdout: result.stdout ?? "",
		stderr: result.stderr ?? "",
	};
}

function createTempDir(): string {
	return mkdtempSync(join(tmpdir(), "live-e2e-test-t2-"));
}

/** Bootstrap a corpus into `root` by calling the runner in a child process. */
function bootstrapRoot(root: string): void {
	const result = runRunnerSync(["bootstrap", "--root", root]);
	if (result.exitCode !== 0) {
		throw new Error(`bootstrap failed: ${result.stderr}`);
	}
}

const E2E_DIR = resolve(REPO_ROOT, ".autorag-e2e");

// ── Before/after helpers ─────────────────────────────────────────────

function cleanE2eState(): void {
	if (existsSync(E2E_DIR)) {
		rmSync(E2E_DIR, { recursive: true, force: true });
	}
}

// ── Tests ────────────────────────────────────────────────────────────

describe("print-env", () => {
	let tmpRoot: string;

	beforeEach(() => {
		cleanE2eState();
		tmpRoot = createTempDir();
		bootstrapRoot(tmpRoot);
	});

	afterEach(() => {
		cleanE2eState();
		if (tmpRoot && existsSync(tmpRoot)) {
			rmSync(tmpRoot, { recursive: true, force: true });
		}
	});

	test("print-env --json outputs all five override env vars", () => {
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `exit code, stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		expect(parsed).toHaveProperty("AUTORAG_HOME");
		expect(parsed).toHaveProperty("AUTORAG_CONFIG");
		expect(parsed).toHaveProperty("AUTORAG_WORKSPACE");
		expect(parsed).toHaveProperty("AUTORAG_SEARCH_PATHS");
		expect(parsed).toHaveProperty("AUTORAG_MEMORY_PATH");
	});

	test("every mutable path is beneath the current repo's .autorag-e2e", () => {
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, string>;

		for (const key of ["AUTORAG_HOME", "AUTORAG_CONFIG", "AUTORAG_WORKSPACE", "AUTORAG_MEMORY_PATH"] as const) {
			const val = parsed[key];
			expect(val, `${key} must be a string`).toBeTypeOf("string");
			expect(
				resolve(val!).startsWith(E2E_DIR),
				`${key}=${val!} must be under ${E2E_DIR}`,
			).toBe(true);
		}
	});

	test("AUTORAG_SEARCH_PATHS does not reference the global ~/.autorag", () => {
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		const searchPaths = String(parsed.AUTORAG_SEARCH_PATHS ?? "");
		expect(searchPaths).not.toContain("~/.autorag");
		expect(searchPaths).not.toContain(process.env.HOME + "/.autorag");
	});

	test("globalHomeFallback is false in the env output", () => {
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;
		expect(parsed.globalHomeFallback).toBe(false);
	});

	test("print-env refuses unknown mode", () => {
		const { exitCode, stderr } = runRunnerSync([
			"print-env", "--mode", "unknown", "--json", "--root", tmpRoot,
		]);

		expect(exitCode).toBe(1);
		expect(stderr).toMatch(/mode/);
	});

	test("print-env uses --root or AUTORAG_LIVE_E2E_ROOT, falling back to repo root", () => {
		// Provide explicit --root
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		// AUTORAG_HOME should be under repo's .autorag-e2e regardless of --root
		expect(resolve(String(parsed.AUTORAG_HOME)).startsWith(E2E_DIR)).toBe(true);
	});

	test("print-env prints fingerprint when --mode warm", () => {
		const { exitCode, stdout, stderr } = runRunnerSync([
			"print-env", "--mode", "warm", "--json", "--root", tmpRoot,
		]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;
		expect(parsed).toHaveProperty("fingerprint");
		expect(typeof parsed.fingerprint).toBe("object");
	});
});

// ── Fingerprint / warm-mode mismatch tests ──────────────────────────

describe("fingerprint warm-mode blocking", () => {
	beforeEach(() => {
		cleanE2eState();
	});

	afterEach(() => {
		cleanE2eState();
	});

	test("cold mode from scratch succeeds when shared root is bootstrapped", () => {
		const tmp = createTempDir();
		try {
			bootstrapRoot(tmp);
			const { exitCode, stderr } = runRunnerSync([
				"print-env", "--mode", "cold", "--json", "--root", tmp,
			]);

			expect(exitCode, `cold mode should pass but got stderr: ${stderr}`).toBe(0);
		} finally {
			rmSync(tmp, { recursive: true, force: true });
			cleanE2eState();
		}
	});

	test("warm mode with mismatched fingerprint rejects", () => {
		const tmp = createTempDir();
		try {
			bootstrapRoot(tmp);

			// Store a stale fingerprint in .autorag-e2e that won't match
			mkdirSync(E2E_DIR, { recursive: true });

			const staleFingerprint = {
				runnerSchemaVersion: 0,
				corpusVersion: 999,
				corpusDigest: "stale0000000000000000000000000000000000000",
				gitCommitSha: "abc123",
				gitDirty: false,
				embeddingModel: "old-model",
				embeddingDimension: 0,
				embeddingService: "none",
				parserConfig: "old",
				minSyncConfig: "old",
			};
			mkdirSync(join(E2E_DIR, "fingerprint"), { recursive: true });
			writeFileSync(join(E2E_DIR, "fingerprint", "state.json"), JSON.stringify(staleFingerprint));

			const { exitCode, stderr } = runRunnerSync([
				"print-env", "--mode", "warm", "--json", "--root", tmp,
			]);

			expect(exitCode).toBe(1);
			expect(stderr).toContain("live-e2e-fingerprint-mismatch");
		} finally {
			rmSync(tmp, { recursive: true, force: true });
			cleanE2eState();
		}
	});

	test("warm mode passes with matching fingerprint", () => {
		const tmp = createTempDir();
		try {
			bootstrapRoot(tmp);

			// Run cold first — this should create the state with matching fingerprint
			const coldResult = runRunnerSync([
				"print-env", "--mode", "cold", "--json", "--root", tmp,
			]);
			expect(coldResult.exitCode).toBe(0);

			// Now warm should succeed
			const { exitCode, stderr } = runRunnerSync([
				"print-env", "--mode", "warm", "--json", "--root", tmp,
			]);

			expect(exitCode, `warm mode should pass after cold init, stderr: ${stderr}`).toBe(0);
		} finally {
			rmSync(tmp, { recursive: true, force: true });
			cleanE2eState();
		}
	});
});

// ── Lock contention tests ────────────────────────────────────────────

describe("lock-probe contention", () => {
	beforeEach(() => {
		cleanE2eState();
	});

	afterEach(() => {
		cleanE2eState();
	});

	test("lock-probe acquires and releases a lock", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["lock-probe"]);

		expect(exitCode, `lock-probe should pass, stderr: ${stderr}`).toBe(0);
		expect(stdout).toContain("live-e2e-lock-acquired");
			// Lock directory should be gone after release
		const lockFile = join(E2E_DIR, "locks", "live-e2e.lock");
		expect(existsSync(lockFile), "lock file should be released").toBe(false);
	});

	test("two concurrent lock-probes yield exactly one failure", async () => {
		const { execSync } = await import("node:child_process");
		const { existsSync } = await import("node:fs");

		// Ensure clean lock state before starting
		const lockDir = join(E2E_DIR, "locks");
		if (existsSync(lockDir)) {
			const { rmSync } = await import("node:fs");
			rmSync(lockDir, { recursive: true, force: true });
		}

		// Start two concurrent lock-probe processes via shell backgrounding.
		// Each has a 1-second hold. Collect outputs from both.
		const runner = RUNNER_PATH;
		const nodeExe = process.execPath;

		// Write a temp script that runs both and captures their outputs
		// Use mkdtempSync from fs to create temp dir
		const { mkdtempSync } = await import("node:fs");
		const tmpDir = mkdtempSync((await import("node:os").then((o) => o.tmpdir())) + "/live-e2e-locktest-");
		const { writeFileSync } = await import("node:fs");
		writeFileSync(join(tmpDir, "run.sh"), [
			"#!/bin/sh",
			`"${nodeExe}" "${runner}" lock-probe --hold-ms 1000 > "${tmpDir}/out1" 2>"${tmpDir}/err1" &`,
			`"${nodeExe}" "${runner}" lock-probe --hold-ms 1000 > "${tmpDir}/out2" 2>"${tmpDir}/err2" &`,
			"wait",
		].join("\n"));
		const { chmodSync } = await import("node:fs");
		chmodSync(join(tmpDir, "run.sh"), 0o755);

		execSync(join(tmpDir, "run.sh"), { timeout: 10_000 });

		const out1 = (await import("node:fs")).readFileSync(join(tmpDir, "out1"), "utf-8").trim();
		const out2 = (await import("node:fs")).readFileSync(join(tmpDir, "out2"), "utf-8").trim();

		const acquired = [out1, out2].filter((o) => o.includes("live-e2e-lock-acquired"));
		const held = [out1, out2].filter((o) => o.includes("live-e2e-lock-held"));

		expect(acquired.length, "exactly one should acquire the lock").toBe(1);
		expect(held.length, "exactly one should be held").toBe(1);

		// Clean up tmp dir
		const { rmSync } = await import("node:fs");
		rmSync(tmpDir, { recursive: true, force: true });

		// Lock should be released after both exit
		const lockFile = join(E2E_DIR, "locks", "live-e2e.lock");
		expect(existsSync(lockFile), "lock file should be released").toBe(false);
	});

	test("lock-probe --hold-ms rejects non-numeric value", () => {
		const { exitCode, stderr } = runRunnerSync(["lock-probe", "--hold-ms", "abc"]);

		expect(exitCode).toBe(1);
		expect(stderr).toContain("hold-ms");
	});
});

// ── Cold-mode cleanup boundary tests ────────────────────────────────

describe("cold-mode cleanup boundary", () => {
	const tmp = createTempDir();

	beforeAll(() => {
		bootstrapRoot(tmp);
	});

	beforeEach(() => {
		cleanE2eState();
	});

	afterAll(() => {
		if (existsSync(tmp)) {
			rmSync(tmp, { recursive: true, force: true });
		}
		cleanE2eState();
	});

	test("cold mode deletes .autorag-e2e and rebuilds", () => {
		// Create some state within .autorag-e2e
		mkdirSync(E2E_DIR, { recursive: true });
		writeFileSync(join(E2E_DIR, "some-state.bin"), "stale\n");
		expect(existsSync(join(E2E_DIR, "some-state.bin"))).toBe(true);

		// Run cold mode
		const { exitCode, stderr } = runRunnerSync([
			"print-env", "--mode", "cold", "--json", "--root", tmp,
		]);

		expect(exitCode, `cold mode should pass, stderr: ${stderr}`).toBe(0);
		// .autorag-e2e should still exist (it was recreated by cold mode)
		expect(existsSync(E2E_DIR)).toBe(true);
		// But stale-state.bin should be gone
		expect(existsSync(join(E2E_DIR, "some-state.bin"))).toBe(false);
	});

	test("cold mode never deletes the shared corpus root", () => {
		expect(existsSync(join(tmp, "corpus", "MANIFEST.json")), "shared corpus must exist").toBe(true);

		// Create .autorag-e2e state
		mkdirSync(E2E_DIR, { recursive: true });
		writeFileSync(join(E2E_DIR, "state.bin"), "data\n");

		runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmp]);

		// Shared root corpus must survive
		expect(existsSync(join(tmp, "corpus", "MANIFEST.json")), "shared corpus must survive cold mode").toBe(true);
	});

	test("refuse when shared root is not bootstrapped", () => {
		const emptyRoot = createTempDir();
		try {
			// Empty root — no corpus
			const { exitCode, stderr } = runRunnerSync([
				"print-env", "--mode", "cold", "--json", "--root", emptyRoot,
			]);

			expect(exitCode).toBe(1);
			expect(stderr).toContain("live-e2e-root-not-bootstrapped");
		} finally {
			rmSync(emptyRoot, { recursive: true, force: true });
		}
	});
});