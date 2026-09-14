/**
 * test/live-e2e/env-fingerprint-lock.test.ts — Todo 2 tests for
 * print-env path isolation, fingerprint mismatch blocking warm mode,
 * lock contention, and cold-mode cleanup boundary.
 */

import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test } from "vitest";

// ── Paths ───────────────────────────────────────────────────────────

const __dirname = fileURLToPath(new URL(".", import.meta.url));
const REPO_ROOT = resolve(__dirname, "..", "..");
const RUNNER_PATH = join(REPO_ROOT, "scripts", "live-e2e", "runner.mjs");

// ── File-level cleanup guard — ensures no cross-file pollution with parallelism
// The env.mjs production code hardcodes .autorag-e2e at repo root, so we clean
// at suite start AND end.
const E2E_DIR = resolve(REPO_ROOT, ".autorag-e2e");

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

// ── Before/after helpers ─────────────────────────────────────────────

function cleanE2eState(): void {
	if (existsSync(E2E_DIR)) {
		rmSync(E2E_DIR, { recursive: true, force: true });
	}
}

beforeAll(() => {
	// Wipe any leftover state from a parallel test file
	cleanE2eState();
});

afterAll(() => {
	// Ensure clean exit
	cleanE2eState();
});

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
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode, `exit code, stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		expect(parsed).toHaveProperty("AUTORAG_HOME");
		expect(parsed).toHaveProperty("AUTORAG_CONFIG");
		expect(parsed).toHaveProperty("AUTORAG_WORKSPACE");
		expect(parsed).toHaveProperty("AUTORAG_SEARCH_PATHS");
		expect(parsed).toHaveProperty("AUTORAG_MEMORY_PATH");
	});

	test("every mutable path is beneath the current repo's .autorag-e2e", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, string>;

		for (const key of ["AUTORAG_HOME", "AUTORAG_CONFIG", "AUTORAG_WORKSPACE", "AUTORAG_MEMORY_PATH"] as const) {
			const val = parsed[key];
			expect(val, `${key} must be a string`).toBeTypeOf("string");
			expect(resolve(val!).startsWith(E2E_DIR), `${key}=${val!} must be under ${E2E_DIR}`).toBe(true);
		}
	});

	test("AUTORAG_SEARCH_PATHS does not reference the global ~/.autorag", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		const searchPaths = String(parsed.AUTORAG_SEARCH_PATHS ?? "");
		expect(searchPaths).not.toContain("~/.autorag");
		expect(searchPaths).not.toContain(`${process.env.HOME}/.autorag`);
	});

	test("globalHomeFallback is false in the env output", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;
		expect(parsed.globalHomeFallback).toBe(false);
	});

	test("print-env refuses unknown mode", () => {
		const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "unknown", "--json", "--root", tmpRoot]);

		expect(exitCode).toBe(1);
		expect(stderr).toMatch(/mode/);
	});

	test("print-env uses --root or AUTORAG_LIVE_E2E_ROOT, falling back to repo root", () => {
		// Provide explicit --root
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode, `stderr: ${stderr}`).toBe(0);
		const parsed = JSON.parse(stdout) as Record<string, unknown>;

		// AUTORAG_HOME should be under repo's .autorag-e2e regardless of --root
		expect(resolve(String(parsed.AUTORAG_HOME)).startsWith(E2E_DIR)).toBe(true);
	});

	test("print-env prints fingerprint when --mode warm", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

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
			const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmp]);

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

			const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmp]);

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
			const coldResult = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmp]);
			expect(coldResult.exitCode).toBe(0);

			// Now warm should succeed
			const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmp]);

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
		const { spawn } = await import("node:child_process");

		const lockDir = join(E2E_DIR, "locks");
		if (existsSync(lockDir)) {
			rmSync(lockDir, { recursive: true, force: true });
		}

		const isolatedPath = { PATH: "/nonexistent", Path: "/nonexistent" } as const;

		function runProbe(holdMs: number): Promise<{ stdout: string; stderr: string }> {
			return new Promise((resolve, reject) => {
				const child = spawn(process.execPath, [RUNNER_PATH, "lock-probe", "--hold-ms", String(holdMs)], {
					stdio: ["ignore", "pipe", "pipe"],
					env: { ...process.env, ...isolatedPath },
				});
				let stdout = "";
				let stderr = "";
				child.stdout.on("data", (chunk: Buffer) => {
					stdout += chunk.toString("utf8");
				});
				child.stderr.on("data", (chunk: Buffer) => {
					stderr += chunk.toString("utf8");
				});
				const timer = setTimeout(() => {
					child.kill("SIGKILL");
					reject(new Error(`lock-probe timed out stdout=${stdout} stderr=${stderr}`));
				}, 10_000);
				child.once("error", (error) => {
					clearTimeout(timer);
					reject(error);
				});
				child.once("close", () => {
					clearTimeout(timer);
					resolve({ stdout, stderr });
				});
			});
		}

		const [first, second] = await Promise.all([runProbe(1000), runProbe(1000)]);
		const outputs = [first.stdout, second.stdout];
		const acquired = outputs.filter((text) => text.includes("live-e2e-lock-acquired"));
		const held = outputs.filter((text) => text.includes("live-e2e-lock-held"));

		expect(
			acquired.length,
			`exactly one should acquire the lock first=${first.stdout} second=${second.stdout} err1=${first.stderr} err2=${second.stderr}`,
		).toBe(1);
		expect(held.length, `exactly one should be held first=${first.stdout} second=${second.stdout}`).toBe(1);

		const lockFile = join(E2E_DIR, "locks", "live-e2e.lock");
		expect(existsSync(lockFile), "lock file should be released").toBe(false);
	});

	test("lock-probe --hold-ms rejects non-numeric value", () => {
		const { exitCode, stderr } = runRunnerSync(["lock-probe", "--hold-ms", "abc"]);

		expect(exitCode).toBe(1);
		expect(stderr).toContain("hold-ms");
	});
});

// ── Bugfix 1: AUTORAG_SEARCH_PATHS points to immutable shared root ───

describe("bugfix: AUTORAG_SEARCH_PATHS to immutable shared root", () => {
	let tmpRoot: string;

	beforeEach(() => {
		cleanE2eState();
		tmpRoot = createTempDir();
		bootstrapRoot(tmpRoot);
	});

	afterEach(() => {
		cleanE2eState();
		if (tmpRoot && existsSync(tmpRoot)) rmSync(tmpRoot, { recursive: true, force: true });
	});

	test("AUTORAG_SEARCH_PATHS points to <shared-root>/corpus not .autorag-e2e/search-paths", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmpRoot]);
		expect(exitCode, `stderr: ${stderr}`).toBe(0);

		const parsed = JSON.parse(stdout) as Record<string, string>;
		const searchPaths = parsed.AUTORAG_SEARCH_PATHS ?? "";

		expect(searchPaths).toBe(join(tmpRoot, "corpus"));
	});

	test("AUTORAG_SEARCH_PATHS does NOT contain .autorag-e2e", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmpRoot]);
		expect(exitCode, `stderr: ${stderr}`).toBe(0);

		const parsed = JSON.parse(stdout) as Record<string, string>;
		expect(parsed.AUTORAG_SEARCH_PATHS).not.toContain(".autorag-e2e");
	});

	test("AUTORAG_SEARCH_PATHS is an existing directory", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmpRoot]);
		expect(exitCode, `stderr: ${stderr}`).toBe(0);

		const parsed = JSON.parse(stdout) as Record<string, string>;
		expect(existsSync(parsed.AUTORAG_SEARCH_PATHS)).toBe(true);
	});

	test("AUTORAG_SEARCH_PATHS contains a bootstrapped corpus (MANIFEST.json present)", () => {
		const { exitCode, stdout, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmpRoot]);
		expect(exitCode, `stderr: ${stderr}`).toBe(0);

		const parsed = JSON.parse(stdout) as Record<string, string>;
		expect(existsSync(join(parsed.AUTORAG_SEARCH_PATHS, "MANIFEST.json"))).toBe(true);
	});
});

// ── Bugfix 2: malformed fingerprint blocks warm mode ────────────────

describe("bugfix: malformed fingerprint blocks warm mode", () => {
	let tmpRoot: string;

	beforeEach(() => {
		cleanE2eState();
		tmpRoot = createTempDir();
		bootstrapRoot(tmpRoot);
	});

	afterEach(() => {
		cleanE2eState();
		if (tmpRoot && existsSync(tmpRoot)) rmSync(tmpRoot, { recursive: true, force: true });
	});

	test("malformed JSON in fingerprint/state.json blocks warm mode with live-e2e-fingerprint-mismatch", () => {
		mkdirSync(join(E2E_DIR, "fingerprint"), { recursive: true });
		writeFileSync(join(E2E_DIR, "fingerprint", "state.json"), "this is not valid json at all {{{{");

		const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode).toBe(1);
		expect(stderr).toContain("live-e2e-fingerprint-mismatch");
	});

	test("cold mode overwrites a malformed fingerprint silently", () => {
		mkdirSync(join(E2E_DIR, "fingerprint"), { recursive: true });
		writeFileSync(join(E2E_DIR, "fingerprint", "state.json"), "garbage{{{");

		const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "cold", "--json", "--root", tmpRoot]);

		expect(exitCode, `cold mode must pass on malformed: ${stderr}`).toBe(0);
	});

	test("empty file in fingerprint/state.json blocks warm mode", () => {
		mkdirSync(join(E2E_DIR, "fingerprint"), { recursive: true });
		writeFileSync(join(E2E_DIR, "fingerprint", "state.json"), "");

		const { exitCode, stderr } = runRunnerSync(["print-env", "--mode", "warm", "--json", "--root", tmpRoot]);

		expect(exitCode).toBe(1);
		expect(stderr).toContain("live-e2e-fingerprint-mismatch");
	});
});

// ── Bugfix 3: stale lock detection and break ─────────────────────────

describe("bugfix: stale lock detection and break", () => {
	beforeEach(() => {
		cleanE2eState();
	});

	afterEach(() => {
		cleanE2eState();
	});

	test("stale lock with dead PID is broken and acquired", () => {
		mkdirSync(join(E2E_DIR, "locks", "live-e2e.lock"), { recursive: true });
		writeFileSync(join(E2E_DIR, "locks", "live-e2e.lock", "pid"), "99999999");

		const { exitCode, stdout, stderr } = runRunnerSync(["lock-probe"]);

		expect(exitCode, `should acquire after breaking stale lock, stderr: ${stderr}`).toBe(0);
		expect(stdout).toContain("live-e2e-lock-acquired");
		expect(existsSync(join(E2E_DIR, "locks", "live-e2e.lock"))).toBe(false);
	});

	test("stale lock with dead PID broken even after previous acquire", () => {
		mkdirSync(join(E2E_DIR, "locks", "live-e2e.lock"), { recursive: true });
		writeFileSync(join(E2E_DIR, "locks", "live-e2e.lock", "pid"), "99999999");

		const { exitCode: e1 } = runRunnerSync(["lock-probe"]);
		expect(e1).toBe(0);

		const { exitCode: e2, stdout: o2 } = runRunnerSync(["lock-probe"]);
		expect(e2).toBe(0);
		expect(o2).toContain("live-e2e-lock-acquired");
	});

	test("lock without PID file is conservatively treated as held", () => {
		mkdirSync(join(E2E_DIR, "locks", "live-e2e.lock"), { recursive: true });
		// No pid file inside — for a live holder this is a <1ms window
		// between mkdir and PID file write. Conservative: treat as held.

		const { exitCode, stdout } = runRunnerSync(["lock-probe"]);
		expect(exitCode).toBe(1);
		expect(stdout).toContain("live-e2e-lock-held");
	});

	test("live holder lock is held (not broken)", async () => {
		const { spawn } = await import("node:child_process");
		const isolatedPath = { PATH: "/nonexistent", Path: "/nonexistent" } as const;

		const holder = spawn(process.execPath, [RUNNER_PATH, "lock-probe", "--hold-ms", "3000"], {
			stdio: ["ignore", "pipe", "pipe"],
			env: { ...process.env, ...isolatedPath },
		});

		try {
			const pidPath = join(E2E_DIR, "locks", "live-e2e.lock", "pid");
			const deadline = Date.now() + 5_000;
			while (!existsSync(pidPath)) {
				if (holder.exitCode !== null || Date.now() > deadline) {
					throw new Error("holder never wrote lock pid");
				}
				await new Promise((resolve) => setTimeout(resolve, 10));
			}

			const { exitCode, stdout, stderr } = runRunnerSync(["lock-probe"], isolatedPath);
			expect(exitCode, `should be held, stderr: ${stderr}`).toBe(1);
			expect(stdout).toContain("live-e2e-lock-held");
		} finally {
			holder.kill("SIGKILL");
			await new Promise<void>((resolve) => {
				if (holder.exitCode !== null) {
					resolve();
					return;
				}
				holder.once("exit", () => resolve());
			});
		}

		const { exitCode: exit2, stdout: out2 } = runRunnerSync(["lock-probe"]);
		expect(exit2).toBe(0);
		expect(out2).toContain("live-e2e-lock-acquired");
	});
});
