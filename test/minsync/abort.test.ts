import { spawn } from "node:child_process";
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { readRefreshProgress } from "../../src/agent/refresh-progress.ts";
import { MinSyncClient, MinSyncVectorMethod, minSyncConfigPath } from "../../src/minsync/index.ts";
import { spawnProcess } from "../../src/minsync/process.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

const posixOnly = process.platform === "win32" ? describe.skip : describe;

let root: string;
let workspace: string;
let binary: string;
let descendantPidFile: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-abort-"));
	workspace = join(root, ".autorag", "minsync");
	binary = join(root, "fake-minsync.mjs");
	descendantPidFile = join(root, "descendant.pid");
	mkdirSync(join(workspace, ".minsync"), { recursive: true });
	mkdirSync(join(root, "docs"), { recursive: true });
	writeFileSync(join(root, "docs", "a.txt"), "alpha\n");
});

afterEach(() => {
	if (existsSync(descendantPidFile)) {
		const pid = Number(readFileSync(descendantPidFile, "utf8"));
		if (isAlive(pid)) process.kill(pid, "SIGKILL");
	}
	rmSync(root, { recursive: true, force: true });
});

function isAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

/** A reaped descendant can still answer `kill(pid, 0)` for a moment as an exiting process under load. */
async function expectDead(pid: number): Promise<void> {
	await vi.waitFor(() => expect(isAlive(pid)).toBe(false), { timeout: 5_000 });
}

/**
 * A fake `minsync`: `check` passes; `sync` starts a descendant that inherits
 * the stdio pipes (a launcher, like the real tool's runtime), records the
 * descendant's pid, then never finishes on its own.
 */
function writeHangingSyncBinary(): void {
	writeFileSync(
		binary,
		`#!/usr/bin/env node
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { createRequire } from "node:module";
const args = process.argv.slice(2);
if (args[0] === "stage") { createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({stateDir: args[args.indexOf("--state-dir") + 1]}); } else {
if (args[0] === "init") {
  mkdirSync(join(process.cwd(), ".minsync"), { recursive: true });
  writeFileSync(join(process.cwd(), ".minsync", "config.toml"), "[embedder]\\nid = \\"openai\\"\\n");
  process.exit(0);
}
if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}
if (args[0] === "sync") {
  spawn(process.execPath, ["-e", ${JSON.stringify(
		`require("node:fs").writeFileSync(${JSON.stringify(descendantPidFile)}, String(process.pid)); setInterval(() => {}, 1 << 30);`,
  )}], { stdio: "inherit" });
  setInterval(() => {}, 1 << 30);
} else {
  process.exit(2);
}
}
`,
	);
	chmodSync(binary, 0o755);
}

/** `heldAt` picks the command that hits the lock; real MinSync 0.4.8 exits 3 from both `check` and `sync`. */
function writeLockHeldBinary(heldAt: "check" | "sync" = "sync"): void {
	writeFileSync(
		binary,
		`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const args = process.argv.slice(2);
if (args[0] === "stage") { createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({stateDir: args[args.indexOf("--state-dir") + 1]}); } else {
if (args[0] === "init") {
  mkdirSync(".minsync", { recursive: true });
  writeFileSync(".minsync/config.toml", "[embedder]\\nid = \\"openai\\"\\n");
  process.exit(0);
}
if (args[0] === "check") {
  ${heldAt === "check" ? 'console.error("Error: another sync is in progress"); process.exit(3);' : "console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0);"}
}
if (args[0] === "sync") {
  console.error("Error: another sync is in progress");
  process.exit(3);
}
process.exit(2);
}
`,
	);
	chmodSync(binary, 0o755);
}

async function descendantPid(): Promise<number> {
	// Spawning node twice is slow under a loaded parallel suite.
	await vi.waitFor(() => expect(existsSync(descendantPidFile)).toBe(true), { timeout: 15_000 });
	await vi.waitFor(() => expect(readFileSync(descendantPidFile, "utf8")).not.toBe(""), { timeout: 15_000 });
	return Number(readFileSync(descendantPidFile, "utf8"));
}

posixOnly("spawnProcess process-tree lifecycle", () => {
	it("reaps descendants when its signal aborts", async () => {
		writeHangingSyncBinary();
		const controller = new AbortController();

		const running = spawnProcess(binary, ["sync"], workspace, { signal: controller.signal });
		const pid = await descendantPid();
		expect(isAlive(pid)).toBe(true);
		controller.abort();

		await expect(running).rejects.toThrow(/aborted/i);
		await expectDead(pid);
	});

	it("reaps descendants when its timeout fires", async () => {
		writeHangingSyncBinary();

		const result = await spawnProcess(binary, ["sync"], workspace, { timeoutMs: 1_500 });

		expect(result).toMatchObject({ ok: false, timedOut: true });
		await expectDead(await descendantPid());
	});

	it("reaps a running child when the host process is interrupted by SIGINT", async () => {
		// A detached child no longer shares the terminal's process group, so
		// Ctrl-C reaches only the host. The host must take the tree down with it.
		writeHangingSyncBinary();
		const host = spawn(
			process.execPath,
			[
				"-e",
				`import(${JSON.stringify(new URL("../../src/minsync/process.ts", import.meta.url).href)}).then(({ spawnProcess }) => {
  spawnProcess(${JSON.stringify(binary)}, ["sync"], ${JSON.stringify(workspace)});
  console.log("started");
});`,
			],
			{ stdio: ["ignore", "pipe", "inherit"] },
		);
		const exited = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) =>
			host.on("exit", (code, signal) => resolve({ code, signal })),
		);
		const pid = await descendantPid();
		expect(isAlive(pid)).toBe(true);

		host.kill("SIGINT");

		// Default SIGINT disposition is preserved: the host still dies from SIGINT.
		await expect(exited).resolves.toMatchObject({ signal: "SIGINT" });
		await expectDead(pid);
	});
});

describe("MinSyncClient.sync cancellation", () => {
	it("aborts a running sync, reaps the tree, and restores the rewritten config", async () => {
		writeHangingSyncBinary();
		const original = '[embedder]\nid = "openai"\n[vectorstore.options]\ndimension = 1024\n';
		writeFileSync(minSyncConfigPath(workspace), original);
		const controller = new AbortController();
		const client = new MinSyncClient({
			binaryPath: binary,
			workspacePath: workspace,
			embedder: { id: "openai", timeoutMs: 5_000 },
		});

		const running = client.sync(false, controller.signal);
		const pid = await descendantPid();
		// The embedder settings were written for this run; they must not outlive a cancelled one.
		expect(readFileSync(minSyncConfigPath(workspace), "utf8")).not.toBe(original);
		controller.abort();

		await expect(running).rejects.toThrow(/aborted/i);
		await expectDead(pid);
		expect(readFileSync(minSyncConfigPath(workspace), "utf8")).toBe(original);
	});

	it("rejects without spawning when the signal is already aborted", async () => {
		writeHangingSyncBinary();
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');
		const controller = new AbortController();
		controller.abort();

		await expect(
			new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync(false, controller.signal),
		).rejects.toThrow(/aborted/i);
		expect(existsSync(descendantPidFile)).toBe(false);
	});

	it("reports a held lock as minsync-lock-held, distinct from sync-failed", async () => {
		writeLockHeldBinary();
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({
			ok: false,
			synced: 0,
			reason: "lock-held",
			diagnostic: { code: "minsync-lock-held", retryable: true, message: expect.stringContaining("another sync") },
		});
	});

	it("reports a lock held at the check stage as minsync-lock-held too", async () => {
		writeLockHeldBinary("check");
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({ ok: false, reason: "lock-held", diagnostic: { code: "minsync-lock-held" } });
	});

	it("keeps a generic sync failure as sync-failed", async () => {
		writeFileSync(
			binary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
console.error("embedder down");
process.exit(1);
`,
		);
		chmodSync(binary, 0o755);
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({ ok: false, reason: "sync-failed", diagnostic: { code: "sync-failed" } });
	});

	it("streams JSONL batch progress while preserving the final sync result", async () => {
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
const args = process.argv.slice(2);
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
writeFileSync(".minsync/cursor.json", "{}");
if (args.includes("--progress")) {
  console.log(JSON.stringify({ event: "start", total_files: 2 }));
  console.log(JSON.stringify({ event: "batch", processed: 1, total: 2 }));
  console.log(JSON.stringify({ event: "heartbeat", processed: 1, total: 2 }));
  console.log(JSON.stringify({ event: "done", processed: 2, total: 2 }));
}
console.log(JSON.stringify({ files_processed: 2 }));
`,
		);
		chmodSync(binary, 0o755);
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');
		const events: { processed: number; total: number }[] = [];

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync(
			false,
			undefined,
			(progress) => events.push(progress),
		);

		expect(events).toEqual([
			{ processed: 0, total: 2 },
			{ processed: 1, total: 2 },
			{ processed: 1, total: 2 },
			{ processed: 2, total: 2 },
		]);
		expect(result).toMatchObject({ ok: true, synced: 2 });
	});

	it("treats MinSync's partial exit (6) as a completed sync and names the skipped files", async () => {
		mkdirSync(join(workspace, ".minsync"), { recursive: true });
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
const args = process.argv.slice(2);
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
writeFileSync(".minsync/cursor.json", "{}");
console.log(JSON.stringify({ files_processed: 2, partial: true, files_failed: [{ path: "files/a.md", stage: "read", error: "Permission denied" }] }));
process.exit(6);
`,
		);
		chmodSync(binary, 0o755);
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({
			ok: true,
			synced: 2,
			filesFailed: [{ path: "files/a.md", stage: "read", error: "Permission denied" }],
		});
	});

	it("keeps exit 6 without a partial report as sync-failed", async () => {
		writeFileSync(
			binary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
console.error("unexpected");
process.exit(6);
`,
		);
		chmodSync(binary, 0o755);
		writeFileSync(minSyncConfigPath(workspace), '[embedder]\nid = "openai"\n');

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({ ok: false, reason: "sync-failed" });
	});
});

posixOnly("refresh cancellation", () => {
	it("stops a running minsync sync when the refresh signal aborts", async () => {
		writeHangingSyncBinary();
		const agent = new AutoRAGAgent({
			searchPaths: [join(root, "docs")],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: binary, autoInstall: false },
			jikji: false,
			fsearch: false,
		});
		const controller = new AbortController();

		const refreshing = agent.refresh(false, { methods: ["minsync"], signal: controller.signal });
		const pid = await descendantPid();
		controller.abort();

		await expect(refreshing).rejects.toThrow(/aborted/i);
		await expectDead(pid);
	});

	it("surfaces a held lock through refresh diagnostics as minsync-lock-held", async () => {
		writeLockHeldBinary();
		const agent = new AutoRAGAgent({
			searchPaths: [join(root, "docs")],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: binary, autoInstall: false },
			jikji: false,
			fsearch: false,
		});

		const result = await agent.refresh(false, { methods: ["minsync"] });

		expect(result.minsync).toMatchObject({ ok: false });
		const codes = result.diagnostics.map((diagnostic) => diagnostic.code);
		expect(codes).toContain("minsync-lock-held");
		expect(codes).not.toContain("minsync-sync-failed");
	});

	it("reports a partial MinSync sync as ok with one minsync-file-skipped warning per skipped file", async () => {
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
const args = process.argv.slice(2);
if (args[0] === "stage") { createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({stateDir: args[args.indexOf("--state-dir") + 1]}); } else {
if (args[0] === "init") {
  mkdirSync(".minsync", { recursive: true });
  writeFileSync(".minsync/config.toml", "[embedder]\\nid = \\"openai\\"\\n");
  process.exit(0);
}
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
writeFileSync(".minsync/cursor.json", "{}");
console.log(JSON.stringify({ files_processed: 1, partial: true, files_failed: [{ path: "files/docs/a.txt.md", stage: "read", error: "Permission denied" }] }));
process.exit(6);
}
`,
		);
		chmodSync(binary, 0o755);
		const agent = new AutoRAGAgent({
			searchPaths: [join(root, "docs")],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			minSync: { binaryPath: binary, autoInstall: false, embedder: { id: "native:Qwen/Qwen3-Embedding-0.6B" } },
			fsearch: false,
		});

		const result = await agent.refresh(false, { methods: ["minsync"] });

		expect(result.minsync).toMatchObject({ ok: true, filesSkippedCount: 1 });
		expect(readRefreshProgress(root)?.minsync).toMatchObject({ synced: 1, filesSkipped: 1 });
		const skipped = result.diagnostics.filter((diagnostic) => diagnostic.code === "minsync-file-skipped");
		expect(skipped).toEqual([expect.objectContaining({ severity: "warning", source: "files/docs/a.txt.md" })]);
		expect(result.diagnostics.map((diagnostic) => diagnostic.code)).not.toContain("minsync-sync-failed");
	});
});

describe("MinSyncVectorMethod.sync cancellation", () => {
	it("rejects a queued sync whose signal aborted before it started", async () => {
		writeLockHeldBinary();
		const controller = new AbortController();
		controller.abort();

		await expect(
			new MinSyncVectorMethod({ binaryPath: binary, root, workspacePath: workspace }).sync(false, controller.signal),
		).rejects.toThrow(/aborted/i);
	});
});
