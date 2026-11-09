import { spawn } from "node:child_process";
import {
	chmodSync,
	copyFileSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runIndex } from "../../src/cli/commands/index.ts";
import { runLiteRefresh } from "../../src/cli/commands/lite-refresh.ts";
import { runRefresh } from "../../src/cli/commands/refresh.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";
import { runWatch } from "../../src/cli/commands/watch.ts";
import { SIGNAL_EXIT_CODES } from "../../src/cli/signal-abort.ts";
import { fakeMinSyncExecutableName, fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

const CLI_ENTRY = fileURLToPath(new URL("../../src/cli/index.ts", import.meta.url));

let root: string;
let docs: string;
const previousEnv = new Map<string, string | undefined>();

/** The config resolver reads these before it falls back to `$HOME/.autorag`, so a test pins all three. */
function isolateEnv(env: Record<string, string>): void {
	for (const [key, value] of Object.entries(env)) {
		previousEnv.set(key, process.env[key]);
		process.env[key] = value;
	}
}

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-refresh-signal-"));
	isolateEnv({
		HOME: join(root, "home"),
		AUTORAG_HOME: join(root, "home", ".autorag"),
		AUTORAG_CONFIG: join(root, "home", ".autorag", "config.json"),
	});
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	for (let index = 0; index < 5; index += 1) writeFileSync(join(docs, `f${index}.txt`), `content ${index}\n`);
});

afterEach(() => {
	for (const [key, value] of previousEnv) {
		if (value === undefined) delete process.env[key];
		else process.env[key] = value;
	}
	previousEnv.clear();
	rmSync(root, { recursive: true, force: true });
});

/** A MinSync whose `sync` runs until it is killed, and records its own pid so a test can find the process. */
function writeHangingMinSync(binaryPath: string, pidPath: string): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { createRequire } from "node:module";
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
if (args[0] === "stage") {
  const eqArg = args.find((a) => a.startsWith("--state-dir="));
  const flagAt = args.indexOf("--state-dir");
  const stateDir = eqArg ? eqArg.slice("--state-dir=".length) : flagAt >= 0 ? args[flagAt + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  mkdirSync(join(process.cwd(), ".minsync"), { recursive: true });
  writeFileSync(join(process.cwd(), ".minsync", "config.toml"), '[embedder]\\nid = "fixture"\\n');
  console.log(JSON.stringify({ initialized: true }));
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
} else if (args[0] === "sync") {
  writeFileSync(${JSON.stringify(pidPath)}, String(process.pid));
  mkdirSync(dirname(join(process.cwd(), ".minsync", "cursor.json")), { recursive: true });
  // A real sync can run for hours; hold until the host stops it.
  setInterval(() => {}, 1000);
} else {
  console.error("unexpected fake minsync command: " + args.join(" "));
  process.exit(2);
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

/** The resolver takes the first `minsync` on PATH, so the fake goes in its own directory at the front. */
function installMinSync(source: string): void {
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	copyFileSync(source, join(binDir, fakeMinSyncExecutableName()));
	chmodSync(join(binDir, fakeMinSyncExecutableName()), 0o755);
	isolateEnv({ PATH: `${binDir}${delimiter}${process.env.PATH ?? ""}` });
}

function writeConfig(binary: string): string {
	installMinSync(binary);
	const configDir = join(root, "home", ".autorag");
	mkdirSync(configDir, { recursive: true });
	const configPath = join(configDir, "config.json");
	writeFileSync(
		configPath,
		`${JSON.stringify({
			searchPaths: [docs],
			workspacePath: join(root, "ws"),
			memoryPath: join(root, "memory.json"),
			minSync: { workspacePath: join(root, "ws", ".autorag", "minsync"), autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
		})}\n`,
	);
	return configPath;
}

function isAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

async function waitFor(condition: () => boolean, what: string, timeoutMs = 60_000): Promise<void> {
	const started = Date.now();
	while (!condition()) {
		if (Date.now() - started > timeoutMs) throw new Error(`timed out waiting for ${what}`);
		await new Promise((resolve) => setImmediate(resolve));
		await new Promise((resolve) => setTimeout(resolve, 25));
	}
}

function makeCtx(overrides: Partial<CommandContext> = {}): CommandContext {
	return {
		positionals: [],
		flags: {},
		json: true,
		debug: false,
		cwd: root,
		stdout: () => {},
		stderr: () => {},
		...overrides,
	};
}

describe("runRefresh signal handling", () => {
	it("aborts the refresh when the process is told to stop, and exits with the signal's conventional code", async () => {
		const binary = join(root, "hang-minsync.mjs");
		const pidPath = join(root, "sync.pid");
		writeHangingMinSync(binary, pidPath);
		writeConfig(binary);
		const err: string[] = [];

		const running = runRefresh(makeCtx({ flags: { method: "parsed,minsync" }, stderr: (line) => err.push(line) }));
		await waitFor(() => existsSync(pidPath), "the fake sync to start");
		const syncPid = Number(readFileSync(pidPath, "utf8"));
		expect(isAlive(syncPid)).toBe(true);

		process.emit("SIGTERM");
		const code = await running;

		expect(code).toBe(SIGNAL_EXIT_CODES.SIGTERM);
		// The refresh stopped because it was aborted, not because the host killed its child and it failed on that.
		expect(err.join("\n")).toContain("Interrupted by SIGTERM");
		// The sync was stopped by the abort path, not left for the host-exit cleanup.
		await waitFor(() => !isAlive(syncPid), "the fake sync to exit", 15_000);
	}, 90_000);

	it("treats SIGINT the same way", async () => {
		const binary = join(root, "hang-minsync.mjs");
		const pidPath = join(root, "sync.pid");
		writeHangingMinSync(binary, pidPath);
		writeConfig(binary);

		const err: string[] = [];
		const running = runRefresh(makeCtx({ flags: { method: "parsed,minsync" }, stderr: (line) => err.push(line) }));
		await waitFor(() => existsSync(pidPath), "the fake sync to start");

		process.emit("SIGINT");

		expect(await running).toBe(SIGNAL_EXIT_CODES.SIGINT);
		expect(err.join("\n")).toContain("Interrupted by SIGINT");
	}, 90_000);

	it("leaves no signal listener behind once the refresh has finished", async () => {
		const binary = join(root, "quick-minsync.mjs");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "check") console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
else if (args[0] === "sync") console.log(JSON.stringify({ files_processed: 0 }));
else console.log(JSON.stringify({ initialized: true }));
`,
		);
		chmodSync(binary, 0o755);
		writeConfig(binary);
		const before = { term: process.listenerCount("SIGTERM"), int: process.listenerCount("SIGINT") };

		await runRefresh(makeCtx({ flags: { method: "parsed" } }));

		expect(process.listenerCount("SIGTERM")).toBe(before.term);
		expect(process.listenerCount("SIGINT")).toBe(before.int);
	}, 60_000);
});

describe("every command that runs a refresh stops on a signal", () => {
	const commands: readonly { readonly name: string; readonly run: (ctx: CommandContext) => Promise<number> }[] = [
		{ name: "lite refresh", run: (ctx) => runLiteRefresh({ ...ctx, flags: { method: "parsed,minsync" } }) },
		{ name: "watch --once", run: (ctx) => runWatch({ ...ctx, flags: { once: true } }) },
		{
			name: "index rebuild",
			run: (ctx) => runIndex({ ...ctx, positionals: ["rebuild"], flags: { yes: true, method: "minsync" } }),
		},
	];

	for (const command of commands) {
		it(`${command.name} exits 143 on SIGTERM and stops the sync`, async () => {
			const binary = join(root, "hang-minsync.mjs");
			const pidPath = join(root, "sync.pid");
			writeHangingMinSync(binary, pidPath);
			writeConfig(binary);

			const err: string[] = [];
			const running = command.run(makeCtx({ stderr: (line) => err.push(line) }));
			await waitFor(() => existsSync(pidPath), "the fake sync to start");
			const syncPid = Number(readFileSync(pidPath, "utf8"));

			process.emit("SIGTERM");

			expect(await running).toBe(SIGNAL_EXIT_CODES.SIGTERM);
			expect(err.join("\n")).toContain("Interrupted by SIGTERM");
			await waitFor(() => !isAlive(syncPid), "the fake sync to exit", 15_000);
		}, 90_000);
	}
});

describe("autorag refresh as a real process", () => {
	it("stops on SIGTERM without orphaning the sync it started", async () => {
		const binary = join(root, "hang-minsync.mjs");
		const pidPath = join(root, "sync.pid");
		writeHangingMinSync(binary, pidPath);
		const configPath = writeConfig(binary);
		const child = spawn(process.execPath, [CLI_ENTRY, "refresh", "--method", "parsed,minsync", "--json"], {
			cwd: root,
			env: { ...process.env, AUTORAG_CONFIG: configPath, AUTORAG_HOME: join(root, "home", ".autorag") },
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stderr = "";
		child.stderr.on("data", (chunk: Buffer) => {
			stderr += chunk.toString();
		});
		const exited = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>((resolve) =>
			child.once("close", (code, signal) => resolve({ code, signal })),
		);
		await waitFor(() => existsSync(pidPath), "the fake sync to start");
		const syncPid = Number(readFileSync(pidPath, "utf8"));

		child.kill("SIGTERM");
		const result = await Promise.race([
			exited,
			new Promise<never>((_, reject) => setTimeout(() => reject(new Error("refresh ignored SIGTERM")), 30_000)),
		]).catch((error) => {
			child.kill("SIGKILL");
			throw error;
		});

		expect(result.code).toBe(SIGNAL_EXIT_CODES.SIGTERM);
		expect(stderr).toContain("Interrupted by SIGTERM");
		await waitFor(() => !isAlive(syncPid), "the fake sync to exit", 15_000);
	}, 120_000);
});
