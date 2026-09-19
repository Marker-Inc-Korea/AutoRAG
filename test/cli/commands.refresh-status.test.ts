import { chmodSync, existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { runRefresh } from "../../src/cli/commands/refresh.ts";
import { runStatus } from "../../src/cli/commands/status.ts";
import type { CommandContext } from "../../src/cli/commands/types.ts";

let root: string;
let docs: string;
let previousHome: string | undefined;
let previousPath: string | undefined;

function pathWithoutMinsync(pathEnv = process.env.PATH ?? ""): string {
	const execName = process.platform === "win32" ? "minsync.exe" : "minsync";
	return pathEnv
		.split(delimiter)
		.filter((dir) => dir.length > 0 && !existsSync(join(dir, execName)))
		.join(delimiter);
}

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cli-refresh-"));
	previousHome = process.env.HOME;
	previousPath = process.env.PATH;
	process.env.HOME = join(root, "home");
	process.env.PATH = pathWithoutMinsync(previousPath);
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "alpha.md"), "# Alpha\n\nAlpha document body content.\n");
});

afterEach(() => {
	if (previousHome === undefined) delete process.env.HOME;
	else process.env.HOME = previousHome;
	if (previousPath === undefined) delete process.env.PATH;
	else process.env.PATH = previousPath;
	rmSync(root, { recursive: true, force: true });
});

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

function writeConfig(minSync: unknown = { autoInstall: false }): void {
	const config: Record<string, unknown> = {
		searchPaths: ["docs"],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		jikji: false,
	};
	if (minSync !== undefined) config.minSync = minSync;
	const configDir = join(process.env.HOME as string, ".autorag");
	mkdirSync(configDir, { recursive: true });
	writeFileSync(join(configDir, "config.json"), `${JSON.stringify(config, null, 2)}\n`);
}

describe("runStatus exit codes", () => {
	it("returns exit 2 when --config points to a missing file", async () => {
		const err: string[] = [];
		const code = await runStatus(
			makeCtx({
				flags: { config: join(root, "nonexistent-config.json") },
				stderr: (line) => err.push(line),
			}),
		);
		expect(code).toBe(2);
		expect(err.join("\n")).toContain("Config file not found");
	});
});

describe("runRefresh + runStatus (cli)", () => {
	it("refresh then status emits JSON with counts and no leaked paths", async () => {
		writeConfig();

		const refreshOut: string[] = [];
		const refreshCode = await runRefresh(makeCtx({ stdout: (line) => refreshOut.push(line) }));
		expect(refreshCode).toBe(0);
		expect(refreshOut).toHaveLength(1);

		const refreshBlob = refreshOut[0];
		// Path opacity: no absolute index path, no temp root, no bm25 subdir literal.
		expect(refreshBlob).not.toContain("indexPath");
		expect(refreshBlob).not.toContain(root);
		expect(refreshBlob).not.toContain(join(".autorag", "bm25"));

		const statusOut: string[] = [];
		const statusCode = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(statusCode).toBe(0);
		expect(statusOut).toHaveLength(1);

		const statusBlob = statusOut[0];
		expect(statusBlob).not.toContain("indexPath");
		expect(statusBlob).not.toContain(root);
		expect(statusBlob).not.toContain(join(".autorag", "bm25"));

		const status = JSON.parse(statusBlob);
		// `status` runs in a fresh agent instance (a separate CLI process in real
		// use), so in-memory `state`/`counts` are not carried across invocations.
		// Freshness is: it comes from the readiness marker the refresh wrote plus the
		// stat-only scan, so a new invocation after a refresh reports the corpus as
		// current and finds no source newer than the recorded mirror index.
		expect(typeof status.state).toBe("string");
		expect(Array.isArray(status.diagnostics)).toBe(true);
		expect(status.stale).toBe(false);
		const staleDiagnostics = (status.diagnostics as { code: string }[]).filter((d) => d.code === "stale-index");
		expect(staleDiagnostics).toHaveLength(0);
		expect(status.components).toBeDefined();
	});

	it("surfaces minsync-unavailable without throwing when the minsync binary is absent", async () => {
		writeConfig({
			workspacePath: join(root, ".autorag", "minsync"),
			autoInstall: false,
		});

		const refreshOut: string[] = [];
		const refreshCode = await runRefresh(makeCtx({ stdout: (line) => refreshOut.push(line) }));
		expect(refreshCode).toBe(0);
		// Refresh must not leak paths even when minsync is unavailable.
		const refreshBlob = refreshOut[0];
		expect(refreshBlob).not.toContain(root);
		expect(refreshBlob).not.toContain("indexPath");

		const statusOut: string[] = [];
		const statusCode = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(statusCode).toBe(0);

		const status = JSON.parse(statusOut[0]);
		// MinSync absence surfaces as a configured component state, not a throw.
		expect(status.components).toBeDefined();
		expect(status.components.minsync).toBe("configured");
		// Path opacity holds on the status path too.
		expect(statusOut[0]).not.toContain(root);
		expect(statusOut[0]).not.toContain("indexPath");
	});

	it("skips unknown datasource entries and reports a non-fatal diagnostic", async () => {
		const configDir = join(process.env.HOME as string, ".autorag");
		mkdirSync(configDir, { recursive: true });
		writeFileSync(
			join(configDir, "config.json"),
			`${JSON.stringify(
				{
					searchPaths: ["docs"],
					workspacePath: root,
					memoryPath: join(root, "memory.json"),
					bm25: { forceEngine: "typescript-fallback" },
					minSync: false,
					datasources: {
						"discord-nomadamas": {},
						"slack-local": {},
					},
				},
				null,
				2,
			)}\n`,
		);

		const stderr: string[] = [];
		const statusOut: string[] = [];
		const code = await runStatus(
			makeCtx({
				stdout: (line) => statusOut.push(line),
				stderr: (line) => stderr.push(line),
			}),
		);
		expect(code).toBe(0);
		expect(stderr.join("\n")).not.toContain("Unknown datasource skill");
		expect(statusOut).toHaveLength(1);
		const status = JSON.parse(statusOut[0]) as { diagnostics: Array<{ code: string; message: string }> };
		expect(status.diagnostics.some((diagnostic) => diagnostic.code === "unknown-datasource-skill")).toBe(true);
		expect(statusOut[0]).toContain("discord-nomadamas");
		expect(statusOut[0]).toContain("slack-local");
		expect(statusOut[0]).not.toContain(root);
	});

	it("reports idle and stale before any refresh has run", async () => {
		writeConfig();

		const statusOut: string[] = [];
		const code = await runStatus(makeCtx({ stdout: (line) => statusOut.push(line) }));
		expect(code).toBe(0);

		const status = JSON.parse(statusOut[0]);
		expect(status.state).toBe("idle");
		expect(status.stale).toBe(true);
		expect(status.counts).toBeUndefined();
	});
});

describe("runRefresh --method", () => {
	it("refreshes only MinSync when --method minsync is given", async () => {
		writeConfig();

		const out: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "minsync" }, stdout: (line) => out.push(line) }));
		expect(code).toBe(0);
		expect(out).toHaveLength(1);
		const blob = out[0];
		expect(blob).not.toContain("indexPath");
		expect(blob).not.toContain(root);
		const parsed = JSON.parse(blob);
		expect(parsed.counts).toBeDefined();
		expect(parsed.minsync).toBeDefined();
		expect(typeof parsed.minsync.ok).toBe("boolean");
	});

	it("surfaces minsync failure in refresh JSON envelope when minsync embedder fails", async () => {
		// The CLI config ignores a persisted `minSync.binaryPath`: MinSync is resolved
		// from PATH and the workspace cache. The fixture therefore owns PATH, which
		// also keeps the test independent of a real `minsync` on the developer's box.
		// PATH injection of a shebang fixture is POSIX-only; the same failure path is
		// covered cross-platform by test/agent/refresh-status.test.ts.
		if (process.platform === "win32") return;

		const fakeBinDir = join(root, "fake-bin");
		mkdirSync(fakeBinDir, { recursive: true });
		const fakeBinary = join(fakeBinDir, "minsync");
		writeFileSync(
			fakeBinary,
			`#!/usr/bin/env node
const { mkdirSync, writeFileSync } = require("node:fs");
const { dirname, join } = require("node:path");

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, '[embedder]\\nid = "fixture"\\n');
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}
if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: false }));
  process.exit(0);
}
process.exit(2);
`,
		);
		chmodSync(fakeBinary, 0o755);
		process.env.PATH = `${fakeBinDir}${delimiter}${pathWithoutMinsync(previousPath)}`;

		writeConfig({
			workspacePath: join(root, ".autorag", "minsync"),
			autoInstall: false,
		});

		const refreshOut: string[] = [];
		const refreshCode = await runRefresh(makeCtx({ stdout: (line) => refreshOut.push(line) }));
		expect(refreshCode).toBe(0);
		expect(refreshOut).toHaveLength(1);

		const parsed = JSON.parse(refreshOut[0]);
		expect(parsed.ok).toBe(false);
		expect(parsed.minsync).toBeDefined();
		expect(parsed.minsync.ok).toBe(false);
		expect(parsed.minsync.reason).toContain("check-failed");
		expect(
			parsed.diagnostics.some(
				(d: { code: string; source?: string }) => d.code === "embedder-unavailable" && d.source === "minsync",
			),
		).toBe(true);
		expect(refreshOut[0]).not.toContain(root);
	});

	it("refreshes with all methods when --method all is given", async () => {
		writeConfig();

		const out: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "all" }, stdout: (line) => out.push(line) }));
		expect(code).toBe(0);
		const parsed = JSON.parse(out[0]);
		expect(parsed.counts).toBeDefined();
	});

	it("rejects an unknown --method value", async () => {
		writeConfig();

		const err: string[] = [];
		const code = await runRefresh(makeCtx({ flags: { method: "bogus" }, stderr: (line) => err.push(line) }));
		expect(code).toBe(1);
		expect(err.join("\n")).toContain("Unknown --method value");
	});
});
