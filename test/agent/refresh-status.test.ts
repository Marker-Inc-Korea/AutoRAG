import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { writeRefreshProgress } from "../../src/agent/refresh-progress.ts";
import { AutoRAGAgent } from "../../src/index.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-status-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "a.txt"), "Alpha content\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(overrides: Record<string, unknown> = {}) {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
		...overrides,
	});
}

describe("getRefreshStatus", () => {
	it("reports idle and stale before any refresh has run", async () => {
		const agent = makeAgent();
		const status = await agent.getRefreshStatus();
		expect(status.state).toBe("idle");
		expect(status.inFlight).toBe(false);
		expect(status.stale).toBe(true);
		expect(status.counts).toBeUndefined();
	});

	it("reports the active indexing phase and known document progress", async () => {
		const agent = makeAgent();
		const releaseMinSync = deferred<void>();
		const minSyncStarted = deferred<void>();
		vi.spyOn(agent, "syncMinSync").mockImplementation(async () => {
			minSyncStarted.resolve();
			await releaseMinSync.promise;
			return { ok: true, synced: 0, workspacePath: "" };
		});

		const refresh = agent.refresh(true, { methods: ["minsync"] });
		await minSyncStarted.promise;
		expect((await agent.getRefreshStatus()).inFlight).toBe(true);

		const status = await agent.getRefreshStatus();
		expect(status.state).toBe("indexing");
		expect(status.progress).toMatchObject({
			phase: "minsync",
			sourceFiles: { total: 1 },
		});
		expect(status.progress?.minsync?.synced).toBeUndefined();

		releaseMinSync.resolve();
		await refresh;
	});

	it("reports success, counts, and freshness after a manual refresh, with no real paths", async () => {
		const agent = makeAgent();
		await agent.refresh(true);
		const status = await agent.getRefreshStatus();

		expect(status.state).toBe("success");
		expect(status.counts?.scanned).toBeGreaterThanOrEqual(1);
		expect(status.stale).toBe(false);
		expect(status.lastStartedAt).toBeDefined();
		expect(status.lastFinishedAt).toBeDefined();
		// Path opacity: no real filesystem paths (root, docs, indexPath) leak.
		const blob = JSON.stringify(status);
		expect(blob).not.toContain(root);
		expect(blob).not.toContain(docs);
		expect(blob).not.toContain("indexPath");
	});

	it("captures a path-free failure summary when a refresh step throws", async () => {
		const agent = makeAgent();
		vi.spyOn(agent, "syncParsedMirrors").mockRejectedValue(new Error("boom at /Users/secret/x"));

		await expect(agent.refresh(true)).rejects.toThrow();
		const status = await agent.getRefreshStatus();

		expect(status.state).toBe("failed");
		expect(status.inFlight).toBe(false);
		expect(status.lastError).toBeDefined();
		expect(status.lastError).not.toContain("/Users/");
	});

	it("becomes stale again when a source file changes after refresh", async () => {
		const agent = makeAgent();
		await agent.refresh(true);
		expect((await agent.getRefreshStatus()).stale).toBe(false);

		// Change a source file so its mtime/size differ from the index.
		writeFileSync(join(docs, "a.txt"), "Alpha content changed and grown\n");
		const status = await agent.getRefreshStatus();
		expect(status.stale).toBe(true);
		expect(status.diagnostics.some((d) => d.code === "stale-index")).toBe(true);
	});

	it("reports a refreshed workspace as current from a fresh instance", async () => {
		await makeAgent().refresh(true);

		// A new instance is what a separate CLI process sees: no in-memory refresh
		// history, so freshness has to come from what the last refresh left on disk.
		const status = await makeAgent().getRefreshStatus();

		expect(status.stale).toBe(false);
	});

	it("reports a dead persisted refresh owner as interrupted instead of indexing", async () => {
		writeRefreshProgress(root, {
			version: 1,
			runId: "dead-refresh",
			pid: 2_147_483_647,
			state: "running",
			phase: "minsync",
			startedAt: new Date().toISOString(),
			updatedAt: new Date().toISOString(),
			sourceFiles: { total: 338 },
		});

		const status = await makeAgent().getRefreshStatus();

		expect(status.state).toBe("failed");
		expect(status.inFlight).toBe(false);
		expect(status.progress).toMatchObject({
			phase: "minsync",
			sourceFiles: { total: 338 },
			ownerAlive: false,
		});
		expect(status.diagnostics).toEqual(
			expect.arrayContaining([
				expect.objectContaining({
					code: "refresh-interrupted",
					severity: "error",
				}),
			]),
		);
	});

	it("stays current after a refresh that deliberately skipped sources", async () => {
		// A product artifact and an unparseable file: refresh reports both, and must
		// not leave the corpus reading as stale afterwards.
		writeFileSync(join(docs, ".jikji_agent_map.md"), "# Jikji Agent Map\n");
		writeFileSync(join(docs, "broken.hwp"), Buffer.from([1, 2, 3, 4]));

		const agent = makeAgent();
		const result = await agent.refresh(true);
		const status = await agent.getRefreshStatus();

		expect(result.diagnostics.some((d) => d.code === "parser-failed")).toBe(true);
		expect(status.stale).toBe(false);
		expect(status.diagnostics.some((d) => d.code === "stale-index")).toBe(false);
	});

	it("reports an unavailable MinSync component in status without leaking paths", async () => {
		const agent = makeAgent({
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
		});
		// Refreshing the other indexes still works; only the MinSync step needs the binary.
		await agent.refresh(true, { methods: ["parsed"] });
		const status = await agent.getRefreshStatus();

		expect(status.components.minsync).toBe("unavailable");
		expect(JSON.stringify(status)).not.toContain(root);
	});

	it("surfaces unknown-datasource-skill startup diagnostics on status", async () => {
		const agent = makeAgent({
			startupDiagnostics: [
				{
					code: "unknown-datasource-skill",
					severity: "warning",
					message: "Unknown datasource skill(s) in config were skipped: dropbox",
					source: "datasources",
				},
			],
		});
		const status = await agent.getRefreshStatus();
		expect(status.diagnostics).toEqual(
			expect.arrayContaining([
				{
					code: "unknown-datasource-skill",
					severity: "warning",
					message: "Unknown datasource skill(s) in config were skipped: dropbox",
					source: "datasources",
				},
			]),
		);
	});

	it("includes startup diagnostics in refresh results", async () => {
		const agent = makeAgent({
			startupDiagnostics: [
				{
					code: "unknown-datasource-skill",
					severity: "warning",
					message: "Unknown datasource skill(s) in config were skipped: dropbox",
					source: "datasources",
				},
			],
		});

		const result = await agent.refresh(true);

		expect(result.diagnostics).toEqual(
			expect.arrayContaining([
				expect.objectContaining({
					code: "unknown-datasource-skill",
					severity: "warning",
					source: "datasources",
				}),
			]),
		);
	});

	it("emits a watch-limited diagnostic when startWatchRefresh exceeds the watcher cap", async () => {
		const agent = makeAgent();
		const handle = agent.startWatchRefresh({
			maxWatchers: 0,
			watcherFactory: () => ({ close: () => undefined }),
		});
		const status = await agent.getRefreshStatus();
		handle.stop();
		expect(status.diagnostics.some((d) => d.code === "watch-limited")).toBe(true);
	});

	it("surfaces minsync failure diagnostics in refresh results and getRefreshStatus", async () => {
		// `.mjs` keeps the fixture spawnable on Windows, where shebang scripts are not.
		const fakeBinary = join(root, "fake-failing-minsync.mjs");
		writeFileSync(
			fakeBinary,
			`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

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

		const agent = makeAgent({
			minSync: {
				binaryPath: fakeBinary,
				workspacePath: join(root, ".autorag", "minsync"),
				autoInstall: false,
			},
		});

		const result = await agent.refresh(true);
		expect(result.minsync).toBeDefined();
		expect(result.minsync?.ok).toBe(false);
		expect(result.minsync?.diagnostics?.some((d) => d.code === "embedder-unavailable")).toBe(true);
		expect(result.diagnostics.some((d) => d.code === "embedder-unavailable" && d.source === "minsync")).toBe(true);
		// Path opacity: MinSync text reaches the public result already sanitized.
		expect(JSON.stringify(result.minsync)).not.toContain(root);

		const status = await agent.getRefreshStatus();
		expect(status.components.minsync).toBe("degraded");
		expect(status.diagnostics.some((d) => d.code === "embedder-unavailable" && d.source === "minsync")).toBe(true);
	});
});

function deferred<T>(): { promise: Promise<T>; resolve: (value: T) => void } {
	let resolve!: (value: T) => void;
	const promise = new Promise<T>((resolver) => {
		resolve = resolver;
	});
	return { promise, resolve };
}
