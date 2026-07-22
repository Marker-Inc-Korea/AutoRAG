import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
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

	it("reports component status for bm25 and minsync without leaking paths", async () => {
		const agent = makeAgent({
			bm25: { forceEngine: "typescript-fallback" },
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
		});
		await agent.refresh(true);
		const status = await agent.getRefreshStatus();

		expect(status.components.bm25).toBeDefined();
		expect(status.components.minsync).toBe("unavailable");
		expect(JSON.stringify(status)).not.toContain(root);
	});
	it("emits a watch-limited diagnostic when startWatchRefresh exceeds the watcher cap", async () => {
		const agent = makeAgent();
		const handle = agent.startWatchRefresh({
			maxWatchers: 0,
			watcherFactory: () => ({ close: () => {} }),
		});
		const status = await agent.getRefreshStatus();
		handle.stop();
		expect(status.diagnostics.some((d) => d.code === "watch-limited")).toBe(true);
	});
});
