import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	bootstrapMappings,
	clearWorkspaceCache,
	getWorkspace,
	planMounts,
	refreshWorkspace,
	workspaceRoot,
} from "../../src/agentdir/workspace.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ws-"));
	clearWorkspaceCache();
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

function makeSource(name: string, files: Record<string, string>): string {
	const dir = join(root, name);
	mkdirSync(dir, { recursive: true });
	for (const [rel, content] of Object.entries(files)) {
		const full = join(dir, rel);
		mkdirSync(join(full, ".."), { recursive: true });
		writeFileSync(full, content);
	}
	return dir;
}

describe("planMounts", () => {
	it("maps each source to /<basename>", () => {
		const plan = planMounts(["/a/docs", "/b/reports"]);
		expect(plan).toEqual([
			{ source: "/a/docs", mount: "/docs" },
			{ source: "/b/reports", mount: "/reports" },
		]);
	});

	it("resolves basename collisions with a deterministic numeric suffix", () => {
		const plan = planMounts(["/x/docs", "/y/docs", "/z/docs"]);
		const mounts = plan.map((p) => p.mount).sort();
		expect(mounts).toEqual(["/docs", "/docs-2", "/docs-3"]);
		// stable across input ordering
		const reordered = planMounts(["/z/docs", "/x/docs", "/y/docs"]);
		expect(reordered).toEqual(plan);
	});

	it("strips trailing separators and falls back to root", () => {
		expect(planMounts(["/a/docs/"])).toEqual([{ source: "/a/docs/", mount: "/docs" }]);
	});
});

describe("getWorkspace", () => {
	it("inits a workspace and exposes a reachable status (AC-1)", async () => {
		const ws = getWorkspace(root);
		const status = await ws.status();
		expect(status.materializedRoot).toContain(workspaceRoot(root));
		expect(status.totalEntries).toBe(0);
	});

	it("returns the same cached handle and reopens an existing workspace", () => {
		const a = getWorkspace(root);
		expect(getWorkspace(root)).toBe(a);
		clearWorkspaceCache();
		// second call must open-or-init without throwing on an existing workspace
		expect(() => getWorkspace(root)).not.toThrow();
	});
});

describe("bootstrapMappings", () => {
	it("maps sources into the virtual tree and is idempotent (AC-6)", async () => {
		const docs = makeSource("docs", { "a.txt": "hello\n", "nested/b.md": "world\n" });
		const ws = getWorkspace(root);

		const first = await bootstrapMappings(ws, [docs]);
		expect(first.mounts).toEqual([{ source: docs, mount: "/docs" }]);
		expect(first.entriesAdded).toBeGreaterThan(0);

		const glob = await ws.rglob("/docs/**/*");
		expect(glob).toContain("/docs/a.txt");
		expect(glob).toContain("/docs/nested/b.md");

		// re-running maps nothing new (mount already exists)
		const second = await bootstrapMappings(ws, [docs]);
		expect(second.entriesAdded).toBe(0);
	});
});

describe("refreshWorkspace", () => {
	it("returns a refresh summary in both modes", async () => {
		const docs = makeSource("docs", { "a.txt": "hello\n" });
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [docs]);

		const plain = await refreshWorkspace(ws, { verifyHashes: false });
		expect(plain).toMatchObject({ added: 0, removed: 0, errors: 0 });

		const verified = await refreshWorkspace(ws, { verifyHashes: true });
		expect(verified).toMatchObject({ added: 0, removed: 0, errors: 0 });
	});
});

describe("AutoRAGAgent workspace lifecycle", () => {
	it("refresh() opens the workspace, maps searchPaths, and returns a summary (AC-1)", async () => {
		const docs = makeSource("docs", { "a.txt": "hello\n" });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		const summary = await agent.refresh();
		expect(summary).toMatchObject({ errors: 0 });

		const ws = getWorkspace(root);
		expect(await ws.exists("/docs/a.txt")).toBe(true);
	});
});
