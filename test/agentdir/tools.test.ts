import { mkdirSync, mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { assertNoSourcePath, isSourcePathFree } from "../../src/agentdir/assert-no-source-path.ts";
import { agentdirGrep } from "../../src/agentdir/grep-core.ts";
import { ACTIVE_TOOLS, AGENTDIR_TOOL_NAMES, createAgentdirTools, SEARCH_TOOLS } from "../../src/agentdir/tools.ts";
import { bootstrapMappings, clearWorkspaceCache, getWorkspace } from "../../src/agentdir/workspace.ts";

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-tools-"));
	clearWorkspaceCache();
	source = join(root, "docs");
	mkdirSync(join(source, "nested"), { recursive: true });
	// "many" matches alpha three times; "few" once; nested deeper for tie-break checks
	writeFileSync(join(source, "many.txt"), "alpha alpha\nalpha here\nbeta\n");
	writeFileSync(join(source, "few.md"), "alpha once\nbeta beta\n");
	writeFileSync(join(source, "nested", "deep.txt"), "gamma\n");
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

async function makeTreeTools(): Promise<{ tools: Map<string, AgentTool>; sourceRoot: string }> {
	const ws = getWorkspace(root);
	await bootstrapMappings(ws, [source]);
	const tools = new Map(createAgentdirTools(ws).map((t) => [t.name, t]));
	return { tools, sourceRoot: source };
}

async function call(tool: AgentTool, params: Record<string, unknown>) {
	return tool.execute("test-call", params as never, undefined, undefined as never);
}

function textOf(res: { content: Array<{ type: string; text?: string }> }): string {
	return res.content
		.filter((c) => c.type === "text" && typeof c.text === "string")
		.map((c) => c.text as string)
		.join("\n");
}

describe("exports", () => {
	it("declares the closed active tool surface", () => {
		expect(AGENTDIR_TOOL_NAMES).toEqual(["grep", "find", "read", "ls", "stat", "mv", "cp", "mkdir", "rmdir"]);
		expect(SEARCH_TOOLS).toEqual(["grep", "find"]);
		expect(ACTIVE_TOOLS).toContain("check_memory");
		expect(ACTIVE_TOOLS).toContain("organize");
		// builtin write/edit/bash must NOT be part of the closed surface
		for (const banned of ["bash", "edit", "write"]) {
			expect(ACTIVE_TOOLS).not.toContain(banned);
		}
	});
});

describe("agentdirGrep scoring", () => {
	it("ranks a file with more matches above one with fewer (non-degenerate)", async () => {
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);
		const hits = await agentdirGrep(ws, "alpha");
		expect(hits.length).toBe(2);
		expect(hits[0].virtualPath).toBe("/docs/many.txt");
		expect(hits[0].matchCount).toBe(3);
		expect(hits[1].virtualPath).toBe("/docs/few.md");
		expect(hits[1].matchCount).toBe(1);
		expect(hits[0].score).toBeGreaterThan(hits[1].score);
	});

	it("supports case-insensitive matching and literal fallback for invalid regex", async () => {
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);
		const ci = await agentdirGrep(ws, "ALPHA", { ignoreCase: true });
		expect(ci.length).toBe(2);
		// invalid regex "(" falls back to literal and simply finds nothing here
		const literal = await agentdirGrep(ws, "(");
		expect(Array.isArray(literal)).toBe(true);
	});
});

describe("agentdir search/nav tools", () => {
	it("grep returns virtual paths and populated details with no source leak (AC-4)", async () => {
		const { tools, sourceRoot } = await makeTreeTools();
		const res = await call(tools.get("grep")!, { pattern: "alpha" });
		const details = res.details as { method: string; sources: string[]; resultCount: number };
		expect(details.method).toBe("grep");
		expect(details.resultCount).toBe(2);
		expect(details.sources).toContain("/docs/many.txt");
		expect(() => assertNoSourcePath(res, [sourceRoot])).not.toThrow();
		expect(textOf(res)).not.toContain(sourceRoot);
	});

	it("find lists files by glob (AC-6)", async () => {
		const { tools } = await makeTreeTools();
		const res = await call(tools.get("find")!, { pattern: "*.txt" });
		expect(textOf(res)).toContain("/docs/many.txt");
		expect(textOf(res)).toContain("/docs/nested/deep.txt");
		expect(textOf(res)).not.toContain("/docs/few.md");
	});

	it("read returns content by virtual path and degrades gracefully", async () => {
		const { tools } = await makeTreeTools();
		const ok = await call(tools.get("read")!, { path: "/docs/few.md" });
		expect(textOf(ok)).toContain("alpha once");
		const missing = await call(tools.get("read")!, { path: "/docs/nope.txt" });
		expect((missing.details as { resultCount: number }).resultCount).toBe(0);
	});

	it("ls lists virtual entries", async () => {
		const { tools } = await makeTreeTools();
		const res = await call(tools.get("ls")!, { path: "/docs" });
		expect(textOf(res)).toContain("/docs/many.txt");
	});

	it("stat hides the source path (AC-4)", async () => {
		const { tools, sourceRoot } = await makeTreeTools();
		const res = await call(tools.get("stat")!, { path: "/docs/many.txt" });
		expect(textOf(res)).toContain("/docs/many.txt");
		expect(textOf(res)).not.toContain("sourcePath");
		expect(textOf(res)).not.toContain(sourceRoot);
		expect(isSourcePathFree(res, [sourceRoot])).toBe(true);
	});
});

describe("agentdir virtual ops leave source files unchanged (AC-2)", () => {
	it("mv/cp/mkdir change only the virtual namespace", async () => {
		const { tools, sourceRoot } = await makeTreeTools();
		const before = readFileSync(join(sourceRoot, "many.txt"), "utf8");
		const beforeStat = statSync(join(sourceRoot, "many.txt"));

		await call(tools.get("mkdir")!, { path: "/reports" });
		await call(tools.get("cp")!, { from: "/docs/many.txt", to: "/reports/copy.txt" });
		await call(tools.get("mv")!, { from: "/docs/few.md", to: "/reports/moved.md" });

		const ws = getWorkspace(root);
		expect(await ws.exists("/reports/copy.txt")).toBe(true);
		expect(await ws.exists("/reports/moved.md")).toBe(true);

		// originals on disk are untouched
		expect(readFileSync(join(sourceRoot, "many.txt"), "utf8")).toBe(before);
		expect(statSync(join(sourceRoot, "many.txt")).size).toBe(beforeStat.size);
		// few.md source still exists on disk even though it was moved in the virtual tree
		expect(readFileSync(join(sourceRoot, "few.md"), "utf8")).toContain("alpha once");
	});

	it("rmdir removes a virtual directory without deleting source files", async () => {
		const { tools, sourceRoot } = await makeTreeTools();
		await call(tools.get("rmdir")!, { path: "/docs/nested", recursive: true });
		const ws = getWorkspace(root);
		expect(await ws.exists("/docs/nested")).toBe(false);
		// source file on disk still present
		expect(readFileSync(join(sourceRoot, "nested", "deep.txt"), "utf8")).toContain("gamma");
	});
});

describe("assertNoSourcePath", () => {
	it("throws when a value embeds a known source root", () => {
		expect(() => assertNoSourcePath({ leaked: "/abs/source/docs/a.txt" }, ["/abs/source"])).toThrow(
			/path opacity violation/,
		);
	});
	it("passes clean virtual-only values", () => {
		expect(() => assertNoSourcePath({ path: "/docs/a.txt" }, ["/abs/source"])).not.toThrow();
	});
});
