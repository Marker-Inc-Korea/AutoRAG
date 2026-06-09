import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { SEARCH_TOOLS } from "../../src/agentdir/tools.ts";
import { bootstrapMappings, clearWorkspaceCache, getWorkspace } from "../../src/agentdir/workspace.ts";
import { AgentdirPosixMethod } from "../../src/retrieval/methods/posix.ts";

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-posix-"));
	clearWorkspaceCache();
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
	writeFileSync(join(source, "many.txt"), "alpha alpha\nalpha\n");
	writeFileSync(join(source, "few.md"), "alpha\nbeta\n");
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

describe("AgentdirPosixMethod", () => {
	it("describes itself as an active posix method", () => {
		const method = new AgentdirPosixMethod(() => getWorkspace(root));
		const d = method.describe();
		expect(d.name).toBe("posix");
		expect(d.type).toBe("posix");
		expect(d.status).toBe("active");
	});

	it("maps grep hits to RetrievalResult with virtual source and no source leak", async () => {
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);
		const method = new AgentdirPosixMethod(ws);
		const results = await method.retrieve("alpha", {});
		expect(results.length).toBe(2);
		expect(results[0].source).toBe("/docs/many.txt");
		expect(results[0].score).toBeGreaterThan(results[1].score);
		expect(JSON.stringify(results)).not.toContain(source);
		expect(results[0].metadata.method).toBe("posix");
	});
});

describe("AutoRAGAgent.retrieve (registry + parallel retriever + merger)", () => {
	it("returns merged, non-degenerate results ranked by match count (AC-3)", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		const results = await agent.retrieve("alpha");
		expect(results.length).toBe(2);
		// many.txt (2 matches) must rank above few.md (1 match) after min-max normalization
		expect(results[0].source).toBe("/docs/many.txt");
		expect(results[1].source).toBe("/docs/few.md");
		// posix method is registered and active
		expect(agent.getMethodRegistry().getByType("posix").length).toBe(1);
	});

	it("dedups by virtual source", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		const results = await agent.retrieve("alpha");
		const sources = results.map((r) => r.source);
		expect(new Set(sources).size).toBe(sources.length);
	});
});

describe("memory recording gate (AC-9)", () => {
	it("records only search tools (grep/find), not navigation/mutation tools", () => {
		// Mirrors the afterToolCall guard: !SEARCH_TOOLS.includes(toolName) => skip
		const records = (name: string) => (SEARCH_TOOLS as readonly string[]).includes(name);
		expect(records("grep")).toBe(true);
		expect(records("find")).toBe(true);
		for (const navOrMutate of ["read", "ls", "stat", "mv", "cp", "mkdir", "rmdir", "check_memory", "organize"]) {
			expect(records(navOrMutate)).toBe(false);
		}
	});
});
