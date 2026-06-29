import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { PosixMethod } from "../../src/retrieval/methods/posix.ts";

let root: string;
let source: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-posix-"));
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
	writeFileSync(join(source, "many.txt"), "alpha alpha\nalpha\n");
	writeFileSync(join(source, "few.md"), "alpha\nbeta\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("PosixMethod", () => {
	it("describes itself as an active posix method", () => {
		const method = new PosixMethod({ root, searchPaths: [source] });
		const d = method.describe();
		expect(d.name).toBe("posix");
		expect(d.type).toBe("posix");
		expect(d.status).toBe("active");
	});

	it("maps grep hits from real directories to opaque root-relative sources", async () => {
		const method = new PosixMethod({ root, searchPaths: [source] });
		const results = await method.retrieve("alpha", {});
		expect(results.length).toBe(2);
		expect(results[0].source).toBe("/docs/many.txt");
		expect(results[0].score).toBeGreaterThan(results[1].score);
		expect(JSON.stringify(results)).not.toContain(source);
		expect(results[0].metadata.method).toBe("posix");
	});

	it("treats bare folder scopes as descendant scopes", async () => {
		const method = new PosixMethod({ root, searchPaths: [source] });
		const results = await method.retrieve("alpha", { scope: "/docs" });
		expect(results.map((result) => result.source).sort()).toEqual(["/docs/few.md", "/docs/many.txt"]);
	});

	it("uses stable source-root prefixes for directories outside the workspace root", async () => {
		const externalRoot = mkdtempSync(join(tmpdir(), "autorag-posix-external-"));
		try {
			const externalDocs = join(externalRoot, "docs");
			mkdirSync(join(externalDocs, "sub"), { recursive: true });
			writeFileSync(join(externalDocs, "root.txt"), "gamma root\n");
			writeFileSync(join(externalDocs, "sub", "report.md"), "gamma nested\n");
			const method = new PosixMethod({ root, searchPaths: [externalDocs] });

			const results = await method.retrieve("gamma", {});
			const sources = results.map((result) => result.source).sort();

			expect(sources).toEqual(["/docs/root.txt", "/docs/sub/report.md"]);
			expect(JSON.stringify(results)).not.toContain(externalRoot);
		} finally {
			rmSync(externalRoot, { recursive: true, force: true });
		}
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
		const records = (name: string) => name === "grep" || name === "find" || name === "search_bm25_documents";
		expect(records("grep")).toBe(true);
		expect(records("find")).toBe(true);
		expect(records("search_bm25_documents")).toBe(true);
		for (const navOrMutate of ["read", "ls", "stat", "mv", "cp", "mkdir", "rmdir", "check_memory"]) {
			expect(records(navOrMutate)).toBe(false);
		}
	});
});
