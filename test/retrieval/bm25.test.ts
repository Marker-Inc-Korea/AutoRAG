import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSearchBM25DocumentsTool } from "../../src/agent/search-bm25-tool.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25Method } from "../../src/retrieval/methods/bm25.ts";
import { matchesVirtualPathScope } from "../../src/retrieval/scope.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-bm25-"));
	docs = join(root, "docs");
	mkdirSync(join(docs, "sub"), { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

async function refreshMirrors(): Promise<void> {
	await syncParsedMirrors({ root, searchPaths: [docs], force: true });
}

describe("virtual path scope helpers", () => {
	it("supports unrestricted, folder, file, single-segment, and recursive scopes", () => {
		expect(matchesVirtualPathScope("/docs/a.md", undefined)).toBe(true);
		expect(matchesVirtualPathScope("/docs/a.md", "")).toBe(true);
		expect(matchesVirtualPathScope("/docs/sub/a.md", "docs")).toBe(true);
		expect(matchesVirtualPathScope("/docs/sub/a.md", "/docs")).toBe(true);
		expect(matchesVirtualPathScope("/docs/sub/a.md", "/docs/*.md")).toBe(false);
		expect(matchesVirtualPathScope("/docs/a.md", "/docs/*.md")).toBe(true);
		expect(matchesVirtualPathScope("/docs/sub/a.md", "/docs/**")).toBe(true);
		expect(matchesVirtualPathScope("/docs/a.md", "/docs/a.md")).toBe(true);
		expect(matchesVirtualPathScope("/docs/b.md", "/docs/a.md")).toBe(false);
	});
});

describe("BM25Method", () => {
	it("indexes parsed mirrors with Tantivy and ranks stronger lexical matches higher", async () => {
		writeFileSync(join(docs, "many.txt"), "refund refund manager approval\n");
		writeFileSync(join(docs, "few.txt"), "refund policy\n");
		await refreshMirrors();
		const method = new BM25Method({ root, indexPath: join(root, ".autorag", "bm25-test"), fallback: "disabled" });

		const sync = await method.sync();
		const results = await method.retrieve("refund", { topK: 2 });

		expect(sync.readiness).toBe("ready");
		expect(sync.engine).toBe("tantivy");
		expect(method.describe().type).toBe("bm25");
		expect(method.describe().status).toBe("active");
		expect(results.map((result) => result.source)).toEqual(["/docs/many.txt", "/docs/few.txt"]);
		expect(results[0]?.metadata.method).toBe("bm25");
		expect(JSON.stringify(results)).not.toContain(root);
	});

	it("continues native Tantivy scanning until scoped hits are found", async () => {
		mkdirSync(join(docs, "outside"), { recursive: true });
		for (let index = 0; index < 105; index += 1) {
			writeFileSync(
				join(docs, "outside", `out-${index}.txt`),
				`chargeback chargeback chargeback outside ${index}\n`,
			);
		}
		writeFileSync(join(docs, "sub", "target.txt"), "chargeback scoped target\n");
		await refreshMirrors();
		const method = new BM25Method({
			root,
			indexPath: join(root, ".autorag", "bm25-scope-native"),
			fallback: "disabled",
		});
		await method.sync();

		const results = await method.retrieve("chargeback", { topK: 1, scope: "/docs/sub" });

		expect(results).toHaveLength(1);
		expect(results[0]?.source).toBe("/docs/sub/target.txt");
	});

	it("supports scoped retrieval, folder scope expansion, and stale-index removal", async () => {
		writeFileSync(join(docs, "root.txt"), "alpha root marker\n");
		writeFileSync(join(docs, "sub", "nested.txt"), "alpha nested marker\n");
		await refreshMirrors();
		const method = new BM25Method({
			root,
			indexPath: join(root, ".autorag", "bm25-fallback"),
			forceEngine: "typescript-fallback",
		});
		await method.sync();

		const scoped = await method.retrieve("alpha", { topK: 10, scope: "/docs/sub" });
		expect(scoped.map((result) => result.source)).toEqual(["/docs/sub/nested.txt"]);

		rmSync(join(docs, "sub", "nested.txt"));
		await refreshMirrors();
		await method.sync();
		const afterDelete = await method.retrieve("nested", { topK: 10 });
		expect(afterDelete).toHaveLength(0);
	});

	it("reports dependency unavailable visibly instead of silent empty success", async () => {
		writeFileSync(join(docs, "a.txt"), "alpha\n");
		await refreshMirrors();
		const method = new BM25Method({
			root,
			indexPath: join(root, ".autorag", "bm25-unavailable"),
			fallback: "disabled",
			importBinding: async () => {
				throw new Error("native binding missing");
			},
		});

		const sync = await method.sync();

		expect(sync.readiness).toBe("dependency_unavailable");
		expect(method.describe().status).toBe("stub");
		expect(method.describe().capabilities).toContain("readiness:dependency_unavailable");
		await expect(method.retrieve("alpha", { topK: 1 })).rejects.toMatchObject({
			readiness: "dependency_unavailable",
		});
	});
});

describe("AutoRAG BM25 integration", () => {
	it("registers BM25 and includes it in programmatic retrieve()", async () => {
		writeFileSync(join(docs, "guide.txt"), "chargeback chargeback process\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			bm25: { indexPath: join(root, ".autorag", "agent-bm25"), forceEngine: "typescript-fallback" },
		});
		const refresh = await agent.refresh(true);

		expect(refresh.bm25).toMatchObject({ readiness: "degraded_fallback", engine: "typescript-fallback" });
		const results = await agent.retrieve("chargeback", { topK: 1 });

		expect(agent.getMethodRegistry().getByType("bm25")).toHaveLength(1);
		expect(results[0]?.source).toBe("/docs/guide.txt");
		expect(results[0]?.metadata.method).toBe("bm25");
		expect(agent.getSystemPrompt()).toContain("search_bm25_documents");
	});

	it("search_bm25_documents exposes method, readiness, engine, and sources details", async () => {
		writeFileSync(join(docs, "guide.txt"), "chargeback chargeback process\n");
		writeFileSync(join(docs, "sub", "guide.txt"), "chargeback scoped process\n");
		await refreshMirrors();
		const method = new BM25Method({
			root,
			indexPath: join(root, ".autorag", "tool-bm25"),
			forceEngine: "typescript-fallback",
		});
		await method.sync();
		const tool = createSearchBM25DocumentsTool(() => method);

		const result = await tool.execute("tool-call", { query: "chargeback", topK: 1, scope: "/docs/sub" });

		expect(result.details).toMatchObject({ method: "bm25", resultCount: 1, readiness: "degraded_fallback" });
		expect(result.details?.sources).toEqual(["/docs/sub/guide.txt"]);
		expect(result.content[0]?.type).toBe("text");
	});
});
