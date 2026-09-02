import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parse } from "smol-toml";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSearchBM25DocumentsTool } from "../../src/agent/search-bm25-tool.ts";
import {
	MinSyncBM25Method,
	MinSyncHybridMethod,
	MinSyncVectorMethod,
	minSyncConfigPath,
} from "../../src/minsync/index.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";
import { BM25_SUBDIR, hasLegacyBm25Artifacts, removeLegacyBm25Artifacts } from "../../src/retrieval/methods/bm25.ts";
import { matchesVirtualPathScope, RetrievalScopeError } from "../../src/retrieval/scope.ts";

let root: string;
let docs: string;
let minsyncBinary: string;
let minsyncWorkspace: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-bm25-"));
	docs = join(root, "docs");
	minsyncWorkspace = join(root, ".autorag", "minsync");
	minsyncBinary = join(root, "fake-minsync.mjs");
	logPath = join(root, "minsync-calls.jsonl");
	mkdirSync(join(docs, "sub"), { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

async function refreshMirrors(): Promise<void> {
	await syncParsedMirrors({ root, searchPaths: [docs], force: true });
}

function writeFakeMinSync(): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, cwd: process.cwd() }) + "\\n");

if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}
if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}
if (args[0] === "sync") {
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ files_processed: 1 }));
  process.exit(0);
}
if (args[0] === "query") {
  const mode = args[args.indexOf("--mode") + 1] ?? "vector";
  const score = mode === "bm25" ? 0.91 : mode === "hybrid" ? 0.88 : 0.7;
  console.log(JSON.stringify({
    results: [
      { path: "files/docs/many.txt.md", score, text: "refund refund manager approval" },
      { path: "files/docs/sub/nested.txt.md", score: score - 0.2, text: "refund nested policy" },
    ],
  }));
  process.exit(0);
}
console.error("unexpected fake minsync command: " + args.join(" "));
process.exit(2);
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function minSyncOpts() {
	return { binaryPath: minsyncBinary, workspacePath: minsyncWorkspace, autoInstall: false as const };
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

describe("MinSync BM25/vector/hybrid", () => {
	it("shares canonical chunk IDs and original source paths across modes", async () => {
		writeFileSync(join(docs, "many.txt"), "refund refund manager approval\n");
		writeFileSync(join(docs, "sub", "nested.txt"), "refund nested policy\n");
		await refreshMirrors();
		writeFakeMinSync();
		const opts = { root, ...minSyncOpts() };
		const bm25 = new MinSyncBM25Method(opts);
		const vector = new MinSyncVectorMethod(opts);
		const hybrid = new MinSyncHybridMethod(opts);
		await bm25.syncFromMinSync(await bm25.sync());

		const [bm25Hits, vectorHits, hybridHits] = await Promise.all([
			bm25.retrieve("refund", { topK: 2 }),
			vector.retrieve("refund", { topK: 2 }),
			hybrid.retrieve("refund", { topK: 2 }),
		]);

		expect(bm25Hits.map((hit) => hit.id)).toEqual(vectorHits.map((hit) => hit.id));
		expect(hybridHits.map((hit) => hit.id)).toEqual(bm25Hits.map((hit) => hit.id));
		expect(bm25Hits[0]?.source).toBe(realpathSync(join(docs, "many.txt")));
		expect(existsSync(String(bm25Hits[0]?.source))).toBe(true);
		expect(readFileSync(String(bm25Hits[0]?.source), "utf8")).toContain("refund");
		expect(bm25Hits[0]?.metadata.minsyncChunkId).toBe(vectorHits[0]?.metadata.minsyncChunkId);
		expect(loggedModes()).toEqual(expect.arrayContaining(["bm25", "vector", "hybrid"]));
	});

	it("supports scoped retrieval over MinSync hits", async () => {
		writeFileSync(join(docs, "many.txt"), "refund refund manager approval\n");
		writeFileSync(join(docs, "sub", "nested.txt"), "refund nested policy\n");
		await refreshMirrors();
		writeFakeMinSync();
		const method = new MinSyncBM25Method({ root, ...minSyncOpts() });
		await method.syncFromMinSync(await method.sync());

		const scoped = await method.retrieve("refund", { topK: 10, scope: "/docs/sub" });
		expect(scoped.map((result) => result.source)).toEqual([realpathSync(join(docs, "sub", "nested.txt"))]);
	});

	it("reports missing MinSync visibly instead of writing a local BM25 index", async () => {
		writeFileSync(join(docs, "a.txt"), "alpha\n");
		await refreshMirrors();
		const method = new MinSyncBM25Method({
			root,
			binaryPath: join(root, "missing-minsync"),
			workspacePath: minsyncWorkspace,
			autoInstall: false,
		});

		const sync = await method.syncFromMinSync(await method.sync());

		expect(sync.readiness).toBe("dependency_unavailable");
		expect(sync.engine).toBe("minsync");
		expect(method.describe().status).toBe("stub");
		expect(existsSync(join(root, BM25_SUBDIR))).toBe(false);
		await expect(method.retrieve("alpha", { topK: 1 })).rejects.toMatchObject({
			readiness: "dependency_unavailable",
		});
	});
});

describe("AutoRAG BM25 integration", () => {
	it("registers MinSync BM25 and hybrid, never a local BM25 method", async () => {
		writeFileSync(join(docs, "guide.txt"), "chargeback chargeback process\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			bm25: { autoInstall: false },
			minSync: { autoInstall: false, binaryPath: join(root, "missing-minsync") },
		});
		const refresh = await agent.refresh(true);

		expect(refresh.bm25).toMatchObject({ engine: "minsync" });
		expect(agent.getMethodRegistry().getByType("bm25")).toHaveLength(1);
		expect(agent.getMethodRegistry().getByType("vector")).toHaveLength(1);
		expect(agent.getMethodRegistry().getByType("hybrid")).toHaveLength(1);
		expect(agent.getSystemPrompt()).toContain("lexical_search_local_docs");
		expect(existsSync(join(root, BM25_SUBDIR))).toBe(false);
		expect(readFileSync(new URL("../../src/retrieval/methods/bm25.ts", import.meta.url), "utf8")).not.toContain(
			"class BM25Method",
		);
	});

	it("applies minSync.maxChunkSize during a BM25-only refresh", async () => {
		writeFileSync(join(docs, "guide.txt"), "refund policy\n");
		writeFakeMinSync();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false, binaryPath: minsyncBinary, maxChunkSize: 1000 },
		});

		const refresh = await agent.refresh(true, { methods: ["bm25"] });

		expect(refresh.bm25).toMatchObject({ engine: "minsync" });
		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect((rewritten.chunker?.options as { max_chunk_size?: number } | undefined)?.max_chunk_size).toBe(1000);
	});

	it("does not fall back to a local BM25 index when MinSync is disabled", async () => {
		writeFileSync(join(docs, "guide.txt"), "chargeback chargeback process\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: false,
		});
		await agent.refresh(true);

		expect(agent.getMethodRegistry().getByType("bm25")).toHaveLength(0);
		expect(agent.getMethodRegistry().getByType("hybrid")).toHaveLength(0);
		expect(existsSync(join(root, BM25_SUBDIR))).toBe(false);
		const tool = createSearchBM25DocumentsTool(() => undefined);
		const result = await tool.execute("missing", { query: "chargeback" });
		expect(result.details).toMatchObject({ resultCount: 0, readiness: "disabled", engine: "none" });
	});

	it("removes leftover .autorag/bm25 artifacts on refresh", async () => {
		mkdirSync(join(root, BM25_SUBDIR), { recursive: true });
		writeFileSync(join(root, BM25_SUBDIR, "fallback-index.json"), "{}");
		expect(hasLegacyBm25Artifacts(root)).toBe(true);
		writeFileSync(join(docs, "guide.txt"), "chargeback\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { autoInstall: false, binaryPath: join(root, "missing-minsync") },
		});
		await agent.refresh(true);
		expect(hasLegacyBm25Artifacts(root)).toBe(false);
		removeLegacyBm25Artifacts(root);
		expect(hasLegacyBm25Artifacts(root)).toBe(false);
	});

	it("lexical_search_local_docs exposes method, readiness, engine, and original sources", async () => {
		writeFileSync(join(docs, "many.txt"), "chargeback chargeback process\n");
		writeFileSync(join(docs, "sub", "nested.txt"), "chargeback scoped process\n");
		await refreshMirrors();
		writeFakeMinSync();
		const method = new MinSyncBM25Method({ root, ...minSyncOpts() });
		await method.syncFromMinSync(await method.sync());
		const tool = createSearchBM25DocumentsTool(() => method);

		const result = await tool.execute("tool-call", { query: "chargeback", topK: 1, scope: "/docs/sub" });

		expect(result.details).toMatchObject({ method: "bm25", resultCount: 1, readiness: "ready", engine: "minsync" });
		expect(result.details?.sources).toEqual([realpathSync(join(docs, "sub", "nested.txt"))]);
		expect(result.content[0]?.type).toBe("text");
	});

	it("normalizes model-supplied physical scopes before BM25 retrieval", async () => {
		writeFileSync(join(docs, "many.txt"), "chargeback\n");
		writeFileSync(join(docs, "sub", "nested.txt"), "chargeback scoped process\n");
		await refreshMirrors();
		writeFakeMinSync();
		const method = new MinSyncBM25Method({ root, ...minSyncOpts() });
		await method.syncFromMinSync(await method.sync());
		const normalizeScope = vi.fn(() => "/docs/sub");
		const factory = createSearchBM25DocumentsTool as unknown as (
			getMethod: () => MinSyncBM25Method,
			resolveScope: typeof normalizeScope,
		) => ReturnType<typeof createSearchBM25DocumentsTool>;
		const tool = factory(() => method, normalizeScope);

		const result = await tool.execute("tool-scope", { query: "chargeback", scope: join(docs, "sub") });

		expect(normalizeScope).toHaveBeenCalledWith(join(docs, "sub"));
		expect(result.details.sources).toEqual([realpathSync(join(docs, "sub", "nested.txt"))]);
	});

	it("preserves coded scope errors at the BM25 tool boundary", async () => {
		const method = new MinSyncBM25Method({
			root,
			binaryPath: join(root, "missing-minsync"),
			workspacePath: minsyncWorkspace,
			autoInstall: false,
		});
		const tool = createSearchBM25DocumentsTool(
			() => method,
			() => {
				throw new RetrievalScopeError(["/docs"]);
			},
		);

		await expect(
			tool.execute("tool-invalid-scope", { query: "chargeback", scope: "/outside" }),
		).rejects.toMatchObject({
			code: "invalid-retrieval-scope",
		});
	});
});

function loggedModes(): string[] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.map((line) => JSON.parse(line) as { args: string[] })
		.filter((entry) => entry.args[0] === "query")
		.map((entry) => entry.args[entry.args.indexOf("--mode") + 1]);
}
