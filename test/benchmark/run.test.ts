import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	realpathSync,
	renameSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runBm25Queries, runMethodQueries } from "../../benchmark/miracl/run.ts";
import { materializeBenchmarkWorkspace } from "../../benchmark/miracl/workspace.ts";
import { loadMirrorIndex, type ParsedMirrorEntry, saveMirrorIndex } from "../../src/mirror/index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "../../src/mirror/paths.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

function deterministicClock(values: readonly number[]): () => number {
	let index = 0;
	return () => {
		const value = values[index];
		if (value === undefined) throw new Error("deterministic clock exhausted");
		index += 1;
		return value;
	};
}

function retrievalMethod(retrieve: RetrievalMethod["retrieve"], name = "bm25"): RetrievalMethod {
	return {
		describe: () => ({
			name,
			type: "bm25",
			description: "benchmark test retrieval",
			status: "active",
			capabilities: [],
		}),
		retrieve,
	};
}

function createBenchmarkShapedWorkspace(
	root: string,
	documentIds: readonly string[] = ["doc"],
): ReadonlyMap<string, string> {
	mkdirSync(root, { recursive: true });
	const entries: Record<string, ParsedMirrorEntry> = {};
	const documentBySource = new Map<string, string>();
	for (const documentId of documentIds) {
		const virtualPath = `/miracl/${encodeURIComponent(documentId)}.md`;
		const outputPath = parsedOutputPath(root, virtualPath);
		mkdirSync(dirname(outputPath), { recursive: true });
		writeFileSync(outputPath, `# ${documentId}\n\nquery\n`);
		entries[virtualPath] = {
			virtualPath,
			sourcePath: virtualPath,
			outputPath,
			parserName: "miracl-benchmark",
			sourceMtimeNs: 0,
			sourceSizeBytes: 13,
			updatedAt: "1970-01-01T00:00:00.000Z",
		};
		documentBySource.set(virtualPath, documentId);
	}
	saveMirrorIndex(root, { version: 1, entries });
	return documentBySource;
}

describe("MIRACL benchmark workspace and runner", () => {
	const temporaryRoots: string[] = [];
	const makeParent = () => {
		const parent = mkdtempSync(join(tmpdir(), "autorag-miracl-run-"));
		temporaryRoots.push(parent);
		return parent;
	};

	afterEach(() => {
		for (const root of temporaryRoots.splice(0)) {
			rmSync(root, { recursive: true, force: true });
		}
		vi.restoreAllMocks();
		vi.doUnmock("node:fs");
		vi.resetModules();
	});

	it("materializes document ids as reversible virtual paths using the production mirror index", () => {
		const parent = makeParent();
		const root = join(parent, "workspace");

		const result = materializeBenchmarkWorkspace(root, [
			{ documentId: "123#4", title: "제목", text: "검색 가능한 본문" },
			{ documentId: "123%234", title: "다른 제목", text: "다른 본문" },
		]);

		expect(result.root).toBe(join(realpathSync(parent), "workspace"));
		expect(result.documentBySource.get("/miracl/123%234.md")).toBe("123#4");
		expect(result.documentBySource.get("/miracl/123%25234.md")).toBe("123%234");
		expect(result.mirrorFiles).toHaveLength(2);
		expect(readFileSync(result.mirrorFiles[0] as string, "utf8")).toBe("# 제목\n\n검색 가능한 본문\n");
		expect(Object.keys(loadMirrorIndex(root).entries).sort()).toEqual(["/miracl/123%234.md", "/miracl/123%25234.md"]);
	});

	it("rejects and preserves a pre-existing workspace root", () => {
		const parent = makeParent();
		const root = join(parent, "workspace");
		mkdirSync(root);
		const sentinel = join(root, "keep.txt");
		writeFileSync(sentinel, "user data\n");

		expect(() => materializeBenchmarkWorkspace(root, [{ documentId: "doc", title: "제목", text: "본문" }])).toThrow(
			"must not already exist",
		);
		expect(readFileSync(sentinel, "utf8")).toBe("user data\n");
		expect(existsSync(join(root, ".autorag"))).toBe(false);
	});

	it("rejects a root whose symlinked parent resolves inside an existing .autorag", () => {
		const parent = makeParent();
		const sourceCheckout = join(parent, "source-checkout");
		const userIndex = join(sourceCheckout, ".autorag");
		const alias = join(parent, "index-alias");
		mkdirSync(userIndex, { recursive: true });
		const sentinel = join(userIndex, "keep.txt");
		writeFileSync(sentinel, "user index\n");
		symlinkSync(userIndex, alias, "dir");

		expect(() =>
			materializeBenchmarkWorkspace(join(alias, "benchmark"), [{ documentId: "doc", title: "제목", text: "본문" }]),
		).toThrow("existing .autorag");
		expect(readFileSync(sentinel, "utf8")).toBe("user index\n");
		expect(existsSync(join(userIndex, "benchmark"))).toBe(false);
	});

	it.each([".autorag", ".AuToRaG"])("rejects an absent workspace root named %s", (directoryName) => {
		const parent = makeParent();
		const root = join(parent, directoryName);

		expect(() => materializeBenchmarkWorkspace(root, [{ documentId: "doc", title: "제목", text: "본문" }])).toThrow(
			"existing .autorag",
		);
		expect(existsSync(root)).toBe(false);
	});

	it("rejects a path through a checkout .autorag symlink whose canonical target loses the reserved name", () => {
		const parent = makeParent();
		const checkout = join(parent, "checkout");
		const indexTarget = join(parent, "index-target");
		const autoragLink = join(checkout, ".autorag");
		mkdirSync(checkout);
		mkdirSync(indexTarget);
		symlinkSync(indexTarget, autoragLink, "dir");

		expect(() =>
			materializeBenchmarkWorkspace(join(autoragLink, "benchmark"), [
				{ documentId: "doc", title: "제목", text: "본문" },
			]),
		).toThrow("existing .autorag");
		expect(readdirSync(indexTarget)).toEqual([]);
	});

	it("checks a symlink target's case-insensitive .autorag components", () => {
		const parent = makeParent();
		const checkout = join(parent, "checkout");
		const userIndex = join(checkout, ".AUTORAG");
		const alias = join(parent, "alias");
		mkdirSync(userIndex, { recursive: true });
		symlinkSync(userIndex, alias, "dir");

		expect(() =>
			materializeBenchmarkWorkspace(join(alias, "benchmark"), [{ documentId: "doc", title: "제목", text: "본문" }]),
		).toThrow("existing .autorag");
		expect(readdirSync(userIndex)).toEqual([]);
	});

	it("checks every target in a chained symlink path for .autorag components", () => {
		const parent = makeParent();
		const checkout = join(parent, "checkout");
		const indexTarget = join(parent, "index-target");
		const autoragLink = join(checkout, ".autorag");
		const intermediateAlias = join(parent, "intermediate-alias");
		const outerAlias = join(parent, "outer-alias");
		mkdirSync(checkout);
		mkdirSync(indexTarget);
		symlinkSync(indexTarget, autoragLink, "dir");
		symlinkSync(autoragLink, intermediateAlias, "dir");
		symlinkSync(intermediateAlias, outerAlias, "dir");

		expect(() =>
			materializeBenchmarkWorkspace(join(outerAlias, "benchmark"), [
				{ documentId: "doc", title: "제목", text: "본문" },
			]),
		).toThrow("existing .autorag");
		expect(readdirSync(indexTarget)).toEqual([]);
	});

	it("rejects a benchmark-shaped root beneath a symlinked checkout .autorag before BM25 writes", async () => {
		const parent = realpathSync(makeParent());
		const checkout = join(parent, "checkout");
		const reservedTarget = join(parent, "reserved-target");
		const root = join(reservedTarget, "workspace");
		const autoragLink = join(checkout, ".autorag");
		mkdirSync(checkout);
		mkdirSync(reservedTarget);
		const documentBySource = createBenchmarkShapedWorkspace(root);
		const sentinel = join(reservedTarget, "keep.txt");
		writeFileSync(sentinel, "reserved data\n");
		symlinkSync(reservedTarget, autoragLink, "dir");

		await expect(
			runBm25Queries({
				root: join(autoragLink, "workspace"),
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource,
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("existing .autorag");
		expect(readFileSync(sentinel, "utf8")).toBe("reserved data\n");
		expect(readdirSync(join(root, ".autorag")).sort()).toEqual(["parsed"]);
	});

	it("rejects a benchmark-shaped root beneath a chained .autorag symlink target before BM25 writes", async () => {
		const parent = realpathSync(makeParent());
		const checkout = join(parent, "checkout");
		const reservedTarget = join(parent, "reserved-target");
		const root = join(reservedTarget, "workspace");
		const autoragLink = join(checkout, ".AUTORAG");
		const intermediateAlias = join(parent, "intermediate-alias");
		const outerAlias = join(parent, "outer-alias");
		mkdirSync(checkout);
		mkdirSync(reservedTarget);
		const documentBySource = createBenchmarkShapedWorkspace(root);
		const sentinel = join(reservedTarget, "keep.txt");
		writeFileSync(sentinel, "reserved data\n");
		symlinkSync(reservedTarget, autoragLink, "dir");
		symlinkSync(autoragLink, intermediateAlias, "dir");
		symlinkSync(intermediateAlias, outerAlias, "dir");

		await expect(
			runBm25Queries({
				root: join(outerAlias, "workspace"),
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource,
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("existing .autorag");
		expect(readFileSync(sentinel, "utf8")).toBe("reserved data\n");
		expect(readdirSync(join(root, ".autorag")).sort()).toEqual(["parsed"]);
	});

	it("does not recursively clean a replacement at the workspace pathname", async () => {
		const parent = makeParent();
		const root = join(parent, "workspace");
		const displacedRoot = join(parent, "displaced-workspace");
		const replacementSentinel = join(root, "replacement.txt");
		const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
		let replaced = false;
		vi.doMock("node:fs", () => ({
			...actual,
			writeFileSync: (path: Parameters<typeof actual.writeFileSync>[0], ...args: unknown[]) => {
				if (!replaced && String(path).includes(join(".autorag", "parsed", "files"))) {
					replaced = true;
					renameSync(root, displacedRoot);
					mkdirSync(root);
					writeFileSync(replacementSentinel, "replacement data\n");
					throw new Error("injected mirror write failure");
				}
				return Reflect.apply(actual.writeFileSync, actual, [path, ...args]);
			},
		}));
		const { materializeBenchmarkWorkspace: materializeWithRace } = await import(
			"../../benchmark/miracl/workspace.ts"
		);

		expect(() => materializeWithRace(root, [{ documentId: "doc", title: "제목", text: "본문" }])).toThrow(
			"injected mirror write failure",
		);
		expect(readFileSync(replacementSentinel, "utf8")).toBe("replacement data\n");
		expect(existsSync(displacedRoot)).toBe(true);
	});

	it("rejects a symlink inserted into a parsed-mirror parent before the exclusive write", async () => {
		const parent = makeParent();
		const root = join(parent, "workspace");
		const externalTarget = join(parent, "external-target");
		mkdirSync(externalTarget);
		const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
		let inserted = false;
		vi.doMock("node:fs", () => ({
			...actual,
			mkdirSync: (path: Parameters<typeof actual.mkdirSync>[0], ...args: unknown[]) => {
				const result = Reflect.apply(actual.mkdirSync, actual, [path, ...args]);
				const pathText = String(path);
				if (!inserted && pathText.includes(join(".autorag", "parsed", "files"))) {
					inserted = true;
					actual.renameSync(pathText, `${pathText}-owned`);
					actual.symlinkSync(externalTarget, pathText, "dir");
				}
				return result;
			},
		}));
		const { materializeBenchmarkWorkspace: materializeWithRace } = await import(
			"../../benchmark/miracl/workspace.ts"
		);

		expect(() => materializeWithRace(root, [{ documentId: "doc", title: "제목", text: "본문" }])).toThrow(
			"workspace directory changed",
		);
		expect(readdirSync(externalTarget)).toEqual([]);
	});

	it("constructs, synchronizes, and runs real BM25 with separate indexing timing", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "123#4", title: "제목", text: "검색 가능한 본문" },
			{ documentId: "other", title: "기타", text: "무관한 내용" },
		]);

		const result = await runBm25Queries({
			root: workspace.root,
			queries: [{ queryId: "q1", text: "검색 가능한" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
			now: deterministicClock([0, 3, 10, 14]),
			bm25: { forceEngine: "typescript-fallback" },
		});

		expect(result.indexingLatencyMs).toBe(3);
		expect(result.records).toEqual([
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q1",
				latencyMs: 4,
				hits: [
					expect.objectContaining({
						documentId: "123#4",
						rank: 1,
					}),
				],
			},
		]);
		expect(existsSync(join(workspace.root, ".autorag", "bm25", "fallback-index.json"))).toBe(true);
	});

	it("rejects a workspace root replaced during async BM25 synchronization", async () => {
		const parent = realpathSync(makeParent());
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const displacedRoot = join(parent, "displaced-workspace");
		const replacementSentinel = join(workspace.root, "replacement.txt");
		const retrieve = vi.fn(async () => []);
		const factory = vi.fn(() => ({
			describe: () => ({
				name: "bm25",
				type: "bm25" as const,
				description: "injected BM25",
				status: "active" as const,
				capabilities: [],
			}),
			sync: async () => {
				await Promise.resolve();
				renameSync(workspace.root, displacedRoot);
				mkdirSync(workspace.root);
				writeFileSync(replacementSentinel, "replacement data\n");
				return {
					indexPath: join(displacedRoot, ".autorag", "bm25"),
					indexedChunks: 1,
					readiness: "degraded_fallback" as const,
					engine: "typescript-fallback" as const,
				};
			},
			retrieve,
		}));

		await expect(
			runBm25Queries(
				{
					root: workspace.root,
					queries: [{ queryId: "q1", text: "query" }],
					documentBySource: workspace.documentBySource,
					topK: 5,
				},
				factory,
			),
		).rejects.toThrow("workspace root changed");
		expect(readFileSync(replacementSentinel, "utf8")).toBe("replacement data\n");
		expect(retrieve).not.toHaveBeenCalled();
		expect(existsSync(join(workspace.root, ".autorag"))).toBe(false);
		expect(existsSync(displacedRoot)).toBe(true);
	});

	it("keeps generic query execution independent from descriptor readiness", async () => {
		const retrieve = vi.fn(async () => [
			{ id: "hit", source: "/miracl/hit.md", content: "hit", score: 1, metadata: {} },
		]);
		const retrieval: RetrievalMethod = {
			describe: () => {
				throw new Error("generic runner must not infer lifecycle readiness");
			},
			retrieve,
		};

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: new Map([["/miracl/hit.md", "hit"]]),
			topK: 5,
			now: deterministicClock([0, 1]),
		});

		expect(records[0]?.hits).toEqual([{ documentId: "hit", score: 1, rank: 1 }]);
		expect(retrieve).toHaveBeenCalledTimes(1);
	});

	it("runs benchmark integrity hooks outside the measured retrieval interval", async () => {
		let clock = 0;
		const events: string[] = [];
		const retrieval: RetrievalMethod & {
			beforeBenchmarkBatch(): Promise<void>;
			beforeBenchmarkQuery(): Promise<void>;
			afterBenchmarkQuery(): Promise<void>;
			afterBenchmarkBatch(): Promise<void>;
		} = {
			...retrievalMethod(async () => {
				events.push("retrieve");
				clock += 4;
				return [{ id: "hit", source: "/miracl/hit.md", content: "hit", score: 1, metadata: {} }];
			}),
			beforeBenchmarkBatch: async () => {
				events.push("before-batch");
				clock += 100;
			},
			beforeBenchmarkQuery: async () => {
				events.push("before-query");
				clock += 10;
			},
			afterBenchmarkQuery: async () => {
				events.push("after-query");
				clock += 20;
			},
			afterBenchmarkBatch: async () => {
				events.push("after-batch");
				clock += 200;
			},
		};

		const records = await runMethodQueries({
			method: "minsync",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: new Map([["/miracl/hit.md", "hit"]]),
			topK: 5,
			now: () => clock,
		});

		expect(events).toEqual(["before-batch", "before-query", "retrieve", "after-query", "after-batch"]);
		expect(records[0]?.latencyMs).toBe(4);
		expect(records[0]?.errorCode).toBeUndefined();
	});

	it("invalidates completed query records when the post-batch integrity hook fails", async () => {
		const retrieve = vi.fn(async () => [
			{ id: "hit", source: "/miracl/hit.md", content: "hit", score: 1, metadata: {} },
		]);
		const retrieval: RetrievalMethod & { afterBenchmarkBatch(): Promise<void> } = {
			...retrievalMethod(retrieve),
			afterBenchmarkBatch: async () => {
				throw new Error("private integrity detail");
			},
		};

		const records = await runMethodQueries({
			method: "minsync",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: new Map([["/miracl/hit.md", "hit"]]),
			topK: 5,
			now: deterministicClock([0, 3]),
		});

		expect(records).toEqual([
			{
				schemaVersion: 1,
				method: "minsync",
				queryId: "q1",
				latencyMs: 3,
				hits: [],
				errorCode: "retrieval-failed",
			},
		]);
		expect(JSON.stringify(records)).not.toContain("private integrity detail");
	});

	it("rejects a wrong workspace root before BM25 synchronization", async () => {
		const parent = makeParent();
		const root = join(realpathSync(parent), "user-checkout");
		const virtualPath = "/docs/user.txt";
		const outputPath = parsedOutputPath(root, virtualPath);
		mkdirSync(root);
		mkdirSync(join(root, ".autorag", "parsed", "files"), { recursive: true });
		writeFileSync(outputPath, "user mirror\n");
		saveMirrorIndex(root, {
			version: 1,
			entries: {
				[virtualPath]: {
					virtualPath,
					sourcePath: join(root, "docs", "user.txt"),
					outputPath,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 12,
					updatedAt: "2026-07-24T00:00:00.000Z",
				},
			},
		});

		await expect(
			runBm25Queries({
				root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: new Map([[virtualPath, "user-doc"]]),
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("valid parsed mirror");
		expect(existsSync(join(root, ".autorag", "bm25"))).toBe(false);
	});

	it("rejects stale parsed-mirror entries before BM25 synchronization", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		rmSync(workspace.mirrorFiles[0] as string);

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: workspace.documentBySource,
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("stale");
		expect(existsSync(join(workspace.root, ".autorag", "bm25"))).toBe(false);
	});

	it("rejects a corrupt parsed-mirror index before BM25 synchronization", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		writeFileSync(parsedMirrorIndexPath(workspace.root), "{not-json");

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: workspace.documentBySource,
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("valid parsed mirror");
		expect(existsSync(join(workspace.root, ".autorag", "bm25"))).toBe(false);
	});

	it("treats BM25 synchronization failure as fatal before queries", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: workspace.documentBySource,
				topK: 5,
				now: deterministicClock([0, 5]),
				bm25: {
					fallback: "disabled",
					importBinding: async () => {
						throw new Error("binding unavailable");
					},
				},
			}),
		).rejects.toThrow("BM25 benchmark indexing failed");
	});

	it("rejects a source map whose document id does not encode to its virtual path", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const source = [...workspace.documentBySource.keys()][0] as string;

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: new Map([[source, "wrong-id"]]),
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("bijective");
		expect(existsSync(join(workspace.root, ".autorag", "bm25"))).toBe(false);
	});

	it("rejects duplicate document ids in the source map before BM25 synchronization", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "first", title: "첫째", text: "query" },
			{ documentId: "second", title: "둘째", text: "query" },
		]);
		const sources = [...workspace.documentBySource.keys()];

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: new Map([
					[sources[0] as string, "first"],
					[sources[1] as string, "first"],
				]),
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("bijective");
		expect(existsSync(join(workspace.root, ".autorag", "bm25"))).toBe(false);
	});

	it("rejects blank document ids in the source map before BM25 synchronization", async () => {
		const parent = makeParent();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const source = [...workspace.documentBySource.keys()][0] as string;

		await expect(
			runBm25Queries({
				root: workspace.root,
				queries: [{ queryId: "q1", text: "query" }],
				documentBySource: new Map([[source, " "]]),
				topK: 5,
				bm25: { forceEngine: "typescript-fallback" },
			}),
		).rejects.toThrow("bijective");
		expect(existsSync(join(workspace.root, ".autorag", "bm25"))).toBe(false);
	});

	it("requests 100 candidates and emits stable document-level rankings", async () => {
		const calls: Array<{ query: string; topK: number | undefined }> = [];
		const results: RetrievalResult[] = [
			{ id: "a-low", source: "/miracl/a.md", content: "a1", score: 1, metadata: {} },
			{ id: "b", source: "/miracl/b.md", content: "b", score: 3, metadata: {} },
			{ id: "a-high", source: "/miracl/a.md", content: "a2", score: 3, metadata: {} },
			{ id: "c", source: "/miracl/c.md", content: "c", score: 3, metadata: {} },
		];
		const retrieval = retrievalMethod(async (query, options) => {
			calls.push({ query, topK: options.topK });
			return results;
		});

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: new Map([
				["/miracl/a.md", "a"],
				["/miracl/b.md", "b"],
				["/miracl/c.md", "c"],
			]),
			topK: 2,
			now: deterministicClock([10, 12]),
		});

		expect(calls).toEqual([{ query: "query", topK: 100 }]);
		expect(records[0]?.hits).toEqual([
			{ documentId: "a", score: 3, rank: 1 },
			{ documentId: "b", score: 3, rank: 2 },
		]);
	});

	it("overfetches bounded chunk rankings until Recall@100 has 100 unique documents", async () => {
		const allResults: RetrievalResult[] = [
			...Array.from({ length: 100 }, (_, index) => ({
				id: `duplicate:${index}`,
				source: "/miracl/duplicate.md",
				content: `duplicate ${index}`,
				score: 1_000 - index,
				metadata: {},
			})),
			...Array.from({ length: 99 }, (_, index) => ({
				id: `unique:${index}`,
				source: `/miracl/doc-${index}.md`,
				content: `unique ${index}`,
				score: 899 - index,
				metadata: {},
			})),
		];
		const requested: number[] = [];
		const retrieval = retrievalMethod(async (_query, options) => {
			requested.push(options.topK ?? 0);
			return allResults.slice(0, options.topK);
		});
		const documentBySource = new Map<string, string>([
			["/miracl/duplicate.md", "duplicate"],
			...Array.from({ length: 99 }, (_, index) => [`/miracl/doc-${index}.md`, `doc-${index}`] as const),
		]);

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource,
			topK: 100,
		});

		expect(requested).toEqual([100, 200]);
		expect(records[0]?.hits).toHaveLength(100);
		expect(new Set(records[0]?.hits.map((hit) => hit.documentId)).size).toBe(100);
		expect(records[0]?.hits[0]).toMatchObject({ documentId: "duplicate", score: 1_000, rank: 1 });
	});

	it("records opaque query failures durably and continues sequentially", async () => {
		let inFlight = 0;
		let maximumInFlight = 0;
		const retrieval = retrievalMethod(async (query) => {
			inFlight += 1;
			maximumInFlight = Math.max(maximumInFlight, inFlight);
			try {
				await Promise.resolve();
				if (query === "first") throw new Error("secret endpoint and filesystem path");
				return [{ id: "ok", source: "/miracl/ok.md", content: "ok", score: 1, metadata: {} }];
			} finally {
				inFlight -= 1;
			}
		});

		const records = await runMethodQueries({
			method: "bm25",
			retrieval,
			queries: [
				{ queryId: "q1", text: "first" },
				{ queryId: "q2", text: "second" },
			],
			documentBySource: new Map([["/miracl/ok.md", "ok"]]),
			topK: 5,
			now: deterministicClock([0, 2, 3, 7]),
		});

		expect(records).toEqual([
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q1",
				latencyMs: 2,
				hits: [],
				errorCode: "retrieval-failed",
			},
			{
				schemaVersion: 1,
				method: "bm25",
				queryId: "q2",
				latencyMs: 4,
				hits: [{ documentId: "ok", score: 1, rank: 1 }],
			},
		]);
		expect(maximumInFlight).toBe(1);
		expect(JSON.stringify(records)).not.toContain("secret endpoint");
		expect(JSON.stringify(records)).not.toContain("filesystem path");
	});
});
