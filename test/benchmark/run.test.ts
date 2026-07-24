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
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runBm25Queries, runMethodQueries } from "../../benchmark/miracl/run.ts";
import { materializeBenchmarkWorkspace } from "../../benchmark/miracl/workspace.ts";
import { loadMirrorIndex, saveMirrorIndex } from "../../src/mirror/index-store.ts";
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

	it("keeps generic query execution independent from descriptor readiness", async () => {
		const retrieve = vi.fn(async () => [{ id: "hit", source: "/hit", content: "hit", score: 1, metadata: {} }]);
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
			documentBySource: new Map([["/hit", "hit"]]),
			topK: 5,
			now: deterministicClock([0, 1]),
		});

		expect(records[0]?.hits).toEqual([{ documentId: "hit", score: 1, rank: 1 }]);
		expect(retrieve).toHaveBeenCalledTimes(1);
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

	it("requests 100 candidates and emits stable document-level rankings", async () => {
		const calls: Array<{ query: string; topK: number | undefined }> = [];
		const results: RetrievalResult[] = [
			{ id: "a-low", source: "/a", content: "a1", score: 1, metadata: {} },
			{ id: "b", source: "/b", content: "b", score: 3, metadata: {} },
			{ id: "a-high", source: "/a", content: "a2", score: 3, metadata: {} },
			{ id: "c", source: "/c", content: "c", score: 3, metadata: {} },
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
				["/a", "a"],
				["/b", "b"],
				["/c", "c"],
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

	it("records opaque query failures durably and continues sequentially", async () => {
		let inFlight = 0;
		let maximumInFlight = 0;
		const retrieval = retrievalMethod(async (query) => {
			inFlight += 1;
			maximumInFlight = Math.max(maximumInFlight, inFlight);
			try {
				await Promise.resolve();
				if (query === "first") throw new Error("secret endpoint and filesystem path");
				return [{ id: "ok", source: "/ok", content: "ok", score: 1, metadata: {} }];
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
			documentBySource: new Map([["/ok", "ok"]]),
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
