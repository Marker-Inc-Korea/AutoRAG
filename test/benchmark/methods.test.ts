import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { CreateBenchmarkMethodsOptions } from "../../benchmark/miracl/methods.ts";
import {
	createBenchmarkMethods,
	loadBenchmarkConfig,
	retrieveHybrid,
	sanitizeMethodConfig,
} from "../../benchmark/miracl/methods.ts";
import { runMethodQueries } from "../../benchmark/miracl/run.ts";
import { materializeBenchmarkWorkspace } from "../../benchmark/miracl/workspace.ts";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

type AssertFalse<T extends false> = T;
type AssertTrue<T extends true> = T;
type _NoPublicEnvironmentBypass = AssertFalse<"env" extends keyof CreateBenchmarkMethodsOptions ? true : false>;
type _NoPublicBm25FactoryBypass = AssertFalse<"createBm25" extends keyof CreateBenchmarkMethodsOptions ? true : false>;
type _NoPublicMinSyncFactoryBypass = AssertFalse<
	"createMinSync" extends keyof CreateBenchmarkMethodsOptions ? true : false
>;
type _ConfigLoaderUsesProcessEnvironment = AssertTrue<
	Parameters<typeof loadBenchmarkConfig>["length"] extends 1 ? true : false
>;

async function importMethodsWithFsInstrumentation(instrumentation: {
	readonly onCreateReadStream?: (path: unknown, options: unknown) => void;
	readonly onReadDirectory?: (path: unknown) => void;
	readonly onReadFile?: (path: unknown) => void;
}): Promise<typeof import("../../benchmark/miracl/methods.ts")> {
	vi.resetModules();
	vi.doMock("node:fs", async () => {
		const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
		return {
			...actual,
			createReadStream: ((...args: unknown[]) => {
				instrumentation.onCreateReadStream?.(args[0], args[1]);
				return Reflect.apply(actual.createReadStream, actual, args);
			}) as typeof actual.createReadStream,
			readFileSync: ((...args: unknown[]) => {
				instrumentation.onReadFile?.(args[0]);
				return Reflect.apply(actual.readFileSync, actual, args);
			}) as typeof actual.readFileSync,
			readdirSync: ((...args: unknown[]) => {
				instrumentation.onReadDirectory?.(args[0]);
				return Reflect.apply(actual.readdirSync, actual, args);
			}) as typeof actual.readdirSync,
		};
	});
	return import("../../benchmark/miracl/methods.ts");
}

function methodStub(name: string, results: readonly RetrievalResult[]): RetrievalMethod {
	return {
		describe: () => ({
			name,
			type: name === "minsync" ? "vector" : "bm25",
			description: "benchmark method stub",
			status: "active",
			capabilities: [],
		}),
		retrieve: async () => [...results],
	};
}

function writeFakeMinSync(root: string): { binaryPath: string; queryStatePath: string; syncStatePath: string } {
	const binaryPath = join(root, "minsync");
	const queryStatePath = join(root, "query-state.json");
	const syncStatePath = join(root, "sync-state.json");
	writeFileSync(queryStatePath, JSON.stringify({ ok: true, stdout: JSON.stringify({ results: [] }) }));
	writeFileSync(syncStatePath, JSON.stringify({ ok: true, stdout: JSON.stringify({ synced: 1 }) }));
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
const { mkdirSync, readFileSync, statSync, utimesSync, writeFileSync } = require("node:fs");
const { join } = require("node:path");
const args = process.argv.slice(2);
if (args[0] === "init") {
  const stateDir = join(process.cwd(), ".minsync");
  const storeDir = join(stateDir, "store");
  mkdirSync(storeDir, { recursive: true });
  writeFileSync(join(stateDir, "config.toml"), [
    "version = 1",
    'source_id = "benchmark-test"',
    "[collection]",
    'name = "benchmark"',
    'path = "store"',
    "[embedder]",
    'id = "test"',
    "[vectorstore]",
    'id = "lancedb"',
    "[vectorstore.options]",
    "dimension = 1024",
    "",
  ].join("\\n"));
  writeFileSync(join(stateDir, "manifest.json"), JSON.stringify({ files: {} }));
  writeFileSync(join(storeDir, "index.lance"), "benchmark-index\\n");
  console.log(JSON.stringify({ ok: true }));
  process.exit(0);
}
if (args[0] === "sync") {
  const state = JSON.parse(readFileSync(${JSON.stringify(syncStatePath)}, "utf8"));
  writeFileSync(join(process.cwd(), ".minsync", "cursor.json"), JSON.stringify({
    source_id: "benchmark-test",
    collection_path: ".minsync/store",
  }));
  if (state.stdout) process.stdout.write(state.stdout);
  if (state.stderr) process.stderr.write(state.stderr);
  process.exit(state.ok ? 0 : 1);
}
if (args[0] === "query") {
  const state = JSON.parse(readFileSync(${JSON.stringify(queryStatePath)}, "utf8"));
  if (state.mutateCollection) {
    const indexPath = join(process.cwd(), ".minsync", "store", "index.lance");
    const originalStats = statSync(indexPath);
    writeFileSync(indexPath, "tampered-indexx\\n");
    utimesSync(indexPath, originalStats.atime, originalStats.mtime);
  }
  if (state.stdout) process.stdout.write(state.stdout);
  if (state.stderr) process.stderr.write(state.stderr);
  process.exit(state.ok ? 0 : 1);
}
process.exit(2);
`,
	);
	chmodSync(binaryPath, 0o755);
	return { binaryPath, queryStatePath, syncStatePath };
}

describe("MIRACL benchmark methods", () => {
	const temporaryRoots: string[] = [];
	const makeRoot = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-methods-"));
		temporaryRoots.push(root);
		return root;
	};

	afterEach(() => {
		for (const root of temporaryRoots.splice(0)) {
			rmSync(root, { recursive: true, force: true });
		}
		vi.restoreAllMocks();
		vi.doUnmock("node:fs");
		vi.resetModules();
	});

	it("requires embedder settings before constructing MinSync", () => {
		expect(() =>
			createBenchmarkMethods({
				names: ["minsync"],
				root: makeRoot(),
				config: undefined,
			}),
		).toThrow("MinSync benchmark requires an embedder configuration");
	});

	it("loads only documented MinSync settings and verifies the key environment reference", () => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		writeFileSync(
			configPath,
			JSON.stringify({
				binaryPath: "/opt/minsync",
				autoInstall: false,
				embedder: {
					id: "model",
					baseUrl: "https://private.example/v1",
					apiKeyEnv: "MIRACL_EMBEDDING_TOKEN",
					dimension: 1024,
					timeoutMs: 30_000,
				},
			}),
		);

		process.env.MIRACL_EMBEDDING_TOKEN = "do-not-serialize-this-secret";
		let config!: ReturnType<typeof loadBenchmarkConfig>;
		try {
			config = loadBenchmarkConfig(configPath);
		} finally {
			delete process.env.MIRACL_EMBEDDING_TOKEN;
		}

		expect(config).toEqual({
			binaryPath: "/opt/minsync",
			autoInstall: false,
			embedder: {
				id: "model",
				baseUrl: "https://private.example/v1",
				apiKeyEnv: "MIRACL_EMBEDDING_TOKEN",
				dimension: 1024,
				timeoutMs: 30_000,
			},
		});
		expect(JSON.stringify(sanitizeMethodConfig(config))).not.toContain("do-not-serialize-this-secret");
	});

	it.each([
		"/tmp/embedder",
		"C:\\models\\embedder",
		"C:/models/embedder",
		"\\\\server\\share\\embedder",
		"file:///tmp/embedder",
	])("rejects filesystem-like embedder ID %s", (embedderId) => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		writeFileSync(configPath, JSON.stringify({ embedder: { id: embedderId, dimension: 1024 } }));

		expect(() => loadBenchmarkConfig(configPath)).toThrow("embedder.id must not be an absolute filesystem path");
		expect(() =>
			sanitizeMethodConfig({
				embedder: { id: embedderId, dimension: 1024 },
			}),
		).toThrow("embedder.id must not be an absolute filesystem path");
	});

	it("allows repository-style model IDs containing a slash", () => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		writeFileSync(
			configPath,
			JSON.stringify({ embedder: { id: "intfloat/multilingual-e5-large", dimension: 1024 } }),
		);

		expect(loadBenchmarkConfig(configPath).embedder.id).toBe("intfloat/multilingual-e5-large");
	});

	it.each([
		[{ unknown: true }, "not a recognized field"],
		[{ authorization: "Bearer private-token" }, "not a recognized field"],
		[{ embedder: { dimension: 0 } }, "positive safe integer"],
		[{ embedder: { dimension: Number.MAX_SAFE_INTEGER + 1 } }, "positive safe integer"],
		[{ embedder: { dimension: 1024, apiKeyEnv: "1INVALID" } }, "valid environment-variable name"],
	])("rejects malformed or extra configuration", (value, expectedMessage) => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		writeFileSync(configPath, JSON.stringify(value));

		expect(() => loadBenchmarkConfig(configPath)).toThrow(expectedMessage);
	});

	it("rejects an absent referenced key without exposing a configured endpoint", () => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		const endpoint = "https://private.example/v1";
		writeFileSync(
			configPath,
			JSON.stringify({
				embedder: {
					baseUrl: endpoint,
					apiKeyEnv: "MISSING_TOKEN",
					dimension: 1024,
				},
			}),
		);

		let message = "";
		delete process.env.MISSING_TOKEN;
		try {
			loadBenchmarkConfig(configPath);
		} catch (error) {
			message = error instanceof Error ? error.message : String(error);
		}
		expect(message).toContain("MISSING_TOKEN");
		expect(message).not.toContain(endpoint);
	});

	it("redacts endpoint and secret values", () => {
		expect(
			sanitizeMethodConfig({
				embedder: {
					id: "model",
					baseUrl: "https://private.example/v1",
					apiKeyEnv: "TOKEN",
					dimension: 1024,
				},
			}),
		).toEqual({
			embedderId: "model",
			endpointKind: "remote",
			apiKeyEnv: "TOKEN",
			dimension: 1024,
		});
	});

	it.each([
		["http://localhost:8080/v1", "local"],
		["http://127.0.0.1:8080/v1", "local"],
		["http://[::1]:8080/v1", "local"],
		["https://embeddings.example/v1", "remote"],
	] as const)("reports only the endpoint kind for %s", (baseUrl, endpointKind) => {
		const sanitized = sanitizeMethodConfig({
			embedder: { id: "model", baseUrl, dimension: 1024 },
		});

		expect(sanitized.endpointKind).toBe(endpointKind);
		expect(JSON.stringify(sanitized)).not.toContain(baseUrl);
	});

	it("uses production diagnostic retrieval and merger for hybrid results", async () => {
		const bm25Stub = methodStub("bm25", [{ id: "bm25:a", source: "/a.md", content: "a", score: 3, metadata: {} }]);
		const minSyncStub = methodStub("minsync", [
			{ id: "minsync:b", source: "/b.md", content: "b", score: 2, metadata: {} },
		]);
		const retrieveSpy = vi.spyOn(ParallelRetriever.prototype, "retrieveWithDiagnostics");
		const mergeSpy = vi.spyOn(ResultMerger.prototype, "merge");

		const hits = await retrieveHybrid([bm25Stub, minSyncStub], "질문", 10);

		expect(hits.map((hit) => hit.source)).toEqual(["/a.md", "/b.md"]);
		expect(retrieveSpy).toHaveBeenCalledWith([bm25Stub, minSyncStub], "질문", { topK: 10 });
		expect(mergeSpy).toHaveBeenCalledWith(expect.any(Map), { topK: 10, dedup: true });
	});

	it("fails hybrid retrieval instead of returning partial output when either method emits a diagnostic", async () => {
		const healthy = methodStub("bm25", [{ id: "bm25:a", source: "/a.md", content: "a", score: 3, metadata: {} }]);
		const failing: RetrievalMethod = {
			...methodStub("minsync", []),
			retrieve: async () => {
				throw new Error("Authorization: Bearer private-token at https://private.example/v1");
			},
		};

		await expect(retrieveHybrid([healthy, failing], "질문", 10)).rejects.toThrow("Hybrid benchmark retrieval failed");
	});

	it("records a hybrid diagnostic as an opaque failed query instead of grading partial hits", async () => {
		const healthy = methodStub("bm25", [
			{ id: "bm25:a", source: "/miracl/a.md", content: "a", score: 3, metadata: {} },
		]);
		const failing: RetrievalMethod = {
			...methodStub("minsync", []),
			retrieve: async () => {
				throw new Error("Authorization: Bearer private-token at https://private.example/v1");
			},
		};
		const hybrid: RetrievalMethod = {
			describe: () => ({
				name: "hybrid",
				type: "hybrid",
				description: "benchmark hybrid",
				status: "active",
				capabilities: [],
			}),
			retrieve: (query, options) => retrieveHybrid([healthy, failing], query, options.topK ?? 100),
		};
		const timestamps = [0, 3];

		const records = await runMethodQueries({
			method: "hybrid",
			retrieval: hybrid,
			queries: [{ queryId: "q1", text: "질문" }],
			documentBySource: new Map([["/miracl/a.md", "a"]]),
			topK: 5,
			now: () => timestamps.shift() as number,
		});

		expect(records).toEqual([
			{
				schemaVersion: 1,
				method: "hybrid",
				queryId: "q1",
				latencyMs: 3,
				hits: [],
				errorCode: "retrieval-failed",
			},
		]);
		expect(JSON.stringify(records)).not.toContain("private-token");
		expect(JSON.stringify(records)).not.toContain("private.example");
	});

	it("records standalone MinSync as failed when its synchronized executable disappears before query", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		rmSync(binaryPath);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("does not grade BM25-only partial output when the synchronized MinSync executable disappears", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["hybrid"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		rmSync(binaryPath);

		const records = await runMethodQueries({
			method: "hybrid",
			retrieval: created.methods.get("hybrid") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("records a non-zero MinSync query process as failed instead of a legitimate empty result", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(
			queryStatePath,
			JSON.stringify({
				ok: false,
				stderr: "Authorization: Bearer private-token at https://private.example/v1",
			}),
		);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
		expect(JSON.stringify(records)).not.toContain("private-token");
		expect(JSON.stringify(records)).not.toContain("private.example");
	});

	it("records malformed successful MinSync output as failed instead of a legitimate empty result", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(queryStatePath, JSON.stringify({ ok: true, stdout: "{not-json" }));

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("keeps a verifiably executed zero-hit MinSync query as a successful empty result", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "no matches" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [] });
		expect(records[0]?.errorCode).toBeUndefined();
	});

	it("maps a verifiably executed MinSync hit through the benchmark source identity", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(
			queryStatePath,
			JSON.stringify({
				ok: true,
				stdout: JSON.stringify({
					results: [{ path: "files/miracl/doc.md.md", score: 0.8, text: "query" }],
				}),
			}),
		);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]?.hits).toEqual([{ documentId: "doc", score: 0.8, rank: 1 }]);
		expect(records[0]?.errorCode).toBeUndefined();
	});

	it("fails an otherwise successful MinSync query when every returned path is outside the path map", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(
			queryStatePath,
			JSON.stringify({
				ok: true,
				stdout: JSON.stringify({
					results: [{ path: "/outside/unknown.md", score: 0.8, text: "query" }],
				}),
			}),
		);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("fails a mixed mapped and unmapped MinSync result instead of grading the mapped subset", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(
			queryStatePath,
			JSON.stringify({
				ok: true,
				stdout: JSON.stringify({
					results: [
						{ path: "files/miracl/doc.md.md", score: 0.8, text: "query" },
						{ path: "/outside/unknown.md", score: 0.7, text: "query" },
					],
				}),
			}),
		);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("detects replacement of the MinSync workspace beneath an unchanged benchmark root", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		const minSyncWorkspace = join(workspace.root, ".autorag", "minsync");
		const displacedWorkspace = join(workspace.root, ".autorag", "minsync-displaced");
		renameSync(minSyncWorkspace, displacedWorkspace);
		mkdirSync(minSyncWorkspace);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("detects replacement of the real MinSync collection index beneath an unchanged workspace", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		const collection = join(workspace.root, ".autorag", "minsync", ".minsync", "store");
		const displacedCollection = join(workspace.root, ".autorag", "minsync", ".minsync", "store-displaced");
		renameSync(collection, displacedCollection);
		mkdirSync(collection);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("invalidates the batch when collection content changes without changing its file identity metadata", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath, queryStatePath } = writeFakeMinSync(parent);
		const created = await createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		writeFileSync(
			queryStatePath,
			JSON.stringify({
				ok: true,
				mutateCollection: true,
				stdout: JSON.stringify({
					results: [{ path: "files/miracl/doc.md.md", score: 0.8, text: "query" }],
				}),
			}),
		);

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]).toMatchObject({ hits: [], errorCode: "retrieval-failed" });
	});

	it("checks the synchronized executable after the injected indexing clock stops", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		const originalExecutable = readFileSync(binaryPath, "utf8");
		let clockCalls = 0;

		await expect(
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
				now: () => {
					clockCalls += 1;
					if (clockCalls === 2) {
						writeFileSync(binaryPath, originalExecutable.replace("process.exit(2);", "process.exit(3);"));
					}
					return clockCalls === 1 ? 10 : 15;
				},
			}),
		).rejects.toThrow("MinSync benchmark executable changed");
		expect(clockCalls).toBe(2);
	});

	it("keeps recursive collection metadata walks outside the measured retrieval interval", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		let collectionWalks = 0;
		const instrumentedMethods = await importMethodsWithFsInstrumentation({
			onReadDirectory: (path) => {
				if (String(path).includes(`${join(".minsync", "store")}`)) {
					collectionWalks += 1;
				}
			},
		});
		const created = await instrumentedMethods.createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		const clockWalkCounts: number[] = [];

		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
			now: () => {
				clockWalkCounts.push(collectionWalks);
				return clockWalkCounts.length === 1 ? 10 : 15;
			},
		});

		expect(records[0]?.errorCode).toBeUndefined();
		expect(clockWalkCounts).toHaveLength(2);
		expect(clockWalkCounts[0]).toBeGreaterThan(0);
		expect(clockWalkCounts[1]).toBe(clockWalkCounts[0]);
		expect(collectionWalks).toBeGreaterThan(clockWalkCounts[1] as number);
	});

	it("streams collection files without whole-file synchronous reads", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		let wholeCollectionReads = 0;
		const streamBufferSizes: number[] = [];
		const instrumentedMethods = await importMethodsWithFsInstrumentation({
			onCreateReadStream: (path, options) => {
				if (!String(path).includes(`${join(".minsync", "store")}${process.platform === "win32" ? "\\" : "/"}`)) {
					return;
				}
				const highWaterMark = (options as { highWaterMark?: unknown } | undefined)?.highWaterMark;
				if (typeof highWaterMark === "number") streamBufferSizes.push(highWaterMark);
			},
			onReadFile: (path) => {
				if (String(path).includes(`${join(".minsync", "store")}${process.platform === "win32" ? "\\" : "/"}`)) {
					wholeCollectionReads += 1;
				}
			},
		});

		const created = await instrumentedMethods.createBenchmarkMethods({
			names: ["minsync"],
			root: workspace.root,
			documentBySource: workspace.documentBySource,
			config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
		});
		const records = await runMethodQueries({
			method: "minsync",
			retrieval: created.methods.get("minsync") as RetrievalMethod,
			queries: [{ queryId: "q1", text: "query" }],
			documentBySource: workspace.documentBySource,
			topK: 5,
		});

		expect(records[0]?.errorCode).toBeUndefined();
		expect(wholeCollectionReads).toBe(0);
		expect(streamBufferSizes.length).toBeGreaterThan(0);
		expect(streamBufferSizes.every((size) => size <= 64 * 1024)).toBe(true);
	});

	it("does not fall back to PATH when an explicit MinSync binary path is missing", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const fallbackDirectory = join(parent, "fallback-bin");
		mkdirSync(fallbackDirectory);
		writeFakeMinSync(fallbackDirectory);
		const previousPath = process.env.PATH;
		process.env.PATH = `${fallbackDirectory}:${previousPath ?? ""}`;
		try {
			await expect(
				createBenchmarkMethods({
					names: ["minsync"],
					root: workspace.root,
					documentBySource: workspace.documentBySource,
					config: {
						binaryPath: join(parent, "missing-explicit-minsync"),
						autoInstall: false,
						embedder: { dimension: 1024 },
					},
				}),
			).rejects.toThrow("MinSync benchmark executable is unavailable");
		} finally {
			process.env.PATH = previousPath;
		}
	});

	it("rejects an explicit MinSync binary path that is not executable", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "query" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);
		chmodSync(binaryPath, 0o644);

		await expect(
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
			}),
		).rejects.toThrow("MinSync benchmark executable is unavailable");
	});

	it("stabilizes equal-score hybrid truncation regardless of retrieval completion order", async () => {
		const runWithOrder = async (first: "bm25" | "minsync") => {
			let releaseBm25!: () => void;
			let releaseMinSync!: () => void;
			const bm25Ready = new Promise<void>((resolve) => {
				releaseBm25 = resolve;
			});
			const minSyncReady = new Promise<void>((resolve) => {
				releaseMinSync = resolve;
			});
			const bm25: RetrievalMethod = {
				...methodStub("bm25", []),
				retrieve: async () => {
					await bm25Ready;
					return [{ id: "bm25:a", source: "/a.md", content: "a", score: 1, metadata: {} }];
				},
			};
			const minsync: RetrievalMethod = {
				...methodStub("minsync", []),
				retrieve: async () => {
					await minSyncReady;
					return [{ id: "minsync:b", source: "/b.md", content: "b", score: 1, metadata: {} }];
				},
			};
			const pending = retrieveHybrid([bm25, minsync], "질문", 1);
			if (first === "bm25") {
				releaseBm25();
				await Promise.resolve();
				releaseMinSync();
			} else {
				releaseMinSync();
				await Promise.resolve();
				releaseBm25();
			}
			return pending;
		};

		const bm25First = await runWithOrder("bm25");
		const minSyncFirst = await runWithOrder("minsync");

		expect(bm25First.map((hit) => hit.source)).toEqual(["/a.md"]);
		expect(minSyncFirst).toEqual(bm25First);
	});

	it("rejects a degraded MinSync synchronization before returning query methods", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const { binaryPath, syncStatePath } = writeFakeMinSync(parent);
		writeFileSync(
			syncStatePath,
			JSON.stringify({
				ok: false,
				stderr: "Authorization: Bearer private-token at https://private.example/v1",
			}),
		);

		await expect(
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
			}),
		).rejects.toThrow("MinSync benchmark indexing failed");
	});

	it("rejects a zero-document MinSync synchronization for a non-empty benchmark mirror", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const { binaryPath, syncStatePath } = writeFakeMinSync(parent);
		writeFileSync(syncStatePath, JSON.stringify({ ok: true, stdout: JSON.stringify({ synced: 0 }) }));

		await expect(
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
			}),
		).rejects.toThrow("MinSync benchmark indexing failed");
	});

	it("rejects an invalid source mapping before constructing MinSync", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const { binaryPath } = writeFakeMinSync(parent);

		expect(() =>
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: new Map([["/miracl/doc.md", "wrong"]]),
				config: { binaryPath, autoInstall: false, embedder: { dimension: 1024 } },
			}),
		).toThrow("bijective");
		expect(existsSync(join(workspace.root, ".autorag", "minsync"))).toBe(false);
	});

	it("constructs a real MinSync method by default and fails when no binary is available", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const previousPath = process.env.PATH;
		process.env.PATH = join(parent, "empty-path");
		mkdirSync(process.env.PATH);
		try {
			await expect(
				createBenchmarkMethods({
					names: ["minsync"],
					root: workspace.root,
					documentBySource: workspace.documentBySource,
					config: {
						binaryPath: join(parent, "missing-minsync"),
						autoInstall: false,
						embedder: { dimension: 1024 },
					},
				}),
			).rejects.toThrow("MinSync benchmark executable is unavailable");
		} finally {
			process.env.PATH = previousPath;
		}
	});
});
