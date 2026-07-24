import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import {
	createBenchmarkMethods,
	loadBenchmarkConfig,
	retrieveHybrid,
	sanitizeMethodConfig,
} from "../../benchmark/miracl/methods.ts";
import { runMethodQueries } from "../../benchmark/miracl/run.ts";
import { materializeBenchmarkWorkspace } from "../../benchmark/miracl/workspace.ts";
import type { MinSyncSyncResult } from "../../src/minsync/types.ts";
import { ParallelRetriever, ResultMerger } from "../../src/retrieval/merger.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";

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

		const config = loadBenchmarkConfig(configPath, {
			MIRACL_EMBEDDING_TOKEN: "do-not-serialize-this-secret",
		});

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
		[{ unknown: true }, "not a recognized field"],
		[{ authorization: "Bearer private-token" }, "not a recognized field"],
		[{ embedder: { dimension: 0 } }, "positive safe integer"],
		[{ embedder: { dimension: Number.MAX_SAFE_INTEGER + 1 } }, "positive safe integer"],
		[{ embedder: { dimension: 1024, apiKeyEnv: "1INVALID" } }, "valid environment-variable name"],
	])("rejects malformed or extra configuration", (value, expectedMessage) => {
		const root = makeRoot();
		const configPath = join(root, "minsync.json");
		writeFileSync(configPath, JSON.stringify(value));

		expect(() => loadBenchmarkConfig(configPath, {})).toThrow(expectedMessage);
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
		try {
			loadBenchmarkConfig(configPath, {});
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

	it("rejects a degraded MinSync synchronization before returning query methods", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const retrieve = vi.fn(async () => []);
		const sync = vi.fn(
			async (): Promise<MinSyncSyncResult> => ({
				ok: false,
				synced: 0,
				workspacePath: join(workspace.root, ".autorag", "minsync"),
				reason: "Authorization: Bearer private-token at https://private.example/v1",
			}),
		);

		await expect(
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: workspace.documentBySource,
				config: { embedder: { dimension: 1024 } },
				createMinSync: () => ({
					...methodStub("minsync", []),
					sync,
					retrieve,
				}),
			}),
		).rejects.toThrow("MinSync benchmark indexing failed");
		expect(sync).toHaveBeenCalledTimes(1);
		expect(retrieve).not.toHaveBeenCalled();
	});

	it("rejects an invalid source mapping before constructing MinSync", async () => {
		const parent = makeRoot();
		const workspace = materializeBenchmarkWorkspace(join(parent, "workspace"), [
			{ documentId: "doc", title: "제목", text: "본문" },
		]);
		const createMinSync = vi.fn();

		expect(() =>
			createBenchmarkMethods({
				names: ["minsync"],
				root: workspace.root,
				documentBySource: new Map([["/miracl/doc.md", "wrong"]]),
				config: { embedder: { dimension: 1024 } },
				createMinSync,
			}),
		).toThrow("bijective");
		expect(createMinSync).not.toHaveBeenCalled();
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
			).rejects.toThrow("MinSync benchmark indexing failed");
		} finally {
			process.env.PATH = previousPath;
		}
	});
});
