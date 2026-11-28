import { chmodSync, existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { normalizeIndexingConfig } from "../../src/cli/config.ts";
import { createEmbedderFromConfig } from "../../src/embedding/index.ts";
import { buildLtrDataset, LTR_FEATURE_NAMES } from "../../src/memory/ltr-export.ts";
import { RANKING_LOG_SCHEMA_VERSION, type RankingRunRecord } from "../../src/memory/ranking-log.ts";
import {
	AUTORAG_HARDWARE_OVERRIDE_ENV,
	configuredBm25Config,
	configuredSparseId,
	DEFAULT_MINSYNC_EMBEDDER_ID,
	DEFAULT_MINSYNC_SPARSE_ID,
	evaluateMinSyncHardwareTier,
	isBm25OnlyConfig,
	MinSyncClient,
	MinSyncHybridMethod,
	MinSyncVectorMethod,
	NONE_SPARSE_ID,
	parseNvidiaSmiOutput,
	probeMinSyncHardware,
	resolveMinSyncRetrieverPlan,
	rewriteEmbedderConfig,
	SPARSE_ONLY_EMBEDDER_ID,
} from "../../src/minsync/index.ts";

const GIB = 1024 ** 3;

describe("MinSync hardware-aware tiering and BM25 support", () => {
	let workDir: string;
	const savedOverride = process.env[AUTORAG_HARDWARE_OVERRIDE_ENV];

	beforeEach(() => {
		workDir = mkdtempSync(join(tmpdir(), "autorag-bm25-hw-"));
		delete process.env[AUTORAG_HARDWARE_OVERRIDE_ENV];
	});

	afterEach(() => {
		if (savedOverride === undefined) delete process.env[AUTORAG_HARDWARE_OVERRIDE_ENV];
		else process.env[AUTORAG_HARDWARE_OVERRIDE_ENV] = savedOverride;
		rmSync(workDir, { recursive: true, force: true });
	});

	it("maps Apple Silicon, Windows, and Linux hardware to VM-calibrated indexing tiers", () => {
		// Low-RAM machines (< 16 GiB) -> bm25-only (zero ML)
		expect(
			evaluateMinSyncHardwareTier({
				platform: "darwin",
				arch: "arm64",
				cores: 8,
				totalRamBytes: 8 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("bm25-only");
		expect(
			evaluateMinSyncHardwareTier({
				platform: "win32",
				arch: "x64",
				cores: 8,
				totalRamBytes: 8 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("bm25-only");
		// Low-core CPU-only 16 GiB -> bm25-only
		expect(
			evaluateMinSyncHardwareTier({
				platform: "win32",
				arch: "x64",
				cores: 2,
				totalRamBytes: 16 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("bm25-only");

		// Mid-spec (16 GiB Apple Silicon or 16+ GiB 4-core CPU-only) -> sparse+bm25
		expect(
			evaluateMinSyncHardwareTier({
				platform: "darwin",
				arch: "arm64",
				cores: 8,
				totalRamBytes: 16 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("sparse+bm25");
		expect(
			evaluateMinSyncHardwareTier({
				platform: "win32",
				arch: "x64",
				cores: 4,
				totalRamBytes: 16 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("sparse+bm25");
		expect(
			evaluateMinSyncHardwareTier({
				platform: "linux",
				arch: "x64",
				cores: 8,
				totalRamBytes: 32 * GIB,
				cudaGpus: [{ name: "RTX 3050", vramMib: 6144 }],
			}).tier,
		).toBe("sparse+bm25");

		// High-spec (Apple Silicon >= 24 GiB OR CUDA >= 8 GB VRAM + >= 16 GiB RAM) -> full
		expect(
			evaluateMinSyncHardwareTier({
				platform: "darwin",
				arch: "arm64",
				cores: 8,
				totalRamBytes: 24 * GIB,
				cudaGpus: [],
			}).tier,
		).toBe("full");
		expect(
			evaluateMinSyncHardwareTier({
				platform: "win32",
				arch: "x64",
				cores: 8,
				totalRamBytes: 16 * GIB,
				cudaGpus: [{ name: "NVIDIA GeForce RTX 4060", vramMib: 8188 }],
			}).tier,
		).toBe("full");
	});

	it("parses nvidia-smi output and resolves auto retriever plans while honoring explicit overrides", () => {
		expect(parseNvidiaSmiOutput("NVIDIA GeForce RTX 4060, 8188\nNVIDIA TITAN Xp, 12288\n")).toEqual([
			{ name: "NVIDIA GeForce RTX 4060", vramMib: 8188 },
			{ name: "NVIDIA TITAN Xp", vramMib: 12288 },
		]);

		const lowHw = { platform: "win32", arch: "x64", cores: 4, totalRamBytes: 8 * GIB, cudaGpus: [] };
		const autoLow = resolveMinSyncRetrieverPlan({
			embedder: { id: "auto" },
			sparse: { id: "auto" },
			hardware: lowHw,
		});
		expect(autoLow.tier).toBe("bm25-only");
		expect(autoLow.embedder?.id).toBe(SPARSE_ONLY_EMBEDDER_ID);
		expect(autoLow.sparse?.id).toBe(NONE_SPARSE_ID);
		expect(isBm25OnlyConfig(autoLow.embedder, autoLow.sparse)).toBe(true);

		// Explicit remote embedder overrides auto even on 8 GB hardware, while sparse stays auto -> none
		const explicitRemote = resolveMinSyncRetrieverPlan({
			embedder: { id: "openai:text-embedding-3-small", apiKeyEnv: "OPENAI_API_KEY" },
			sparse: { id: "auto" },
			hardware: lowHw,
		});
		expect(explicitRemote.embedder?.id).toBe("openai:text-embedding-3-small");
		expect(explicitRemote.sparse?.id).toBe(NONE_SPARSE_ID);

		const highHw = { platform: "darwin", arch: "arm64", cores: 10, totalRamBytes: 32 * GIB, cudaGpus: [] };
		const autoHigh = resolveMinSyncRetrieverPlan({
			embedder: { id: "auto" },
			sparse: { id: "auto" },
			hardware: highHw,
		});
		expect(autoHigh.tier).toBe("full");
		expect(autoHigh.embedder?.id).toBe(DEFAULT_MINSYNC_EMBEDDER_ID);
		expect(autoHigh.sparse?.id).toBe(DEFAULT_MINSYNC_SPARSE_ID);
	});

	it("honors AUTORAG_HARDWARE_OVERRIDE in probeMinSyncHardware, createEmbedderFromConfig, and AutoRAGAgent", async () => {
		process.env[AUTORAG_HARDWARE_OVERRIDE_ENV] = JSON.stringify({
			platform: "win32",
			arch: "x64",
			cores: 4,
			totalRamBytes: 8 * GIB,
			cudaGpus: [],
		});
		expect(probeMinSyncHardware().totalRamBytes).toBe(8 * GIB);

		const embedder = createEmbedderFromConfig({ embedder: { id: "auto" } });
		expect((await embedder.health()).backend).toBe("none");

		const method = new MinSyncVectorMethod({
			root: workDir,
			autoInstall: false,
			embedder: { id: "auto" },
			sparse: { id: "auto" },
		});
		expect(method.describe().type).toBe("bm25");

		const hybrid = new MinSyncHybridMethod({
			root: workDir,
			autoInstall: false,
			embedder: { id: "auto" },
			sparse: { id: "auto" },
		});
		expect(hybrid.describe().type).toBe("bm25");

		const agent = new AutoRAGAgent({
			searchPaths: [workDir],
			workspacePath: workDir,
			memoryPath: join(workDir, "memory.json"),
			minSync: { autoInstall: false, embedder: { id: "auto" }, sparse: { id: "auto" } },
			jikji: false,
		});
		const registered = agent
			.getMethodRegistry()
			.list()
			.find((m) => m.describe().name === "hybrid");
		expect(registered?.describe().type).toBe("bm25");
	});

	it("validates minSync.sparse and minSync.bm25 in config normalization and writes them to config.toml", () => {
		const normalized = normalizeIndexingConfig({
			minSync: {
				enabled: true,
				embedder: { id: "none" },
				sparse: { id: "none" },
				bm25: { enabled: true, language: "ko" },
			},
		});
		expect(normalized.minSync.sparse?.id).toBe("none");
		expect(normalized.minSync.bm25).toEqual({ enabled: true, language: "ko" });

		expect(() =>
			normalizeIndexingConfig({
				minSync: { bm25: { language: "klingon" } },
			}),
		).toThrow(/minSync\.bm25\.language/);

		// rewriteEmbedderConfig persists [sparse] and [bm25] into .minsync/config.toml
		const ws = join(workDir, "ws");
		const cfgDir = join(ws, ".minsync");
		writeFileSync(
			join(workDir, "seed.toml"),
			'[embedder]\nid = "native:Qwen/Qwen3-Embedding-0.6B"\n[vectorstore]\nid = "lancedb"\n[vectorstore.options]\ndimension = 1024\n',
		);
		const { mkdirSync } = require("node:fs") as typeof import("node:fs");
		mkdirSync(cfgDir, { recursive: true });
		writeFileSync(join(cfgDir, "config.toml"), readFileSync(join(workDir, "seed.toml"), "utf8"));

		expect(
			rewriteEmbedderConfig(
				ws,
				{ id: "none", dimension: 1 },
				{ sparse: { id: "none" }, bm25: { enabled: true, language: "ko" } },
			),
		).toBe(true);
		expect(configuredSparseId(ws)).toBe("none");
		expect(configuredBm25Config(ws)).toEqual({ enabled: true, language: "ko" });
	});

	it("passes --vector/--sparse/--bm25 flags and parses per-retriever LTR scores in MinSyncClient", async () => {
		const fakeBin = join(workDir, "fake-minsync");
		const argsLog = join(workDir, "args.log");
		writeFileSync(
			fakeBin,
			`#!/bin/sh
echo "$@" >> "${argsLog}"
echo '[{"doc_id":"c1","path":"files/ko.md","heading_path":"ko.md","score":0.049,"text":"환불 정책","vector_rank":1,"sparse_rank":2,"bm25_rank":1,"vector_score":0.82,"sparse_score":95.4,"bm25_score":7.21}]'
`,
		);
		chmodSync(fakeBin, 0o755);

		const client = new MinSyncClient({ binaryPath: fakeBin, workspacePath: workDir });
		const hits = await client.queryWithRetrievers("환불 정책", 5, { vector: true, sparse: true, bm25: true });
		expect(hits).toHaveLength(1);
		expect(hits[0]).toMatchObject({
			docId: "c1",
			path: "files/ko.md",
			vectorRank: 1,
			sparseRank: 2,
			bm25Rank: 1,
			vectorScore: 0.82,
			sparseScore: 95.4,
			bm25Score: 7.21,
		});
		const logged = readFileSync(argsLog, "utf8");
		expect(logged).toContain("--vector");
		expect(logged).toContain("--sparse");
		expect(logged).toContain("--bm25");
	});

	it("exports dedicated bm25_* LTR features from minsync-bm25 observations", () => {
		const now = Date.now();
		const run: RankingRunRecord = {
			schemaVersion: RANKING_LOG_SCHEMA_VERSION,
			runId: "run-bm25-1",
			conversationId: "conv-1",
			question: "환불 정책",
			startedAt: now - 1000,
			finishedAt: now,
			searchQueries: ["환불 정책"],
			datasources: [],
			web: false,
			judgments: { citationRelevance: "judged", support: "judged" },
			candidates: [
				{
					chunkId: "minsync:/docs/a.md:1",
					source: "/docs/a.md",
					origin: "local",
					contentChars: 120,
					observations: [
						{
							stage: "prefetch",
							surface: "minsync",
							method: "minsync-bm25",
							query: "환불 정책",
							rank: 1,
							score: 7.5,
						},
						{
							stage: "prefetch",
							surface: "minsync",
							method: "minsync",
							query: "환불 정책",
							rank: 2,
							score: 0.71,
						},
					],
					exposures: [{ surface: "baseline", query: "환불 정책", position: 1 }],
					text: { bm25: 3.1, questionCoverage: 1.0 },
					memory: { priorSupport: 0, similarQuestionSupport: 0 },
					labels: { citedFinal: true, modelCited: true, support: 0.9 },
				},
				{
					chunkId: "minsync:/docs/b.md:1",
					source: "/docs/b.md",
					origin: "local",
					contentChars: 80,
					observations: [
						{
							stage: "prefetch",
							surface: "minsync",
							method: "minsync",
							query: "환불 정책",
							rank: 1,
							score: 0.75,
						},
					],
					exposures: [{ surface: "baseline", query: "환불 정책", position: 2 }],
					text: { bm25: 0.0, questionCoverage: 0.0 },
					memory: { priorSupport: 0, similarQuestionSupport: 0 },
					labels: { citedFinal: false, modelCited: false },
				},
			],
		};
		const dataset = buildLtrDataset([run], { validFraction: 0 });
		expect(dataset.train.rows).toHaveLength(2);
		const aRow = dataset.train.rows.find((r) => r.source === "/docs/a.md");
		const bRow = dataset.train.rows.find((r) => r.source === "/docs/b.md");
		expect(aRow?.features[LTR_FEATURE_NAMES.indexOf("bm25_best_rank")]).toBe(1);
		expect(aRow?.features[LTR_FEATURE_NAMES.indexOf("bm25_max_score")]).toBe(7.5);
		expect(aRow?.features[LTR_FEATURE_NAMES.indexOf("bm25_hits")]).toBe(1);
		expect(Number.isNaN(bRow?.features[LTR_FEATURE_NAMES.indexOf("bm25_best_rank")] ?? 0)).toBe(true);
		expect(bRow?.features[LTR_FEATURE_NAMES.indexOf("bm25_hits")]).toBe(0);
	});

	it("runs live E2E BM25-only sync and multilingual search when local BM25-enabled minsync binary is present", async () => {
		const localBin = process.env.AUTORAG_TEST_MINSYNC_BIN;
		if (!localBin || !existsSync(localBin)) return;

		const ws = join(workDir, "live-ws");
		const { mkdirSync } = require("node:fs") as typeof import("node:fs");
		mkdirSync(join(ws, "files"), { recursive: true });
		writeFileSync(join(ws, "files", "ko.md"), "환불정책과배송안내입니다 빠른 배송 안내\n");
		writeFileSync(join(ws, "files", "ja.md"), "関西国際空港限定トートバッグの販売情報\n");
		writeFileSync(join(ws, "files", "en.md"), "Handle ERR_503_BACKPRESSURE in retry-queue.v2\n");

		process.env[AUTORAG_HARDWARE_OVERRIDE_ENV] = JSON.stringify({
			platform: "win32",
			arch: "x64",
			cores: 4,
			totalRamBytes: 8 * GIB,
			cudaGpus: [],
		});
		const client = new MinSyncClient({
			binaryPath: localBin,
			workspacePath: ws,
			embedder: { id: "auto" },
			sparse: { id: "auto" },
			bm25: { enabled: true, language: "multilingual" },
		});
		const sync = await client.sync(true);
		expect(sync.ok).toBe(true);

		const koHits = await client.query("환불 정책", 3, "bm25");
		expect(koHits[0]?.path).toContain("ko.md");
		expect(koHits[0]?.bm25Rank).toBe(1);
		expect((koHits[0]?.bm25Score ?? 0) > 0).toBe(true);

		const jaHits = await client.queryWithRetrievers("関西国際空港", 3, { bm25: true });
		expect(jaHits[0]?.path).toContain("ja.md");
	});
});
