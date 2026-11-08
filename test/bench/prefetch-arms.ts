import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { parseArgs } from "node:util";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { unionChannelResults } from "../../src/minsync/channel-union.ts";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import { createReranker, type Reranker } from "../../src/retrieval/rerank.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { type BootstrapDiff, mrrAtK, ndcgAtK, pairedBootstrapDiff, recallAtK } from "./ir-metrics.ts";

/**
 * Retrieval-arm benchmark for the fast-answer baseline pool.
 *
 * Question: with a reranker deciding the final order, which MinSync candidate pool should feed it:
 * dense only (what the baseline did before), SPLADE only, MinSync's own RRF `hybrid`, or the union of
 * the dense and SPLADE channels? Every arm runs on the same indexed corpus through the same
 * `MinSyncVectorMethod` code the agent uses, and reranked arms use the product's `createReranker`.
 *
 *   bun test/bench/prefetch-arms.ts --data DIR --minsync BIN --rerank-model ID [--sample N]
 *
 * DIR holds `corpus.jsonl` ({id,text}), `queries.jsonl` ({id,text}) and `qrels.jsonl` ({query,doc,gain}).
 * The corpus text is sent to the rerank provider, so use public data only. The run is isolated: its
 * AutoRAG home, workspace and index live under a fresh temp root that is removed afterwards.
 */

interface Row {
	readonly id: string;
	readonly text: string;
}

function readJsonl<T>(path: string): T[] {
	return readFileSync(path, "utf8")
		.split("\n")
		.filter((line) => line.length > 0)
		.map((line) => JSON.parse(line) as T);
}

const { values } = parseArgs({
	options: {
		data: { type: "string" },
		minsync: { type: "string" },
		"rerank-model": { type: "string" },
		"rerank-provider": { type: "string", default: "openrouter" },
		depth: { type: "string", default: "100" },
		sample: { type: "string" },
		cache: { type: "string" },
		out: { type: "string" },
		keep: { type: "boolean", default: false },
		workspace: { type: "string" },
	},
});
const dataDir = values.data;
const minsyncBinary = values.minsync;
const rerankModel = values["rerank-model"];
if (dataDir === undefined || minsyncBinary === undefined || rerankModel === undefined) {
	throw new Error("--data, --minsync and --rerank-model are required");
}
const depth = Number(values.depth);

// Isolation first: nothing below may read or write the maintainer's ~/.autorag.
const root = values.workspace ?? mkdtempSync(join(tmpdir(), "autorag-arms-"));
mkdirSync(root, { recursive: true });
process.env.AUTORAG_HOME = join(root, ".autorag-home");
process.env.AUTORAG_CONFIG = join(process.env.AUTORAG_HOME, "config.json");
mkdirSync(process.env.AUTORAG_HOME, { recursive: true });

const corpus = readJsonl<Row>(join(dataDir, "corpus.jsonl"));
let queries = readJsonl<Row>(join(dataDir, "queries.jsonl"));
const qrelRows = readJsonl<{ query: string; doc: string; gain: number }>(join(dataDir, "qrels.jsonl"));
const qrelsByQuery = new Map<string, Map<string, number>>();
for (const { query, doc, gain } of qrelRows) {
	const perQuery = qrelsByQuery.get(query) ?? new Map<string, number>();
	perQuery.set(doc, gain);
	qrelsByQuery.set(query, perQuery);
}
queries = queries.filter((query) => qrelsByQuery.has(query.id));
if (values.sample !== undefined) {
	// Deterministic subsample: stable order by hash of the query id.
	const hashOf = (id: string): string => createHash("sha1").update(id).digest("hex");
	queries = [...queries].sort((a, b) => hashOf(a.id).localeCompare(hashOf(b.id))).slice(0, Number(values.sample));
}

const docs = join(root, "docs");
mkdirSync(docs, { recursive: true });
const docIdByFile = new Map<string, string>();
for (const [index, row] of corpus.entries()) {
	const file = `doc-${String(index).padStart(6, "0")}.txt`;
	if (!existsSync(join(docs, file))) writeFileSync(join(docs, file), row.text);
	docIdByFile.set(file, row.id);
}

const builtReranker = createReranker({ provider: values["rerank-provider"], model: rerankModel });
if (builtReranker === undefined || !builtReranker.describe().available) {
	throw new Error(`reranker unavailable: ${builtReranker?.describe().reason ?? "not built"}`);
}
const reranker: Reranker = builtReranker;

const cachePath = values.cache ?? join(root, "rerank-cache.json");
const rerankCache: Record<string, string[]> = existsSync(cachePath)
	? (JSON.parse(readFileSync(cachePath, "utf8")) as Record<string, string[]>)
	: {};

interface Timing {
	calls: number;
	ms: number;
}
const timings = new Map<string, Timing>();
async function timed<T>(label: string, work: () => Promise<T>): Promise<T> {
	const started = performance.now();
	try {
		return await work();
	} finally {
		const entry = timings.get(label) ?? { calls: 0, ms: 0 };
		entry.calls += 1;
		entry.ms += performance.now() - started;
		timings.set(label, entry);
	}
}

function docIdOf(result: RetrievalResult): string {
	const id = docIdByFile.get(basename(result.source));
	if (id === undefined) throw new Error(`result source is not a corpus file: ${result.source}`);
	return id;
}

function docRanking(results: readonly RetrievalResult[]): string[] {
	const seen = new Set<string>();
	const ranking: string[] = [];
	for (const result of results) {
		const id = docIdOf(result);
		if (seen.has(id)) continue;
		seen.add(id);
		ranking.push(id);
	}
	return ranking;
}

async function limited<T, R>(items: readonly T[], width: number, work: (item: T) => Promise<R>): Promise<R[]> {
	const out: R[] = new Array(items.length);
	let next = 0;
	await Promise.all(
		Array.from({ length: width }, async () => {
			while (next < items.length) {
				const index = next++;
				out[index] = await work(items[index] as T);
			}
		}),
	);
	return out;
}

async function main(): Promise<void> {
	console.error(`corpus=${corpus.length} queries=${queries.length} depth=${depth} root=${root}`);
	const agent = new AutoRAGAgent({
		searchPaths: [docs],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { binaryPath: minsyncBinary, autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
	});
	const refreshStarted = performance.now();
	// A reused workspace syncs incrementally; a fresh one builds the whole index.
	await agent.refresh(values.workspace === undefined);
	console.error(`refresh ${(performance.now() - refreshStarted) / 1000}s`);

	const channel = (mode: "vector" | "sparse" | "hybrid"): MinSyncVectorMethod =>
		new MinSyncVectorMethod({ root, binaryPath: minsyncBinary, autoInstall: false, mode });
	const methods = { vector: channel("vector"), sparse: channel("sparse"), hybrid: channel("hybrid") };

	interface Pools {
		vector: RetrievalResult[];
		sparse: RetrievalResult[];
		hybrid: RetrievalResult[];
		hybridHalf: RetrievalResult[];
		union: RetrievalResult[];
		unionCapped: RetrievalResult[];
	}
	const pools = new Map<string, Pools>();
	await limited(queries, 4, async (query) => {
		const run = (mode: "vector" | "sparse" | "hybrid", topK: number): Promise<RetrievalResult[]> =>
			timed(`first-stage:${mode}@${topK}`, () => methods[mode].retrieve(query.text, { topK }));
		const [vector, sparse, hybrid, hybridHalf] = await Promise.all([
			run("vector", depth),
			run("sparse", depth),
			run("hybrid", depth),
			run("hybrid", Math.floor(depth / 2)),
		]);
		const channels = [
			{ channel: "vector", results: vector },
			{ channel: "sparse", results: sparse },
		];
		pools.set(query.id, {
			vector,
			sparse,
			hybrid,
			hybridHalf,
			union: unionChannelResults(channels),
			unionCapped: unionChannelResults(channels, { limit: depth }),
		});
	});

	const poolKey = (queryText: string, pool: readonly RetrievalResult[]): string =>
		createHash("sha1")
			.update(`${rerankModel}\0${queryText}\0${pool.map((result) => `${result.source}#${result.id}`).join("|")}`)
			.digest("hex");

	async function rerankedRanking(query: Row, pool: readonly RetrievalResult[]): Promise<string[]> {
		const key = poolKey(query.text, pool);
		const cached = rerankCache[key];
		if (cached !== undefined) return cached;
		const ranked = await timed("rerank", () => reranker.rerank(query.text, pool));
		const ranking = docRanking(ranked);
		rerankCache[key] = ranking;
		return ranking;
	}

	type ArmName = string;
	const arms: { name: ArmName; pool: (p: Pools) => RetrievalResult[]; rerank: boolean }[] = [
		{ name: "dense", pool: (p) => p.vector, rerank: false },
		{ name: "sparse", pool: (p) => p.sparse, rerank: false },
		{ name: "hybrid (RRF, MinSync)", pool: (p) => p.hybrid, rerank: false },
		{ name: "union (interleaved, no rerank)", pool: (p) => p.union, rerank: false },
		{ name: "dense -> rerank", pool: (p) => p.vector, rerank: true },
		{ name: "sparse -> rerank", pool: (p) => p.sparse, rerank: true },
		{ name: "hybrid@N -> rerank", pool: (p) => p.hybrid, rerank: true },
		{ name: "hybrid@N/2 -> rerank", pool: (p) => p.hybridHalf, rerank: true },
		{ name: "union capped at N -> rerank", pool: (p) => p.unionCapped, rerank: true },
		{ name: "union (dense N + sparse N) -> rerank", pool: (p) => p.union, rerank: true },
	];

	interface ArmScores {
		ndcg: number[];
		recall10: number[];
		mrr: number[];
		poolRecall: number[];
		poolSize: number[];
	}
	const perArm = new Map<ArmName, ArmScores>();
	for (const arm of arms) perArm.set(arm.name, { ndcg: [], recall10: [], mrr: [], poolRecall: [], poolSize: [] });
	const scoresOf = (name: ArmName): ArmScores => {
		const scores = perArm.get(name);
		if (scores === undefined) throw new Error(`unknown arm: ${name}`);
		return scores;
	};

	for (const arm of arms) {
		const scores = scoresOf(arm.name);
		const rankings = await limited(queries, arm.rerank ? 8 : 16, async (query) => {
			const pool = arm.pool(pools.get(query.id) as Pools);
			return { pool, ranking: arm.rerank ? await rerankedRanking(query, pool) : docRanking(pool) };
		});
		for (const [index, query] of queries.entries()) {
			const { pool, ranking } = rankings[index] as { pool: RetrievalResult[]; ranking: string[] };
			const qrels = qrelsByQuery.get(query.id) as Map<string, number>;
			scores.ndcg.push(ndcgAtK(ranking, qrels, 10));
			scores.recall10.push(recallAtK(ranking, qrels, 10));
			scores.mrr.push(mrrAtK(ranking, qrels, 10));
			scores.poolRecall.push(recallAtK(docRanking(pool), qrels, Number.MAX_SAFE_INTEGER));
			scores.poolSize.push(pool.length);
		}
		console.error(`arm done: ${arm.name}`);
		writeFileSync(cachePath, JSON.stringify(rerankCache));
	}

	const mean = (values: readonly number[]): number =>
		values.reduce((sum, v) => sum + v, 0) / Math.max(values.length, 1);
	const fmt = (value: number): string => value.toFixed(4);
	const baseline = scoresOf("dense -> rerank");
	const hybridBaseline = scoresOf("hybrid@N -> rerank");
	const lines = [
		`| arm | nDCG@10 | Recall@10 | MRR@10 | pool recall | mean pool | Δ nDCG vs dense->rerank [95% CI] | Δ nDCG vs hybrid@N->rerank [95% CI] |`,
		`|---|---|---|---|---|---|---|---|`,
	];
	const summary: Record<string, unknown> = {};
	for (const arm of arms) {
		const scores = scoresOf(arm.name);
		const vsDense = pairedBootstrapDiff(scores.ndcg, baseline.ndcg, { resamples: 2000, seed: 11 });
		const vsHybrid = pairedBootstrapDiff(scores.ndcg, hybridBaseline.ndcg, { resamples: 2000, seed: 13 });
		const ci = (d: BootstrapDiff): string =>
			`${d.mean >= 0 ? "+" : ""}${fmt(d.mean)} [${fmt(d.low)}, ${fmt(d.high)}]`;
		lines.push(
			`| ${arm.name} | ${fmt(mean(scores.ndcg))} | ${fmt(mean(scores.recall10))} | ${fmt(mean(scores.mrr))} | ${fmt(mean(scores.poolRecall))} | ${mean(scores.poolSize).toFixed(1)} | ${ci(vsDense)} | ${ci(vsHybrid)} |`,
		);
		summary[arm.name] = {
			ndcg10: mean(scores.ndcg),
			recall10: mean(scores.recall10),
			mrr10: mean(scores.mrr),
			poolRecall: mean(scores.poolRecall),
			meanPoolSize: mean(scores.poolSize),
			vsDenseRerank: vsDense,
			vsHybridRerank: vsHybrid,
		};
	}
	const timingLines = [...timings].map(
		([label, { calls, ms }]) => `${label}: ${calls} calls, mean ${(ms / calls).toFixed(0)} ms`,
	);
	const report = `${lines.join("\n")}\n\n${timingLines.join("\n")}\n`;
	console.log(report);
	if (values.out !== undefined) {
		writeFileSync(
			values.out,
			JSON.stringify(
				{
					data: dataDir,
					rerankModel,
					depth,
					queries: queries.length,
					corpus: corpus.length,
					summary,
					timings: Object.fromEntries(timings),
				},
				null,
				2,
			),
		);
	}
}

try {
	await main();
} finally {
	if (!values.keep && values.workspace === undefined) rmSync(root, { recursive: true, force: true });
}
