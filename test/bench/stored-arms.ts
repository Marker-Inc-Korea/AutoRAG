import { createHash } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { unionChannelResults } from "../../src/minsync/channel-union.ts";
import { createReranker, type Reranker } from "../../src/retrieval/rerank.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { type BootstrapDiff, mrrAtK, ndcgAtK, pairedBootstrapDiff, recallAtK } from "./ir-metrics.ts";

/**
 * Re-ranks the candidates a past MinSync run stored, so no corpus has to be re-indexed.
 *
 *   bun test/bench/stored-arms.ts --queries queries.json --results results.json --rerank-model ID
 *
 * `queries.json`: a list of `{ path, usable, ... }`; the gold answer is the row's source file `path`. Stored
 * result keys are `<position>-<kind>` where position counts only the `usable` rows, per engine. `results.json`: `{ results: { <engine>: { "<position>-<kind>":
 * { hits: [{ path, text }] } } } }`. Arms are built from the dense (`--dense`), SPLADE (`--sparse`) and MinSync
 * hybrid (`--hybrid`) engine runs. The chunk text is sent to the rerank provider, so only run this on a corpus
 * whose owner approved sending it there. Scoring is file level: a ranking collapses to its files, first hit wins.
 */

interface QueryRow {
	readonly path: string;
	readonly usable: string | boolean;
}

interface StoredHit {
	readonly path: string;
	readonly text: string;
}

interface StoredResults {
	readonly results: Record<string, Record<string, { readonly hits: readonly StoredHit[] }>>;
}

const { values } = parseArgs({
	options: {
		queries: { type: "string" },
		results: { type: "string" },
		"rerank-model": { type: "string" },
		"rerank-provider": { type: "string", default: "openrouter" },
		dense: { type: "string", default: "vector@0.6.0" },
		sparse: { type: "string", default: "sparse@0.6.0" },
		hybrid: { type: "string", default: "hybrid@0.6.0" },
		kinds: { type: "string", default: "natural,keyword" },
		cache: { type: "string" },
		out: { type: "string" },
	},
});
if (values.queries === undefined || values.results === undefined || values["rerank-model"] === undefined) {
	throw new Error("--queries, --results and --rerank-model are required");
}
const rerankModel = values["rerank-model"];
const queryRows = JSON.parse(readFileSync(values.queries, "utf8")) as QueryRow[];
const stored = JSON.parse(readFileSync(values.results, "utf8")) as StoredResults;
const usable = queryRows.flatMap((row, index) => (String(row.usable) === "True" || row.usable === true ? [index] : []));
// The rerank request needs the question text; `natural` is the question, `keyword` the keyword query.
const questions = queryRows as unknown as Record<string, unknown>[];

function engine(name: string): Record<string, { readonly hits: readonly StoredHit[] }> {
	const found = stored.results[name];
	if (found === undefined) throw new Error(`results.json has no engine "${name}"; has ${Object.keys(stored.results)}`);
	return found;
}
const dense = engine(values.dense ?? "vector@0.6.0");
const sparse = engine(values.sparse ?? "sparse@0.6.0");
const hybrid = engine(values.hybrid ?? "hybrid@0.6.0");

const builtReranker = createReranker({ provider: values["rerank-provider"], model: rerankModel });
if (builtReranker === undefined || !builtReranker.describe().available) {
	throw new Error(`reranker unavailable: ${builtReranker?.describe().reason ?? "not built"}`);
}
const reranker: Reranker = builtReranker;

const cachePath = values.cache;
const rerankCache: Record<string, string[]> =
	cachePath !== undefined && existsSync(cachePath)
		? (JSON.parse(readFileSync(cachePath, "utf8")) as Record<string, string[]>)
		: {};

function toResults(hits: readonly StoredHit[]): RetrievalResult[] {
	return hits.map((hit) => ({
		id: createHash("sha1").update(`${hit.path}\0${hit.text}`).digest("hex"),
		source: hit.path,
		content: hit.text,
		score: 1,
		metadata: {},
	}));
}

function fileRanking(results: readonly RetrievalResult[]): string[] {
	const seen = new Set<string>();
	const ranking: string[] = [];
	for (const result of results) {
		if (seen.has(result.source)) continue;
		seen.add(result.source);
		ranking.push(result.source);
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

interface Pools {
	readonly dense: RetrievalResult[];
	readonly sparse: RetrievalResult[];
	readonly hybrid: RetrievalResult[];
	readonly union: RetrievalResult[];
	readonly unionCapped: RetrievalResult[];
}

interface Arm {
	readonly name: string;
	readonly pool: (pools: Pools) => RetrievalResult[];
	readonly rerank: boolean;
}

const ARMS: readonly Arm[] = [
	{ name: "dense", pool: (p) => p.dense, rerank: false },
	{ name: "sparse (SPLADE)", pool: (p) => p.sparse, rerank: false },
	{ name: "hybrid (MinSync RRF)", pool: (p) => p.hybrid, rerank: false },
	{ name: "union (interleaved, no rerank)", pool: (p) => p.union, rerank: false },
	{ name: "dense -> rerank (before)", pool: (p) => p.dense, rerank: true },
	{ name: "sparse -> rerank", pool: (p) => p.sparse, rerank: true },
	{ name: "hybrid -> rerank", pool: (p) => p.hybrid, rerank: true },
	{ name: "union capped at N -> rerank", pool: (p) => p.unionCapped, rerank: true },
	{ name: "union (dense N + sparse N) -> rerank (this PR)", pool: (p) => p.union, rerank: true },
];

const BASELINE = "dense -> rerank (before)";
const SPARSE_BASELINE = "sparse -> rerank";
const mean = (xs: readonly number[]): number => xs.reduce((sum, x) => sum + x, 0) / Math.max(xs.length, 1);
const fmt = (value: number): string => value.toFixed(4);
const ci = (d: BootstrapDiff): string => `${d.mean >= 0 ? "+" : ""}${fmt(d.mean)} [${fmt(d.low)}, ${fmt(d.high)}]`;

interface ArmScores {
	readonly ndcg: number[];
	readonly recall1: number[];
	readonly recall10: number[];
	readonly mrr: number[];
	readonly poolRecall: number[];
	readonly poolSize: number[];
}

async function runKind(kind: string): Promise<string> {
	const rows = usable.flatMap((queryIndex, position) => {
		// Stored result keys count the usable queries only (`<position>-<kind>`), not raw queries.json rows.
		const key = `${position}-${kind}`;
		const d = dense[key];
		const s = sparse[key];
		const h = hybrid[key];
		const question = questions[queryIndex]?.[kind === "keyword" ? "keyword" : "natural"];
		if (d === undefined || s === undefined || h === undefined || typeof question !== "string") return [];
		const denseHits = toResults(d.hits);
		const sparseHits = toResults(s.hits);
		const channels = [
			{ channel: "vector", results: denseHits },
			{ channel: "sparse", results: sparseHits },
		];
		const depth = denseHits.length;
		const pools: Pools = {
			dense: denseHits,
			sparse: sparseHits,
			hybrid: toResults(h.hits),
			union: unionChannelResults(channels),
			unionCapped: unionChannelResults(channels, { limit: depth }),
		};
		return [{ question, gold: (queryRows[queryIndex] as QueryRow).path, pools }];
	});

	const scores = new Map<string, ArmScores>();
	for (const arm of ARMS) {
		const armScores: ArmScores = { ndcg: [], recall1: [], recall10: [], mrr: [], poolRecall: [], poolSize: [] };
		scores.set(arm.name, armScores);
		const rankings = await limited(rows, arm.rerank ? 8 : 16, async (row) => {
			const pool = arm.pool(row.pools);
			if (!arm.rerank) return fileRanking(pool);
			const key = createHash("sha1")
				.update(`${rerankModel}\0${row.question}\0${pool.map((r) => r.id).join("|")}`)
				.digest("hex");
			const cached = rerankCache[key];
			if (cached !== undefined) return cached;
			const ranking = fileRanking(await reranker.rerank(row.question, pool));
			rerankCache[key] = ranking;
			return ranking;
		});
		for (const [position, row] of rows.entries()) {
			const ranking = rankings[position] as string[];
			const qrels = new Map([[row.gold, 1]]);
			armScores.ndcg.push(ndcgAtK(ranking, qrels, 10));
			armScores.recall1.push(recallAtK(ranking, qrels, 1));
			armScores.recall10.push(recallAtK(ranking, qrels, 10));
			armScores.mrr.push(mrrAtK(ranking, qrels, 10));
			armScores.poolRecall.push(recallAtK(fileRanking(arm.pool(row.pools)), qrels, Number.MAX_SAFE_INTEGER));
			armScores.poolSize.push(arm.pool(row.pools).length);
		}
		if (cachePath !== undefined) writeFileSync(cachePath, JSON.stringify(rerankCache));
		console.error(`[${kind}] arm done: ${arm.name}`);
	}

	const base = scores.get(BASELINE) as ArmScores;
	const sparseBase = scores.get(SPARSE_BASELINE) as ArmScores;
	const lines = [
		`### ${kind} queries (n=${rows.length}, file-level gold)`,
		"",
		"| arm | nDCG@10 | R@1 | R@10 | MRR@10 | pool recall | mean pool | Δ nDCG vs dense->rerank [95% CI] | Δ nDCG vs sparse->rerank [95% CI] |",
		"|---|---|---|---|---|---|---|---|---|",
	];
	for (const arm of ARMS) {
		const s = scores.get(arm.name) as ArmScores;
		const diff = pairedBootstrapDiff(s.ndcg, base.ndcg, { resamples: 2000, seed: 17 });
		const diffSparse = pairedBootstrapDiff(s.ndcg, sparseBase.ndcg, { resamples: 2000, seed: 19 });
		lines.push(
			`| ${arm.name} | ${fmt(mean(s.ndcg))} | ${fmt(mean(s.recall1))} | ${fmt(mean(s.recall10))} | ${fmt(mean(s.mrr))} | ${fmt(mean(s.poolRecall))} | ${mean(s.poolSize).toFixed(1)} | ${ci(diff)} | ${ci(diffSparse)} |`,
		);
	}
	return lines.join("\n");
}

const kinds = (values.kinds ?? "natural,keyword").split(",");
const sections: string[] = [];
for (const kind of kinds) sections.push(await runKind(kind));
const report = `${sections.join("\n\n")}\n`;
console.log(report);
if (values.out !== undefined) writeFileSync(values.out, report);
