import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import { OVERFETCH_ABSOLUTE_CAP, RecordCut } from "../../src/retrieval/record-cut.ts";
import { type BootstrapDiff, mrrAtK, ndcgAtK, pairedBootstrapDiff, recallAtK } from "./ir-metrics.ts";

/**
 * Scores the record cut on a corpus that is already indexed, read-only.
 *
 *   bun test/bench/record-cut-arms.ts --queries queries.json --workspace <dir holding .minsync> --out run.json
 *
 * `queries.json`: a list of `{ path, usable, natural, keyword }`; the gold answer is the row's source file
 * `path` (a MinSync `files/...md` path). Each query is sent to `minsync query` once per arm exactly as the
 * product would: the OFF arm asks for `k` hits, the ON arm asks for `RecordCut.fusedLimit(k)` hits and cuts them
 * back to `k` by document. The same code path as `MinSyncVectorMethod`, so the arms differ only in the cut.
 *
 * Nothing is written to the workspace or to ~/.autorag: `minsync query` only reads, and the run refuses a
 * workspace under ~/.autorag unless --allow-real-workspace is given, because a query there shares the user's
 * live index. Chunk text stays on this machine; no model is called.
 */

interface QueryRow {
	readonly path: string;
	readonly usable: string | boolean;
	readonly natural?: string;
	readonly keyword?: string;
}

interface Hit {
	readonly path: string;
	readonly text: string;
}

const { values } = parseArgs({
	options: {
		queries: { type: "string" },
		workspace: { type: "string" },
		mode: { type: "string", default: "hybrid" },
		kinds: { type: "string", default: "natural,keyword" },
		k: { type: "string", default: "10" },
		limit: { type: "string" },
		out: { type: "string" },
		"allow-real-workspace": { type: "boolean", default: false },
	},
});
if (values.queries === undefined || values.workspace === undefined)
	throw new Error("--queries and --workspace are required");
if (values.workspace.includes("/.autorag/") && values["allow-real-workspace"] !== true) {
	throw new Error("refusing a workspace under ~/.autorag without --allow-real-workspace (read-only, but it is live)");
}
if (!existsSync(`${values.workspace}/.minsync`)) throw new Error(`no .minsync under ${values.workspace}`);

const k = Number(values.k);
const cut = new RecordCut();
const rows = (JSON.parse(readFileSync(values.queries, "utf8")) as QueryRow[]).filter(
	(row) => String(row.usable) === "True" || row.usable === true,
);
const kinds = values.kinds.split(",") as ("natural" | "keyword")[];
const selected = values.limit === undefined ? rows : rows.slice(0, Number(values.limit));

function query(text: string, count: number): Hit[] {
	const result = spawnSync(
		"minsync",
		["query", "--format", "json", "--mode", values.mode as string, "-k", String(count), text],
		{
			cwd: values.workspace,
			encoding: "utf8",
			maxBuffer: 256 * 1024 * 1024,
		},
	);
	if (result.status !== 0) throw new Error(`minsync query failed: ${result.stderr}`);
	return JSON.parse(result.stdout) as Hit[];
}

/** Ranked file paths, first hit of each file wins. */
const files = (hits: readonly Hit[]): string[] => hits.map((hit) => hit.path);

interface Arm {
	readonly ndcg: number[];
	readonly recall: number[];
	readonly mrr: number[];
	/** Distinct files in the top k: how many documents the cut let into the result. */
	readonly distinct: number[];
}
const emptyArm = (): Arm => ({ ndcg: [], recall: [], mrr: [], distinct: [] });
const arms = { off: emptyArm(), on: emptyArm() };
const perQuery: { kind: string; path: string; off: string[]; on: string[] }[] = [];

let done = 0;
for (const row of selected) {
	const qrels = new Map([[row.path, 1]]);
	for (const kind of kinds) {
		const text = row[kind];
		if (text === undefined || text.length === 0) continue;
		const off = query(text, k).slice(0, k);
		const fused = cut.fusedLimit(k, OVERFETCH_ABSOLUTE_CAP);
		const on = cut.cut(query(text, fused), k, (hit) => hit.path);
		for (const [name, hits] of [
			["off", off],
			["on", on],
		] as const) {
			const ranking = files(hits);
			arms[name].ndcg.push(ndcgAtK(ranking, qrels, k));
			arms[name].recall.push(recallAtK(ranking, qrels, k));
			arms[name].mrr.push(mrrAtK(ranking, qrels, k));
			arms[name].distinct.push(new Set(ranking).size);
		}
		perQuery.push({ kind, path: row.path, off: files(off), on: files(on) });
	}
	done++;
	if (done % 25 === 0) console.error(`${done}/${selected.length} rows`);
}

const mean = (xs: readonly number[]): number => xs.reduce((sum, x) => sum + x, 0) / Math.max(xs.length, 1);
const diff = (a: readonly number[], b: readonly number[]): BootstrapDiff =>
	pairedBootstrapDiff(a, b, { resamples: 2000, seed: 7 });
const fmt = (d: BootstrapDiff): string =>
	`${d.mean >= 0 ? "+" : ""}${d.mean.toFixed(4)} [${d.low.toFixed(4)}, ${d.high.toFixed(4)}]`;

const report = {
	mode: values.mode,
	k,
	fusedLimit: cut.fusedLimit(k, OVERFETCH_ABSOLUTE_CAP),
	queries: arms.off.ndcg.length,
	off: {
		ndcg: mean(arms.off.ndcg),
		recall: mean(arms.off.recall),
		mrr: mean(arms.off.mrr),
		distinct: mean(arms.off.distinct),
	},
	on: {
		ndcg: mean(arms.on.ndcg),
		recall: mean(arms.on.recall),
		mrr: mean(arms.on.mrr),
		distinct: mean(arms.on.distinct),
	},
	diff: {
		ndcg: diff(arms.on.ndcg, arms.off.ndcg),
		recall: diff(arms.on.recall, arms.off.recall),
		mrr: diff(arms.on.mrr, arms.off.mrr),
	},
};
console.log(`mode=${report.mode} k=${k} fused=${report.fusedLimit} queries=${report.queries}`);
console.log(`          nDCG@${k}  Recall@${k}  MRR@${k}  distinct-files`);
for (const name of ["off", "on"] as const) {
	const r = report[name];
	console.log(
		`${name.padEnd(4)}      ${r.ndcg.toFixed(4)}   ${r.recall.toFixed(4)}     ${r.mrr.toFixed(4)}   ${r.distinct.toFixed(2)}`,
	);
}
console.log(`on - off  nDCG ${fmt(report.diff.ndcg)}  Recall ${fmt(report.diff.recall)}  MRR ${fmt(report.diff.mrr)}`);
if (values.out !== undefined) writeFileSync(values.out, JSON.stringify({ report, perQuery }, null, 2));
