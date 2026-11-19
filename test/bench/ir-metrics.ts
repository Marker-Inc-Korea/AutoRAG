/**
 * Ranking metrics for the retrieval-arm benchmark. A ranking is a list of document ids, best first;
 * `qrels` maps a relevant document id to its graded gain. A document counts once, at its first rank.
 */

function firstOccurrences(ranking: readonly string[], k: number): string[] {
	const seen = new Set<string>();
	const unique: string[] = [];
	for (const id of ranking) {
		if (seen.has(id)) continue;
		seen.add(id);
		unique.push(id);
		if (unique.length >= k) break;
	}
	return unique;
}

export function ndcgAtK(ranking: readonly string[], qrels: ReadonlyMap<string, number>, k: number): number {
	const gains = [...qrels.values()].filter((gain) => gain > 0).sort((a, b) => b - a);
	if (gains.length === 0) return 0;
	let dcg = 0;
	for (const [index, id] of firstOccurrences(ranking, k).entries()) {
		dcg += (qrels.get(id) ?? 0) / Math.log2(index + 2);
	}
	let ideal = 0;
	for (const [index, gain] of gains.slice(0, k).entries()) ideal += gain / Math.log2(index + 2);
	return dcg / ideal;
}

export function recallAtK(ranking: readonly string[], qrels: ReadonlyMap<string, number>, k: number): number {
	const relevant = [...qrels].filter(([, gain]) => gain > 0).map(([id]) => id);
	if (relevant.length === 0) return 0;
	const top = new Set(firstOccurrences(ranking, k));
	return relevant.filter((id) => top.has(id)).length / relevant.length;
}

export function mrrAtK(ranking: readonly string[], qrels: ReadonlyMap<string, number>, k: number): number {
	for (const [index, id] of firstOccurrences(ranking, k).entries()) {
		if ((qrels.get(id) ?? 0) > 0) return 1 / (index + 1);
	}
	return 0;
}

export interface BootstrapDiff {
	/** Mean of `a - b` over the queries. */
	readonly mean: number;
	/** 2.5th percentile of the resampled mean difference. */
	readonly low: number;
	/** 97.5th percentile of the resampled mean difference. */
	readonly high: number;
}

/** mulberry32: a small seeded generator, so a benchmark run's intervals reproduce. */
function seededRandom(seed: number): () => number {
	let state = seed >>> 0;
	return () => {
		state = (state + 0x6d2b79f5) >>> 0;
		let value = Math.imul(state ^ (state >>> 15), state | 1);
		value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
		return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
	};
}

/** Paired percentile bootstrap over queries of the mean per-query difference `a - b`. */
export function pairedBootstrapDiff(
	a: readonly number[],
	b: readonly number[],
	options: { readonly resamples: number; readonly seed: number },
): BootstrapDiff {
	if (a.length !== b.length) throw new Error("pairedBootstrapDiff needs score lists of the same length");
	const diffs = a.map((score, index) => score - (b[index] as number));
	const mean = diffs.reduce((sum, value) => sum + value, 0) / Math.max(diffs.length, 1);
	if (diffs.length === 0) return { mean: 0, low: 0, high: 0 };
	const random = seededRandom(options.seed);
	const means: number[] = [];
	for (let resample = 0; resample < options.resamples; resample++) {
		let sum = 0;
		for (let draw = 0; draw < diffs.length; draw++) sum += diffs[Math.floor(random() * diffs.length)] as number;
		means.push(sum / diffs.length);
	}
	means.sort((x, y) => x - y);
	const at = (quantile: number): number =>
		means[Math.min(means.length - 1, Math.floor(quantile * means.length))] as number;
	return { mean, low: at(0.025), high: at(0.975) };
}
