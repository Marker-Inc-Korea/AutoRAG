/** Standard RRF constant (Cormack et al.); damps the head of each list. */
const RRF_K = 60;

export interface ScoredKey {
	readonly key: string;
	readonly score: number;
}

/**
 * Reciprocal rank fusion over ranked key lists (best first). A key repeated
 * inside one list counts once, at its best rank. Ties keep first-seen order,
 * so earlier lists win.
 */
export function reciprocalRankFusion(lists: readonly (readonly string[])[]): ScoredKey[] {
	const scores = new Map<string, number>();
	for (const list of lists) {
		const seen = new Set<string>();
		for (const [index, key] of list.entries()) {
			if (seen.has(key)) continue;
			seen.add(key);
			scores.set(key, (scores.get(key) ?? 0) + 1 / (RRF_K + index + 1));
		}
	}
	return [...scores.entries()].map(([key, score]) => ({ key, score })).sort((a, b) => b.score - a.score);
}
