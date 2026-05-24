import type { RetrievalMethod, RetrievalOptions, RetrievalResult } from "./types.ts";

export interface MergeOptions {
	topK: number;
	dedup: boolean;
}

export class ResultMerger {
	merge(results: Map<string, RetrievalResult[]>, options: MergeOptions): RetrievalResult[] {
		const { topK, dedup } = options;
		if (results.size === 0) return [];

		// Normalize scores per method using min-max normalization
		const normalized = new Map<string, RetrievalResult[]>();
		for (const [method, methodResults] of results) {
			if (methodResults.length === 0) {
				normalized.set(method, []);
				continue;
			}
			const scores = methodResults.map((r) => r.score);
			const min = Math.min(...scores);
			const max = Math.max(...scores);
			const range = max - min;
			normalized.set(
				method,
				methodResults.map((r) => ({
					...r,
					score: range > 0 ? (r.score - min) / range : 1.0,
				})),
			);
		}

		// Merge all results
		const allResults: RetrievalResult[] = [];
		for (const methodResults of normalized.values()) {
			allResults.push(...methodResults);
		}

		if (dedup) {
			// Deduplicate by source — keep highest scoring
			const bySource = new Map<string, RetrievalResult>();
			for (const r of allResults) {
				const existing = bySource.get(r.source);
				if (!existing || r.score > existing.score) {
					bySource.set(r.source, r);
				}
			}
			const deduped = Array.from(bySource.values());
			return deduped.sort((a, b) => b.score - a.score).slice(0, topK);
		}

		return allResults.sort((a, b) => b.score - a.score).slice(0, topK);
	}
}

export class ParallelRetriever {
	async retrieve(
		methods: RetrievalMethod[],
		query: string,
		options: RetrievalOptions,
	): Promise<Map<string, RetrievalResult[]>> {
		const results = new Map<string, RetrievalResult[]>();
		await Promise.all(
			methods.map(async (method) => {
				const name = method.describe().name;
				try {
					const methodResults = await method.retrieve(query, options);
					results.set(name, methodResults);
				} catch {
					results.set(name, []);
				}
			}),
		);
		return results;
	}
}
