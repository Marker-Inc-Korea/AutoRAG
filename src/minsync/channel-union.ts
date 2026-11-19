import type { RetrievalResult } from "../retrieval/types.ts";

/** Ranked results of one MinSync query mode (`vector` or `sparse`), best first. */
export interface MinSyncChannelResults {
	readonly channel: string;
	readonly results: readonly RetrievalResult[];
}

export interface UnionChannelOptions {
	/** Keep at most this many chunks after interleaving. Unset keeps the whole union. */
	readonly limit?: number;
}

/**
 * Union of independently ranked MinSync channels, built for a reranker that decides the final order.
 *
 * Channels are interleaved by rank (rank 1 of each channel, then rank 2, ...) so a limit never cuts one
 * channel's head to keep the other's tail, and a chunk found by several channels appears once, at its
 * best interleaved position. Scores are left as the first channel reported them: dense and SPLADE scores
 * are not comparable and nothing here ranks across them. Each chunk records which channels found it and
 * at what 1-based rank (`minsyncChannels`, `minsyncChannelRanks`).
 *
 * Identity is `source` + `id`, the same key the cross-method merger uses, so two chunks of one document
 * stay distinct while the same chunk reached by two channels collapses.
 */
export function unionChannelResults(
	channels: readonly MinSyncChannelResults[],
	options: UnionChannelOptions = {},
): RetrievalResult[] {
	const byChunk = new Map<string, { result: RetrievalResult; ranks: Record<string, number> }>();
	const depth = Math.max(0, ...channels.map((channel) => channel.results.length));
	for (let rank = 0; rank < depth; rank++) {
		for (const channel of channels) {
			const result = channel.results[rank];
			if (result === undefined) continue;
			const key = `${result.source}\0${result.id}`;
			const existing = byChunk.get(key);
			if (existing === undefined) byChunk.set(key, { result, ranks: { [channel.channel]: rank + 1 } });
			else existing.ranks[channel.channel] ??= rank + 1;
		}
	}
	const union = [...byChunk.values()].map(({ result, ranks }) => ({
		...result,
		metadata: {
			...result.metadata,
			minsyncChannels: Object.keys(ranks).sort(),
			minsyncChannelRanks: ranks,
		},
	}));
	return options.limit === undefined ? union : union.slice(0, options.limit);
}
