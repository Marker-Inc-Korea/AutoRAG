import { reciprocalRankFusion } from "./fusion.ts";
import { describeRetrievalError } from "./skip.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "./types.ts";

export interface FusedRetrievalMethodOptions {
	readonly name: string;
	readonly lexical: RetrievalMethod;
	readonly semantic: RetrievalMethod;
}

const DEFAULT_TOP_K = 20;
const CANDIDATE_MULTIPLIER = 3;

/**
 * `metadata.mode` for every fused hit. A fused hit is never the output of one
 * native search mode, so it must not keep the first-seen input's `keyword` or
 * `semantic` value; the `hybrid` marker matches the descriptor `type`.
 */
const FUSED_MODE = "hybrid";

/** Deterministic hybrid retrieval over two existing ranked methods; no indexing or query expansion. */
export class FusedRetrievalMethod implements RetrievalMethod {
	private readonly options: FusedRetrievalMethodOptions;

	constructor(options: FusedRetrievalMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		const lexical = this.options.lexical.describe();
		const semantic = this.options.semantic.describe();
		return {
			...lexical,
			name: this.options.name,
			type: "hybrid",
			description: `Hybrid retrieval over ${lexical.name} and ${semantic.name} (reciprocal rank fusion)`,
			capabilities: [
				...new Set([
					...lexical.capabilities.filter((capability) => semantic.capabilities.includes(capability)),
					"lexical",
					"semantic",
					"hybrid",
				]),
			],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		options.signal?.throwIfAborted();
		const topK = options.topK ?? DEFAULT_TOP_K;
		if (query.trim().length === 0 || topK <= 0) return [];
		const poolOptions = { ...options, topK: topK * CANDIDATE_MULTIPLIER };
		const settled = await Promise.allSettled(
			[this.options.lexical, this.options.semantic].map((method) =>
				Promise.resolve().then(() => method.retrieve(query, poolOptions)),
			),
		);
		options.signal?.throwIfAborted();
		const failures = settled.filter((result) => result.status === "rejected");
		for (const failure of failures) {
			if (failure.reason instanceof Error && failure.reason.name === "AbortError") throw failure.reason;
		}
		if (failures.length === settled.length) {
			const reasons = failures.map((failure) => failure.reason);
			throw new AggregateError(reasons, reasons.map(describeRetrievalError).join("\n"));
		}
		const lists = settled.map((result) => (result.status === "fulfilled" ? result.value : []));
		const nonempty = lists.filter((list) => list.length > 0);
		// Nothing survived to fuse or fall back to. If an input failed, surface it:
		// retrieveWithDiagnostics then reports this surface instead of returning an
		// empty list that reads like a successful search with no matches. This is the
		// only signal a hybrid-only selection (`methods: ["kakao-hybrid"]`) gets.
		if (nonempty.length === 0 && failures.length > 0) {
			const reasons = failures.map((failure) => failure.reason);
			throw new AggregateError(reasons, reasons.map(describeRetrievalError).join("\n"));
		}
		// A missing side must not change the surviving side's rank or score scale.
		if (nonempty.length < 2) return (nonempty[0] ?? []).slice(0, topK).map((hit) => this.withMethod(hit));

		const evidence = new Map<string, RetrievalResult>();
		const rankedKeys = lists.map((list) =>
			list.map((hit) => {
				// Match ResultMerger's evidence identity, retaining distinct chunks of one source.
				const key = `${hit.source}\0${hit.id}`;
				if (!evidence.has(key)) evidence.set(key, hit);
				return key;
			}),
		);
		return reciprocalRankFusion(rankedKeys)
			.slice(0, topK)
			.map(({ key, score }) => this.withMethod({ ...evidence.get(key)!, score }));
	}

	private withMethod(hit: RetrievalResult): RetrievalResult {
		// The hit is the hybrid method's output, not a single native search: replace
		// the first-seen input's `mode` so metadata never points at one input alone.
		return { ...hit, metadata: { ...hit.metadata, method: this.options.name, mode: FUSED_MODE } };
	}
}
