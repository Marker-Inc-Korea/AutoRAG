import { describeRetrievalError } from "../../../retrieval/skip.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { DatasourceCliError } from "../../errors.ts";
import { matchesDatasourceScope } from "../../scope.ts";
import { reciprocalRankFusion } from "../../sidecar/fusion.ts";
import type { DiscrawlSearchMode } from "./types.ts";

export interface DiscrawlRankedMethodOptions {
	readonly mode: DiscrawlSearchMode;
	readonly lexical: RetrievalMethod;
	readonly nativeFts: RetrievalMethod;
	readonly semantic: RetrievalMethod;
	readonly workspaceId?: string;
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
}

const DEFAULT_TOP_K = 20;
const CANDIDATE_MULTIPLIER = 3;

/** One Discord ranking boundary: native FTS supplements BM25 rather than voting for the same messages again. */
export class DiscrawlRankedMethod implements RetrievalMethod {
	private readonly options: DiscrawlRankedMethodOptions;

	constructor(options: DiscrawlRankedMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		const { mode, nativeFts, semantic } = this.options;
		if (mode === "semantic") return semantic.describe();
		return {
			...nativeFts.describe(),
			name: `discord-${mode}`,
			type: mode === "hybrid" ? "hybrid" : "bm25",
			description:
				mode === "hybrid"
					? "Hybrid BM25 and native semantic retrieval over Discord, with native FTS supplemental recall"
					: "BM25-ranked Discord lexical retrieval, with native FTS supplemental recall",
			capabilities: [
				"lexical",
				"fts5",
				"bm25",
				"scoped",
				"external-cli",
				...(mode === "hybrid" ? ["semantic", "hybrid"] : []),
			],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		options.signal?.throwIfAborted();
		const trimmed = query.trim();
		const topK = options.topK ?? DEFAULT_TOP_K;
		if (trimmed.length === 0 || topK <= 0) return [];
		const { mode, lexical, nativeFts, semantic } = this.options;
		if (mode === "semantic") {
			const hits = await semantic.retrieve(trimmed, options);
			options.signal?.throwIfAborted();
			return hits.filter((hit) => this.matchesScope(hit, options.scope)).slice(0, topK);
		}

		const methods = mode === "hybrid" ? [lexical, nativeFts, semantic] : [lexical, nativeFts];
		const poolOptions = { ...options, topK: topK * CANDIDATE_MULTIPLIER };
		const settled = await Promise.allSettled(
			methods.map((method) => Promise.resolve().then(() => method.retrieve(trimmed, poolOptions))),
		);
		options.signal?.throwIfAborted();
		const failures: { method: string; error: unknown; reason: string }[] = [];
		const lists = settled.map((result, index) => {
			if (result.status === "fulfilled") return result.value.filter((hit) => this.matchesScope(hit, options.scope));
			const error: unknown = result.reason;
			if (
				(error instanceof Error && error.name === "AbortError") ||
				(error instanceof DatasourceCliError && error.reason === "aborted")
			) {
				throw error;
			}
			failures.push({ method: methods[index]!.describe().name, error, reason: describeRetrievalError(error) });
			return [];
		});

		const evidence = new Map<string, RetrievalResult>();
		const keys = (hits: readonly RetrievalResult[]): string[] => {
			const unique = new Set<string>();
			for (const hit of hits) {
				const key = `${hit.source}\0${hit.id}`;
				if (!evidence.has(key)) evidence.set(key, hit);
				unique.add(key);
			}
			return [...unique];
		};
		// Native language-aware FTS can find messages absent from message_fts. Keep those
		// candidates after BM25 matches, without granting duplicated lexical evidence a second vote.
		const lexicalKeys = keys([...(lists[0] ?? []), ...(lists[1] ?? [])]);
		const semanticKeys = keys(lists[2] ?? []);
		if (evidence.size === 0 && failures.length > 0) {
			if (failures.length === 1) throw failures[0]!.error;
			throw new AggregateError(
				failures.map((failure) => failure.error),
				failures.map((failure) => failure.reason).join("\n"),
			);
		}
		const diagnostics = failures.map(({ method, reason }) => ({ method, reason }));
		// Rank scores are required even for one list: the outer merger sorts numerically,
		// and a supplemental native score must never overtake a stronger BM25 match.
		return reciprocalRankFusion([lexicalKeys, semanticKeys])
			.slice(0, topK)
			.map(({ key, score }) => {
				const hit = evidence.get(key)!;
				return {
					...hit,
					score,
					metadata: {
						...hit.metadata,
						method: `discord-${mode}`,
						mode,
						...(diagnostics.length > 0 ? { retrievalDiagnostics: diagnostics } : {}),
						...(settled[2]?.status === "rejected" ? { diagnostic: "semantic-unavailable" } : {}),
					},
				};
			});
	}

	private matchesScope(hit: RetrievalResult, scope: string | undefined): boolean {
		if (!matchesDatasourceScope(hit.source, scope)) return false;
		const { workspaceId, channelIds = [], channelNames = [] } = this.options;
		if (workspaceId !== undefined && hit.metadata.guildId !== workspaceId) return false;
		if (channelIds.length === 0 && channelNames.length === 0) return true;
		return (
			(typeof hit.metadata.channelId === "string" && channelIds.includes(hit.metadata.channelId)) ||
			(typeof hit.metadata.channelName === "string" && channelNames.includes(hit.metadata.channelName))
		);
	}
}
