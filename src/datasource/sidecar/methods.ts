/**
 * RetrievalMethod adapters over a {@link CrawlerSidecar}. One class serves every
 * crawler and every mode; the only per-instance input is the datasource identity
 * (the channel allowlist is applied by the sidecar's SQL).
 */

import { FallbackRetrievalMethod } from "../../retrieval/fallback.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../retrieval/types.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../scope.ts";
import type { CrawlerDescriptor } from "./descriptors.ts";
import type { CrawlerSidecar, SidecarHit } from "./index.ts";

export type SidecarMode = "bm25" | "hybrid";

export interface SidecarMethodOptions {
	/** Resolve lazily when the native CLI owns the archive location. */
	readonly sidecar: CrawlerSidecar | ((signal?: AbortSignal) => Promise<CrawlerSidecar>);
	readonly descriptor: CrawlerDescriptor;
	readonly mode: SidecarMode;
	readonly datasourceId: string;
	readonly skillType: string;
	readonly instanceId: string;
	readonly tags: readonly string[];
}

const DEFAULT_TOP_K = 20;

/** Result method name; `bm25` keeps the `-sidecar-` infix so it never collides with the CLI-backed `-fts` method. */
export function sidecarMethodName(datasourceId: string, mode: SidecarMode): string {
	return mode === "bm25" ? `${datasourceId}-sidecar-bm25` : `${datasourceId}-${mode}`;
}

const DESCRIPTIONS: Record<
	SidecarMode,
	{ type: RetrievalMethodDescriptor["type"]; text: string; capabilities: string[] }
> = {
	bm25: { type: "bm25", text: "BM25-ranked FTS5 retrieval", capabilities: ["lexical", "fts5", "bm25"] },
	hybrid: {
		type: "hybrid",
		text: "Hybrid BM25 + semantic retrieval (reciprocal rank fusion)",
		capabilities: ["lexical", "semantic", "hybrid"],
	},
};

export class SidecarRetrievalMethod implements RetrievalMethod {
	private readonly options: SidecarMethodOptions;

	constructor(options: SidecarMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		const { datasourceId, skillType, mode, tags } = this.options;
		const { type, text, capabilities } = DESCRIPTIONS[mode];
		return {
			name: sidecarMethodName(datasourceId, mode),
			type,
			description: `${text} over ${skillType} via a read-only sidecar index`,
			status: "active",
			capabilities: [...capabilities, "scoped", "path-opaque-sources"],
			datasourceId,
			tags,
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		const topK = options.topK ?? DEFAULT_TOP_K;
		if (topK <= 0) return [];
		options.signal?.throwIfAborted();
		const { mode } = this.options;
		const sidecar =
			typeof this.options.sidecar === "function" ? await this.options.sidecar(options.signal) : this.options.sidecar;
		options.signal?.throwIfAborted();
		const hits = mode === "bm25" ? sidecar.searchLexical(trimmed, topK) : await sidecar.searchHybrid(trimmed, topK);
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const result = this.toResult(hit, options);
			if (result !== undefined) results.push(result);
		}
		return results;
	}

	private toResult(hit: SidecarHit, options: RetrievalOptions): RetrievalResult | undefined {
		const { descriptor, datasourceId, instanceId, mode } = this.options;
		const chunkId = descriptor.chunkId(hit.key, hit.metadata);
		const source = datasourceSourcePath(datasourceId, instanceId, chunkId);
		if (!matchesDatasourceScope(source, options.scope)) return undefined;
		const method = sidecarMethodName(datasourceId, mode);
		return {
			id: `${datasourceId}:${instanceId}:${chunkId}`,
			content: hit.content,
			source,
			score: hit.score,
			metadata: {
				...hit.metadata,
				method,
				datasourceId,
				instanceId,
				backend: "sidecar",
				...(hit.title !== undefined ? { title: hit.title } : {}),
				hierarchy: descriptor.hierarchy(hit.metadata).join("/"),
				...(hit.publishedAt !== undefined ? { publishedAt: hit.publishedAt } : {}),
			},
		};
	}
}

/**
 * The one method a sidecar-backed crawler registers. The sidecar's hybrid search leads when semantic
 * search is enabled and its BM25 search leads otherwise. When the leader throws, the sidecar's own BM25
 * answers next and the crawler's CLI search last, so the datasource still searches when the embedder or
 * the sidecar index is down. A leader that answers with no hits is final.
 */
export function sidecarFirstMethod(
	options: Omit<SidecarMethodOptions, "mode" | "sidecar"> & { readonly sidecar: CrawlerSidecar },
	cli: RetrievalMethod,
): RetrievalMethod {
	const bm25 = new SidecarRetrievalMethod({ ...options, mode: "bm25" });
	if (!options.sidecar.semanticEnabled) return new FallbackRetrievalMethod(bm25, [cli]);
	return new FallbackRetrievalMethod(new SidecarRetrievalMethod({ ...options, mode: "hybrid" }), [bm25, cli]);
}
