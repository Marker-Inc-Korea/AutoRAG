import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { lazykatokSourcePath } from "./paths.ts";
import type {
	LazykatokChunk,
	LazykatokHit,
	LazykatokParentResult,
	LazykatokSearchMode,
	LazykatokSearchOptions,
	LazykatokSearchResult,
} from "./types.ts";

/**
 * Narrow client surface required by the KakaoTalk retrieval methods.
 * The real {@link LazykatokClient} satisfies this structurally; tests may stub it.
 */
export interface LazykatokSearchClient {
	search(mode: LazykatokSearchMode, query: string, options?: LazykatokSearchOptions): Promise<LazykatokSearchResult>;
}

export interface LazykatokMethodOptions {
	readonly client: LazykatokSearchClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

const KAKAO_DATASOURCE_ID = "kakao";
const DEFAULT_KAKAO_TAGS = ["kakaotalk", "personal", "pii"] as const;
const DEFAULT_TOP_K = 20;

/**
 * Lexical (BM25) retrieval over a KakaoTalk datasource via the external
 * `lazykatok` CLI in keyword mode. Lazykatok does not expose source scopes; access is
 * controlled at the datasource/tag level. All client failures collapse to an
 * empty result set; retrieval never throws.
 */
export class LazykatokBm25Method implements RetrievalMethod {
	private readonly client: LazykatokSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: LazykatokMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_KAKAO_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "kakao-bm25",
			type: "bm25",
			description: "BM25 lexical retrieval over KakaoTalk chat chunks via the external lazykatok CLI",
			status: "active",
			capabilities: ["lexical", "keyword-mode", "external-cli", "path-opaque-sources"],
			datasourceId: KAKAO_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveLazykatok(this.client, "keyword", "kakao-bm25", this.instanceId, query, options);
	}
}

/**
 * Semantic (vector) retrieval over a KakaoTalk datasource via the external
 * `lazykatok` CLI in semantic mode. Lazykatok does not expose source scopes; access is
 * controlled at the datasource/tag level. All client failures collapse to an
 * empty result set; retrieval never throws.
 */
export class LazykatokSemanticMethod implements RetrievalMethod {
	private readonly client: LazykatokSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: LazykatokMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_KAKAO_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "kakao-semantic",
			type: "vector",
			description: "Semantic vector retrieval over KakaoTalk chat chunks via the external lazykatok CLI",
			status: "active",
			capabilities: ["semantic", "vector-mode", "external-cli", "path-opaque-sources"],
			datasourceId: KAKAO_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveLazykatok(this.client, "semantic", "kakao-semantic", this.instanceId, query, options);
	}
}

/** Narrow client surface that resolves a micro-chunk to its parent window. */
export interface LazykatokParentClient {
	parent(chunkId: string, signal?: AbortSignal): Promise<LazykatokParentResult>;
}

export interface LazykatokWindowBm25Options {
	readonly client: LazykatokSearchClient & LazykatokParentClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

/** Parent-window lookups run at most this many `lazykatok` processes at a time. */
const PARENT_LOOKUP_CONCURRENCY = 8;

/**
 * BM25 retrieval whose hits are lifted to the parent windows the semantic
 * index ranks. lazykatok searches keywords over micro-chunks (`chunk_…`) but
 * semantic search over the parent windows that contain them (`window_…`), so
 * the two rankings only share an identity once each keyword chunk is treated as
 * a hit on its parent window. Several chunks of one window collapse to the
 * window's best-ranked chunk. This is the lexical input of `kakao-hybrid`; it
 * is not registered as a standalone method.
 *
 * A chunk whose parent lookup fails keeps its own chunk identity so a transient
 * failure cannot drop it. When every lookup fails the first failure is thrown:
 * a lexical ranking that never reaches window identity is not usable to fuse.
 */
export class LazykatokWindowBm25Method implements RetrievalMethod {
	private readonly chunks: LazykatokBm25Method;
	private readonly client: LazykatokParentClient;
	private readonly instanceId: string;

	constructor(options: LazykatokWindowBm25Options) {
		this.chunks = new LazykatokBm25Method(options);
		this.client = options.client;
		this.instanceId = options.instanceId;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			...this.chunks.describe(),
			description: "BM25 lexical retrieval over KakaoTalk chat chunks, lifted to their parent windows",
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const chunkHits = await this.chunks.retrieve(query, options);
		if (chunkHits.length === 0) return [];
		const chunkIds = [...new Set(chunkHits.map(chunkIdOf))];
		const parents = await resolveParents(this.client, chunkIds, options.signal);
		const failures = [...parents.values()].filter((parent) => !parent.ok);
		const firstFailure = failures[0];
		if (firstFailure !== undefined && !firstFailure.ok && failures.length === parents.size) {
			throw datasourceCliError(KAKAO_DATASOURCE_ID, "chunk parent", firstFailure);
		}
		const lifted = new Map<string, RetrievalResult>();
		for (const hit of chunkHits) {
			const parent = parents.get(chunkIdOf(hit));
			const window = parent?.ok ? parent.data[0] : undefined;
			const result = window === undefined ? hit : this.toWindowHit(hit, window);
			if (!lifted.has(result.id)) lifted.set(result.id, result);
		}
		return [...lifted.values()];
	}

	private toWindowHit(chunkHit: RetrievalResult, window: LazykatokChunk): RetrievalResult {
		return {
			id: `kakao:${this.instanceId}:${window.chunkId}`,
			content: window.content,
			source: lazykatokSourcePath(this.instanceId, window.chunkId),
			score: chunkHit.score,
			metadata: {
				...window.metadata,
				method: chunkHit.metadata?.method,
				datasourceId: KAKAO_DATASOURCE_ID,
				instanceId: this.instanceId,
				mode: "keyword",
				chunkId: window.chunkId,
				matchedChunkId: chunkIdOf(chunkHit),
			},
		};
	}
}

function chunkIdOf(hit: RetrievalResult): string {
	return String(hit.metadata?.chunkId);
}

async function resolveParents(
	client: LazykatokParentClient,
	chunkIds: readonly string[],
	signal: AbortSignal | undefined,
): Promise<Map<string, LazykatokParentResult>> {
	const parents = new Map<string, LazykatokParentResult>();
	let next = 0;
	const worker = async (): Promise<void> => {
		while (next < chunkIds.length) {
			signal?.throwIfAborted();
			const chunkId = chunkIds[next++]!;
			parents.set(chunkId, await client.parent(chunkId, signal));
		}
	};
	await Promise.all(Array.from({ length: Math.min(PARENT_LOOKUP_CONCURRENCY, chunkIds.length) }, worker));
	signal?.throwIfAborted();
	return parents;
}

async function retrieveLazykatok(
	client: LazykatokSearchClient,
	mode: LazykatokSearchMode,
	methodName: string,
	instanceId: string,
	query: string,
	options: RetrievalOptions,
): Promise<RetrievalResult[]> {
	const trimmed = query.trim();
	if (trimmed.length === 0) return [];
	const topK = options.topK ?? DEFAULT_TOP_K;
	const searchOptions: LazykatokSearchOptions = { topK, signal: options.signal };

	// lazykatok failures reach the caller verbatim: the pipeline reports this
	// datasource as unsearched with the CLI's own error text.
	const result: LazykatokSearchResult = await client.search(mode, trimmed, searchOptions);
	if (!result.ok) throw datasourceCliError(KAKAO_DATASOURCE_ID, `search --mode ${mode}`, result);

	const mapped: RetrievalResult[] = [];
	for (const hit of result.hits) {
		const source = lazykatokSource(instanceId, hit);
		mapped.push(toRetrievalResult(hit, source, methodName, instanceId, mode));
		if (mapped.length >= topK) break;
	}
	return mapped;
}

function lazykatokSource(instanceId: string, hit: LazykatokHit): string {
	return lazykatokSourcePath(instanceId, hit.chunkId);
}

function toRetrievalResult(
	hit: LazykatokHit,
	source: string,
	methodName: string,
	instanceId: string,
	mode: LazykatokSearchMode,
): RetrievalResult {
	return {
		id: `kakao:${instanceId}:${hit.chunkId}`,
		content: hit.content,
		source,
		score: hit.score,
		metadata: {
			...(hit.metadata ?? {}),
			method: methodName,
			datasourceId: KAKAO_DATASOURCE_ID,
			instanceId,
			mode,
			chunkId: hit.chunkId,
		},
	};
}
