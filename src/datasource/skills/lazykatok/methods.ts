import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { lazykatokSourcePath } from "./paths.ts";
import type { LazykatokHit, LazykatokSearchMode, LazykatokSearchOptions, LazykatokSearchResult } from "./types.ts";

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
