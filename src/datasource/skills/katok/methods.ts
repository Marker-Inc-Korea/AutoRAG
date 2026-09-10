import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import type { KatokHit, KatokSearchMode, KatokSearchOptions, KatokSearchResult } from "./types.ts";
import { katokSourcePath } from "./paths.ts";

/**
 * Narrow client surface required by the KakaoTalk retrieval methods.
 * The real {@link KatokClient} satisfies this structurally; tests may stub it.
 */
export interface KatokSearchClient {
	search(mode: KatokSearchMode, query: string, options?: KatokSearchOptions): Promise<KatokSearchResult>;
}

export interface KatokMethodOptions {
	readonly client: KatokSearchClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

const KAKAO_DATASOURCE_ID = "kakao";
const DEFAULT_KAKAO_TAGS = ["kakaotalk", "personal", "pii"] as const;
const DEFAULT_TOP_K = 20;

/**
 * Lexical (BM25) retrieval over a KakaoTalk datasource via the external
 * `katok` CLI in keyword mode. Katok does not expose source scopes; access is
 * controlled at the datasource/tag level. All client failures collapse to an
 * empty result set; retrieval never throws.
 */
export class KatokBm25Method implements RetrievalMethod {
	private readonly client: KatokSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: KatokMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_KAKAO_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "kakao-bm25",
			type: "bm25",
			description: "BM25 lexical retrieval over KakaoTalk chat chunks via the external katok CLI",
			status: "active",
			capabilities: ["lexical", "keyword-mode", "external-cli", "path-opaque-sources"],
			datasourceId: KAKAO_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveKatok(this.client, "keyword", "kakao-bm25", this.instanceId, query, options);
	}
}

/**
 * Semantic (vector) retrieval over a KakaoTalk datasource via the external
 * `katok` CLI in semantic mode. Katok does not expose source scopes; access is
 * controlled at the datasource/tag level. All client failures collapse to an
 * empty result set; retrieval never throws.
 */
export class KatokSemanticMethod implements RetrievalMethod {
	private readonly client: KatokSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: KatokMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_KAKAO_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "kakao-semantic",
			type: "vector",
			description: "Semantic vector retrieval over KakaoTalk chat chunks via the external katok CLI",
			status: "active",
			capabilities: ["semantic", "vector-mode", "external-cli", "path-opaque-sources"],
			datasourceId: KAKAO_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveKatok(this.client, "semantic", "kakao-semantic", this.instanceId, query, options);
	}
}

async function retrieveKatok(
	client: KatokSearchClient,
	mode: KatokSearchMode,
	methodName: string,
	instanceId: string,
	query: string,
	options: RetrievalOptions,
): Promise<RetrievalResult[]> {
	const trimmed = query.trim();
	if (trimmed.length === 0) return [];
	const topK = options.topK ?? DEFAULT_TOP_K;
	const searchOptions: KatokSearchOptions = { topK, signal: options.signal };

	let result: KatokSearchResult;
	try {
		result = await client.search(mode, trimmed, searchOptions);
	} catch {
		return [];
	}
	if (!result.ok) return [];

	const mapped: RetrievalResult[] = [];
	for (const hit of result.hits) {
		const source = katokSource(instanceId, hit);
		mapped.push(toRetrievalResult(hit, source, methodName, instanceId, mode));
		if (mapped.length >= topK) break;
	}
	return mapped;
}

function katokSource(instanceId: string, hit: KatokHit): string {
	return katokSourcePath(instanceId, hit.chunkId);
}

function toRetrievalResult(
	hit: KatokHit,
	source: string,
	methodName: string,
	instanceId: string,
	mode: KatokSearchMode,
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
