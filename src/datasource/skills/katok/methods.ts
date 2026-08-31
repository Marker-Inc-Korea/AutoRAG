import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { matchesDatasourceScope } from "../../scope.ts";
import type { KatokHit, KatokSearchMode, KatokSearchOptions, KatokSearchResult } from "./types.ts";

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
 * `katok` CLI in keyword mode. All client failures collapse to an empty
 * result set; retrieval never throws.
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
			capabilities: ["lexical", "keyword-mode", "scoped", "external-cli", "path-opaque-sources"],
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
 * `katok` CLI in semantic mode. All client failures collapse to an empty
 * result set; retrieval never throws.
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
			capabilities: ["semantic", "vector-mode", "scoped", "external-cli", "path-opaque-sources"],
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

	const scope = options.scope;
	const allowedScopes = options.allowedScopes;
	if (!matchesScope("", scope, allowedScopes, instanceId)) return [];
	const mapped: RetrievalResult[] = [];
	for (const hit of result.hits) {
		const source = katokSource(instanceId, hit);
		mapped.push(toRetrievalResult(hit, source, methodName, instanceId, mode));
		if (mapped.length >= topK) break;
	}
	return mapped;
}

/**
 * Human-readable kakao source identity. Starts with the `kakao:` scheme so
 * the agent can never mistake it for an OS file path, and carries the chat
 * name and sender when katok provides them (e.g.
 * `kakao:오픈소스 개발과제/류동현투이컨설팅/chunk_58b3`).
 */
function katokSource(instanceId: string, hit: KatokHit): string {
	const chatName = typeof hit.metadata?.chatName === "string" ? hit.metadata.chatName : undefined;
	const sender = typeof hit.metadata?.senderNickname === "string" ? hit.metadata.senderNickname : undefined;
	const room = chatName ?? instanceId;
	const segments = [room, sender, hit.chunkId].filter((segment) => segment !== undefined && segment.length > 0);
	return `kakao:${segments.join("/")}`;
}

/**
 * Scope gating stays on the legacy virtual form (`/kakao/<instance>`) that
 * callers pass in, independent of the human-readable result source. Scope
 * can only narrow to this datasource's own instance tree.
 */
function matchesScope(
	source: string,
	scope: string | undefined,
	allowedScopes: readonly string[] | undefined,
	instanceId: string,
): boolean {
	const virtualSource = `/kakao/${instanceId}`;
	if (!matchesDatasourceScope(virtualSource, scope)) return false;
	if (allowedScopes === undefined || allowedScopes.length === 0) return true;
	return allowedScopes.some((entry) => matchesDatasourceScope(virtualSource, entry));
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
