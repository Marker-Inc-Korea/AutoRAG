import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";
import type { QmdSearchHit, QmdSearchMode, QmdSearchOptions, QmdSearchResult } from "./types.ts";

export interface ObsidianSearchClient {
	search(mode: QmdSearchMode, query: string, options?: QmdSearchOptions): Promise<QmdSearchResult>;
}

export interface ObsidianMethodOptions {
	readonly client: ObsidianSearchClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

const OBSIDIAN_DATASOURCE_ID = "obsidian";
const DEFAULT_OBSIDIAN_TAGS = ["obsidian", "notes"] as const;
const DEFAULT_TOP_K = 20;

export class ObsidianBm25Method implements RetrievalMethod {
	private readonly client: ObsidianSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: ObsidianMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_OBSIDIAN_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "obsidian-bm25",
			type: "bm25",
			description: "BM25 lexical retrieval over an Obsidian vault via the external qmd CLI",
			status: "active",
			capabilities: ["lexical", "keyword-mode", "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveObsidian(this.client, "search", "obsidian-bm25", this.instanceId, query, options);
	}
}

export class ObsidianSemanticMethod implements RetrievalMethod {
	private readonly client: ObsidianSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: ObsidianMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_OBSIDIAN_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "obsidian-semantic",
			type: "vector",
			description: "Semantic vector retrieval over an Obsidian vault via the external qmd CLI",
			status: "active",
			capabilities: ["semantic", "vector-mode", "scoped", "external-cli", "path-opaque-sources"],
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			tags: this.tags,
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveObsidian(this.client, "vsearch", "obsidian-semantic", this.instanceId, query, options);
	}
}

async function retrieveObsidian(
	client: ObsidianSearchClient,
	mode: QmdSearchMode,
	methodName: string,
	instanceId: string,
	query: string,
	options: RetrievalOptions,
): Promise<RetrievalResult[]> {
	const trimmed = query.trim();
	if (trimmed.length === 0) return [];
	const topK = options.topK ?? DEFAULT_TOP_K;
	const searchOptions: QmdSearchOptions = { topK, signal: options.signal };

	let result: QmdSearchResult;
	try {
		result = await client.search(mode, trimmed, searchOptions);
	} catch {
		return [];
	}
	if (!result.ok) return [];

	const mapped: RetrievalResult[] = [];
	for (const hit of result.hits) {
		const source = datasourceSourcePath(OBSIDIAN_DATASOURCE_ID, instanceId, hit.chunkId);
		if (!matchesScope(source, options.scope, options.allowedScopes)) continue;
		mapped.push(toRetrievalResult(hit, source, methodName, instanceId, mode));
		if (mapped.length >= topK) break;
	}
	return mapped;
}

function matchesScope(
	source: string,
	scope: string | undefined,
	allowedScopes: readonly string[] | undefined,
): boolean {
	if (!matchesDatasourceScope(source, scope)) return false;
	if (allowedScopes === undefined || allowedScopes.length === 0) return true;
	return allowedScopes.some((entry) => matchesDatasourceScope(source, entry));
}

function toRetrievalResult(
	hit: QmdSearchHit,
	source: string,
	methodName: string,
	instanceId: string,
	mode: QmdSearchMode,
): RetrievalResult {
	return {
		id: `obsidian:${instanceId}:${hit.chunkId}`,
		content: hit.content,
		source,
		score: hit.score,
		metadata: {
			...(hit.metadata ?? {}),
			method: methodName,
			datasourceId: OBSIDIAN_DATASOURCE_ID,
			instanceId,
			mode,
			chunkId: hit.chunkId,
			...(hit.title !== undefined ? { title: hit.title } : {}),
			...(hit.file !== undefined ? { path: hit.file } : {}),
			...(hit.docid !== undefined ? { docid: hit.docid } : {}),
		},
	};
}
