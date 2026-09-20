import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import { datasourceCliError } from "../../errors.ts";
import { matchesDatasourceScope } from "../../scope.ts";
import { discrawlSourcePath } from "./paths.ts";
import type { DiscrawlSearchHit, DiscrawlSearchMode, DiscrawlSearchOptions, DiscrawlSearchResult } from "./types.ts";

/**
 * Narrow client surface required by the Discord retrieval methods. The real
 * {@link DiscrawlClient} satisfies this structurally; tests may stub it.
 */
export interface DiscrawlSearchClient {
	search(mode: DiscrawlSearchMode, query: string, options?: DiscrawlSearchOptions): Promise<DiscrawlSearchResult>;
}

export interface DiscrawlMethodOptions {
	readonly client: DiscrawlSearchClient;
	readonly instanceId: string;
	readonly tags?: readonly string[];
}

const DISCORD_DATASOURCE_ID = "discord";
const DEFAULT_DISCORD_TAGS = ["discord", "chat", "pii"] as const;
const DEFAULT_TOP_K = 20;

/**
 * Lexical (SQLite FTS5) retrieval over a Discord archive via the external
 * `discrawl` CLI. All client failures collapse to an empty result set;
 * retrieval never throws.
 */
export class DiscrawlFtsMethod implements RetrievalMethod {
	private readonly client: DiscrawlSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: DiscrawlMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_DISCORD_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "discord-fts",
			type: "bm25",
			description: "SQLite FTS5 lexical retrieval over archived Discord messages via the external discrawl CLI",
			status: "active",
			capabilities: ["lexical", "fts5", "scoped", "external-cli"],
			datasourceId: DISCORD_DATASOURCE_ID,
			tags: [...this.tags],
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveDiscrawl(this.client, "fts", "discord-fts", this.instanceId, query, options);
	}
}

/**
 * Semantic (vector) retrieval over a Discord archive via the external
 * `discrawl` CLI. Requires a configured embedding provider and a completed
 * `discrawl embed` pass; when either is missing the CLI exits non-zero and
 * this method returns no results rather than throwing.
 */
export class DiscrawlSemanticMethod implements RetrievalMethod {
	private readonly client: DiscrawlSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: DiscrawlMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_DISCORD_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "discord-semantic",
			type: "vector",
			description: "Semantic vector retrieval over archived Discord messages via the external discrawl CLI",
			status: "active",
			capabilities: ["semantic", "vector-mode", "scoped", "external-cli"],
			datasourceId: DISCORD_DATASOURCE_ID,
			tags: [...this.tags],
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveDiscrawl(this.client, "semantic", "discord-semantic", this.instanceId, query, options);
	}
}

/**
 * Hybrid retrieval (FTS5 + semantic, deduplicated by message id). This is the
 * preferred Discord method: discrawl's FTS index welds words across newlines
 * into one token, so semantic recall is what covers terms FTS cannot reach.
 * See AutoRAG issue #1413.
 */
export class DiscrawlHybridMethod implements RetrievalMethod {
	private readonly client: DiscrawlSearchClient;
	private readonly instanceId: string;
	private readonly tags: readonly string[];

	constructor(options: DiscrawlMethodOptions) {
		this.client = options.client;
		this.instanceId = options.instanceId;
		this.tags = options.tags ?? DEFAULT_DISCORD_TAGS;
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "discord-hybrid",
			type: "hybrid",
			description:
				"Hybrid FTS5 + semantic retrieval over archived Discord messages via the external discrawl CLI; falls back to FTS when embeddings are unavailable",
			status: "active",
			capabilities: ["lexical", "semantic", "hybrid", "scoped", "external-cli"],
			datasourceId: DISCORD_DATASOURCE_ID,
			tags: [...this.tags],
		};
	}

	retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return retrieveDiscrawl(this.client, "hybrid", "discord-hybrid", this.instanceId, query, options);
	}
}

async function retrieveDiscrawl(
	client: DiscrawlSearchClient,
	mode: DiscrawlSearchMode,
	methodName: string,
	instanceId: string,
	query: string,
	options: RetrievalOptions,
): Promise<RetrievalResult[]> {
	const trimmed = query.trim();
	if (trimmed.length === 0) return [];
	const topK = options.topK ?? DEFAULT_TOP_K;

	// discrawl failures reach the caller verbatim: the pipeline reports Discord as
	// unsearched with the CLI's own error text. Hybrid still degrades to FTS first,
	// and only a failing fallback surfaces the error.
	let result: DiscrawlSearchResult = await client.search(mode, trimmed, { ...options, topK });
	if (!result.ok) {
		if (mode !== "hybrid") throw datasourceCliError(DISCORD_DATASOURCE_ID, `search --mode ${mode}`, result);
		result = await client.search("fts", trimmed, { ...options, topK });
		if (!result.ok) throw datasourceCliError(DISCORD_DATASOURCE_ID, "search --mode fts", result);
		return mapDiscrawlHits(result.hits, "discord-hybrid", instanceId, "fts", options, "semantic-unavailable").slice(
			0,
			topK,
		);
	}

	return mapDiscrawlHits(result.hits, methodName, instanceId, mode, options).slice(0, topK);
}

function mapDiscrawlHits(
	hits: readonly DiscrawlSearchHit[],
	methodName: string,
	instanceId: string,
	mode: DiscrawlSearchMode,
	options: RetrievalOptions,
	diagnostic?: string,
): RetrievalResult[] {
	const mapped: RetrievalResult[] = [];
	for (const hit of hits) {
		const source = discrawlSourcePath(instanceId, hit.messageId);
		if (!matchesScope(source, options.scope, options.allowedScopes)) continue;
		mapped.push(toRetrievalResult(hit, source, methodName, instanceId, mode, diagnostic));
		if (mapped.length >= (options.topK ?? DEFAULT_TOP_K)) break;
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
	hit: DiscrawlSearchHit,
	source: string,
	methodName: string,
	instanceId: string,
	mode: DiscrawlSearchMode,
	diagnostic?: string,
): RetrievalResult {
	return {
		id: `discord:${instanceId}:${hit.messageId}`,
		content: hit.content,
		source,
		score: hit.score,
		metadata: {
			...(hit.metadata ?? {}),
			method: methodName,
			datasourceId: DISCORD_DATASOURCE_ID,
			instanceId,
			mode,
			...(diagnostic !== undefined ? { diagnostic } : {}),
			messageId: hit.messageId,
			...(hit.channelName !== undefined ? { channelName: hit.channelName } : {}),
			...(hit.channelId !== undefined ? { channelId: hit.channelId } : {}),
			...(hit.guildName !== undefined ? { guildName: hit.guildName } : {}),
			...(hit.guildId !== undefined ? { guildId: hit.guildId } : {}),
			...(hit.authorName !== undefined ? { authorName: hit.authorName } : {}),
			...(hit.timestamp !== undefined ? { timestamp: hit.timestamp } : {}),
		},
	};
}
