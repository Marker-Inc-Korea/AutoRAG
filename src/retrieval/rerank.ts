/**
 * Reranker seam — a post-retrieval reordering stage over merged evidence.
 *
 * The retrieval pipeline (registry → ParallelRetriever → DatasourceResultFilter
 * → ResultMerger) produces a scored candidate list. A reranker reorders that
 * list with a dedicated relevance model. The default implementation talks to
 * OpenRouter's rerank router through the OpenRouter TypeScript SDK.
 *
 * Local-first is no longer a contract: a remote reranker is opt-in through
 * trusted config, and a local reranker can implement this same interface
 * without touching the pipeline. Nothing here silently falls back between
 * providers — a configured reranker that cannot run is reported, not replaced.
 */

import { OpenRouter } from "@openrouter/sdk";
import type { RetrievalResult } from "./types.ts";

/** Default rerank model routed through OpenRouter. */
export const DEFAULT_RERANK_MODEL = "voyageai/rerank-3-lite";
/** Default rerank provider id. */
export const DEFAULT_RERANK_PROVIDER = "openrouter";
/** Default environment variable holding the OpenRouter API key. */
export const DEFAULT_RERANK_API_KEY_ENV = "OPENROUTER_API_KEY";
/** Default number of merged chunks kept after reranking. */
export const DEFAULT_RERANK_TOP_N = 25;
/** Rerank providers {@link createReranker} can build. */
export const SUPPORTED_RERANK_PROVIDERS: readonly string[] = [DEFAULT_RERANK_PROVIDER];

/** Static description of a reranker, used for diagnostics and inspection. */
export interface RerankerDescriptor {
	/** Stable label (e.g. `openrouter`). */
	readonly name: string;
	/** Provider id (e.g. `openrouter`). */
	readonly provider: string;
	/** Wire model id (e.g. `voyageai/rerank-3`). */
	readonly model: string;
	/** Whether the reranker has everything it needs to run (credentials, config). */
	readonly available: boolean;
	/** Why the reranker is unavailable, when `available` is false. */
	readonly reason?: string;
}

/** Per-call rerank options. */
export interface RerankOptions {
	/** Return only the top N results. Omitted ⇒ every distinct result is returned, reordered. */
	readonly topN?: number;
	/** Abort the in-flight rerank request. */
	readonly signal?: AbortSignal;
}

/** A post-retrieval reordering stage. */
export interface Reranker {
	describe(): RerankerDescriptor;
	/**
	 * Reorder `results` by relevance to `query`. Returns a new array; the input
	 * is not mutated. Implementations must preserve each result's `source`/`id`
	 * identity so evidence stays traceable.
	 */
	rerank(query: string, results: readonly RetrievalResult[], options?: RerankOptions): Promise<RetrievalResult[]>;
}

/** Minimal rerank client shape, so tests can inject a fake without the SDK. */
export interface RerankClient {
	rerank(
		request: {
			requestBody: {
				model: string;
				query: string;
				documents: string[];
				topN?: number;
			};
		},
		options?: { signal?: AbortSignal },
	): Promise<RerankResponse>;
}

/** Response body returned by OpenRouter's rerank router. */
export interface RerankResponse {
	readonly results: ReadonlyArray<{
		readonly index: number;
		readonly relevanceScore: number;
	}>;
	readonly model?: string;
	readonly provider?: string;
}

/** Options for {@link OpenRouterReranker}. */
export interface OpenRouterRerankerOptions {
	/** Wire model id. @default DEFAULT_RERANK_MODEL */
	readonly model?: string;
	/** API key. When omitted, read from `apiKeyEnv` in the environment. */
	readonly apiKey?: string;
	/** Environment variable holding the API key. @default DEFAULT_RERANK_API_KEY_ENV */
	readonly apiKeyEnv?: string;
	/** Override the OpenRouter base URL (e.g. a self-hosted or gateway endpoint). */
	readonly baseUrl?: string;
	/** Per-request timeout in milliseconds. */
	readonly timeoutMs?: number;
	/** Environment to read the API key from. Defaults to `process.env`; a test seam. */
	readonly env?: NodeJS.ProcessEnv;
	/** Pre-built client, bypassing SDK construction. A test seam. */
	readonly client?: RerankClient;
}

/** Options for {@link createReranker}: provider plus the OpenRouter reranker options. */
export interface CreateRerankerOptions extends OpenRouterRerankerOptions {
	/** Provider id. @default DEFAULT_RERANK_PROVIDER */
	readonly provider?: string;
}

/**
 * Build the configured reranker. Returns `undefined` when reranking is disabled
 * or absent. Throws for a provider AutoRAG cannot build (an unsupported
 * `provider` id is a configuration error, not a silent no-op).
 */
export function createReranker(config: CreateRerankerOptions | false | undefined): Reranker | undefined {
	if (config === false || config === undefined) return undefined;
	const provider = config.provider ?? DEFAULT_RERANK_PROVIDER;
	if (!SUPPORTED_RERANK_PROVIDERS.includes(provider)) {
		throw new Error(
			`Unsupported rerank provider "${provider}"; supported providers: ${SUPPORTED_RERANK_PROVIDERS.join(", ")}`,
		);
	}
	return new OpenRouterReranker(config);
}

/**
 * Reranker backed by OpenRouter's rerank router.
 *
 * Documents are sent as their `content`; the original `RetrievalResult` for each
 * returned index is preserved and re-scored with the provider's relevance score.
 */
export class OpenRouterReranker implements Reranker {
	private readonly model: string;
	private readonly provider = DEFAULT_RERANK_PROVIDER;
	private readonly client: RerankClient | undefined;
	private readonly unavailableReason: string | undefined;

	constructor(options: OpenRouterRerankerOptions = {}) {
		this.model = options.model ?? DEFAULT_RERANK_MODEL;
		if (options.client !== undefined) {
			this.client = options.client;
			return;
		}
		const apiKeyEnv = options.apiKeyEnv ?? DEFAULT_RERANK_API_KEY_ENV;
		const env = options.env ?? process.env;
		const apiKey = options.apiKey ?? env[apiKeyEnv];
		if (apiKey === undefined || apiKey.length === 0) {
			this.unavailableReason = `no API key: set ${apiKeyEnv} or configure rerank.apiKeyEnv`;
			return;
		}
		const sdkOptions: ConstructorParameters<typeof OpenRouter>[0] = { apiKey };
		if (options.baseUrl !== undefined) sdkOptions.serverURL = options.baseUrl;
		if (options.timeoutMs !== undefined) sdkOptions.timeoutMs = options.timeoutMs;
		const sdk = new OpenRouter(sdkOptions);
		this.client = {
			rerank: async (request, requestOptions) => {
				const response = await sdk.rerank.rerank(request, requestOptions);
				if (typeof response === "string") throw new Error(response);
				return response;
			},
		};
	}

	describe(): RerankerDescriptor {
		return {
			name: this.provider,
			provider: this.provider,
			model: this.model,
			available: this.client !== undefined,
			...(this.unavailableReason !== undefined ? { reason: this.unavailableReason } : {}),
		};
	}

	async rerank(
		query: string,
		results: readonly RetrievalResult[],
		options: RerankOptions = {},
	): Promise<RetrievalResult[]> {
		if (this.client === undefined) {
			throw new Error(this.unavailableReason ?? "reranker is unavailable");
		}
		if (results.length === 0) return [];

		const requestBody: {
			model: string;
			query: string;
			documents: string[];
			topN?: number;
		} = {
			model: this.model,
			query,
			documents: results.map((result) => result.content),
		};
		if (options.topN !== undefined) requestBody.topN = options.topN;

		const response = await this.client.rerank({ requestBody }, { signal: options.signal });
		const ordered = Array.isArray(response?.results) ? response.results : [];
		const seen = new Set<number>();
		const reranked: RetrievalResult[] = [];
		for (const entry of ordered) {
			const original = results[entry.index];
			if (original === undefined || seen.has(entry.index)) continue;
			seen.add(entry.index);
			reranked.push({
				...original,
				score: entry.relevanceScore,
				metadata: {
					...original.metadata,
					rerankProvider: this.provider,
					rerankModel: response.model ?? this.model,
					rerankScore: entry.relevanceScore,
				},
			});
		}
		// Preserve any results the provider did not score (e.g. truncated upstream),
		// after the ranked ones, so no evidence is silently dropped when topN is unset.
		if (options.topN === undefined) {
			for (let index = 0; index < results.length; index += 1) {
				if (seen.has(index)) continue;
				reranked.push(results[index] as RetrievalResult);
			}
		}
		return reranked;
	}
}
