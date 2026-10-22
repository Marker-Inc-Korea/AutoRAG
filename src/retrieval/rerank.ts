/**
 * Reranker seam — a post-retrieval reordering stage over merged evidence.
 *
 * The retrieval pipeline (registry → ParallelRetriever → filterDatasourceScope
 * → ResultMerger) produces a scored candidate list. A reranker reorders that
 * list with a dedicated relevance model. The default implementation talks to
 * OpenRouter's rerank router through the OpenRouter TypeScript SDK.
 *
 * Local-first is no longer a contract: a remote reranker is opt-in through
 * trusted config, and a local reranker can implement this same interface
 * without touching the pipeline. Nothing here silently falls back between
 * providers — a configured reranker that cannot run is reported, not replaced.
 */

import { API_PATH, PROVIDER_ID as DAZZI_PROVIDER_ID, resolveBaseUrl } from "../cloud/config.ts";
import { DAZZI_LOGIN_REQUIRED_MESSAGE, readDazziLoginToken } from "../cloud/credential.ts";
import type { RetrievalResult } from "./types.ts";

/** OpenRouter's production API root. */
const OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1";

/** Default rerank model routed through OpenRouter. */
export const DEFAULT_RERANK_MODEL = "voyageai/rerank-3-lite";
/** Default rerank provider id. */
export const DEFAULT_RERANK_PROVIDER = "openrouter";
/** Default environment variable holding the OpenRouter API key. */
export const DEFAULT_RERANK_API_KEY_ENV = "OPENROUTER_API_KEY";
/**
 * Default rerank model for the hosted Dazzi Models provider — the id the
 * server lists under `GET <root>/v1/models?operation=rerank`. Rerankers are not
 * chat LLMs, so the project's 2026+ model rule does not cover them.
 */
export const DEFAULT_DAZZI_RERANK_MODEL = "qwen/qwen3-reranker-8b";
/**
 * Default per-request rerank timeout. A reranker that never answers would stall
 * the fast answer. Measured against the hosted reranker, a 172-document pool
 * takes ~15s and a 300-document pool ~45s, so 60s clears the largest baseline
 * pool (3 x 100 candidates) with headroom.
 */
export const DEFAULT_RERANK_TIMEOUT_MS = 60_000;
/** Default number of merged chunks kept after reranking. */
export const DEFAULT_RERANK_TOP_N = 25;
/**
 * Longest document, in UTF-16 units, sent to a reranker. The hosted Dazzi
 * reranker rejects a document of 131,072 characters or more with HTTP 400 and
 * fails the whole request; 131,000 stays under that with a small margin.
 */
export const RERANK_MAX_DOCUMENT_CHARS = 131_000;
/** Rerank providers {@link createReranker} can build. */
export const SUPPORTED_RERANK_PROVIDERS: readonly string[] = [DEFAULT_RERANK_PROVIDER, DAZZI_PROVIDER_ID];

/** Default wire model for a rerank provider. */
export function defaultRerankModel(provider: string): string {
	return provider === DAZZI_PROVIDER_ID ? DEFAULT_DAZZI_RERANK_MODEL : DEFAULT_RERANK_MODEL;
}

/**
 * Default API-key environment variable for a rerank provider, or `undefined`
 * for Dazzi Models, which authenticates with the stored `/login` credential.
 */
export function defaultRerankApiKeyEnv(provider: string): string | undefined {
	return provider === DAZZI_PROVIDER_ID ? undefined : DEFAULT_RERANK_API_KEY_ENV;
}

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

/** Minimal rerank client shape, so tests can inject a fake without network access. */
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

/** Rerank response as {@link OpenRouterReranker} consumes it. */
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
	/** Provider id reported by `describe()` and stamped on metadata. @default DEFAULT_RERANK_PROVIDER */
	readonly provider?: string;
	/** Wire model id. @default DEFAULT_RERANK_MODEL */
	readonly model?: string;
	/**
	 * API key. When omitted, `openrouter` reads `apiKeyEnv` from the environment
	 * and `dazzi` uses the stored `/login` credential.
	 */
	readonly apiKey?: string;
	/** Environment variable holding the API key (`openrouter` only). @default DEFAULT_RERANK_API_KEY_ENV */
	readonly apiKeyEnv?: string;
	/** Pi agent home whose `auth.json` holds the Dazzi `/login` credential (`dazzi` only). Default: Pi's agent dir. */
	readonly agentDir?: string;
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
	return new OpenRouterReranker({ ...config, provider });
}

/** `<root>/v1`, tolerating an `AUTORAG_BASE_URL` that already carries the API path. */
function dazziApiBaseUrl(env: NodeJS.ProcessEnv): string {
	const root = resolveBaseUrl(env);
	return root.endsWith(API_PATH) ? root : `${root}${API_PATH}`;
}

/**
 * Rerank client over plain `fetch`. The OpenRouter SDK validates every result
 * against a schema that requires a `document` echo, which Dazzi does not send:
 * the SDK then rejects a successful HTTP 200. Only `index` and
 * `relevance_score` are read here, and a malformed or non-2xx response throws
 * with the status and body so the failure is reported, not hidden.
 */
function createFetchRerankClient(options: {
	readonly apiKey: string;
	readonly baseUrl: string;
	readonly timeoutMs: number;
}): RerankClient {
	const url = `${trimTrailingSlashes(options.baseUrl)}/rerank`;
	return {
		rerank: async ({ requestBody }, requestOptions) => {
			const { topN, ...rest } = requestBody;
			const timeout = AbortSignal.timeout(options.timeoutMs);
			const signal = AbortSignal.any([requestOptions?.signal, timeout].filter((entry) => entry !== undefined));
			const response = await fetch(url, {
				method: "POST",
				headers: { authorization: `Bearer ${options.apiKey}`, "content-type": "application/json" },
				body: JSON.stringify({ ...rest, ...(topN !== undefined ? { top_n: topN } : {}) }),
				signal,
			});
			const text = await response.text();
			if (!response.ok) throw new Error(`rerank request failed (HTTP ${response.status}): ${text.slice(0, 500)}`);
			let body: unknown;
			try {
				body = JSON.parse(text);
			} catch {
				throw new Error(`rerank response is not JSON: ${text.slice(0, 200)}`);
			}
			const results = readField(body, "results");
			if (!Array.isArray(results)) throw new Error(`rerank response has no results array: ${text.slice(0, 200)}`);
			const parsed = results.map((entry: unknown) => {
				const index = readField(entry, "index");
				const score = readField(entry, "relevance_score");
				if (typeof index !== "number" || !Number.isInteger(index) || typeof score !== "number") {
					throw new Error(`rerank response has a malformed result: ${JSON.stringify(entry).slice(0, 200)}`);
				}
				return { index, relevanceScore: score };
			});
			const model = readField(body, "model");
			const provider = readField(body, "provider");
			return {
				results: parsed,
				...(typeof model === "string" ? { model } : {}),
				...(typeof provider === "string" ? { provider } : {}),
			};
		},
	};
}

/** `value` without its trailing `/` characters, in one backwards scan. */
function trimTrailingSlashes(value: string): string {
	let end = value.length;
	while (end > 0 && value.charCodeAt(end - 1) === 47) end--;
	return value.slice(0, end);
}

/** `value[key]` when `value` is an object that has `key`, else `undefined`. */
function readField(value: unknown, key: string): unknown {
	return typeof value === "object" && value !== null && key in value ? Reflect.get(value, key) : undefined;
}

/**
 * Reranker backed by an OpenRouter-compatible rerank endpoint.
 *
 * OpenRouter is the default provider; the hosted Dazzi Models plan serves the
 * same wire shape at `<root>/v1/rerank`, so {@link createReranker} reuses this
 * client with that base URL and the stored login token. Documents are sent as their `content`;
 * the original `RetrievalResult` for each returned index is preserved and
 * re-scored with the provider's relevance score.
 */
export class OpenRouterReranker implements Reranker {
	private readonly model: string;
	private readonly provider: string;
	private readonly client: RerankClient | undefined;
	private readonly unavailableReason: string | undefined;

	constructor(options: OpenRouterRerankerOptions = {}) {
		this.provider = options.provider ?? DEFAULT_RERANK_PROVIDER;
		this.model = options.model ?? defaultRerankModel(this.provider);
		if (options.client !== undefined) {
			this.client = options.client;
			return;
		}
		const env = options.env ?? process.env;
		const isDazzi = this.provider === DAZZI_PROVIDER_ID;
		const apiKeyEnv = options.apiKeyEnv ?? defaultRerankApiKeyEnv(this.provider);
		const apiKey =
			options.apiKey ??
			(isDazzi ? readDazziLoginToken(options.agentDir) : apiKeyEnv === undefined ? undefined : env[apiKeyEnv]);
		if (apiKey === undefined || apiKey.length === 0) {
			this.unavailableReason = isDazzi
				? DAZZI_LOGIN_REQUIRED_MESSAGE
				: `no API key: set ${apiKeyEnv} or configure rerank.apiKeyEnv`;
			return;
		}
		const baseUrl = options.baseUrl ?? (isDazzi ? dazziApiBaseUrl(env) : OPENROUTER_BASE_URL);
		this.client = createFetchRerankClient({
			apiKey,
			baseUrl,
			timeoutMs: options.timeoutMs ?? DEFAULT_RERANK_TIMEOUT_MS,
		});
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

		// The provider fails the whole request with HTTP 400 for an empty document
		// and for one above its input limit, so blank (empty or whitespace-only)
		// content is never sent and long content is cut to RERANK_MAX_DOCUMENT_CHARS
		// on the wire only; the returned results keep their full content.
		// `sent[i]` is the index in `results` of the i-th document on the wire.
		const sent: number[] = [];
		const documents: string[] = [];
		for (let index = 0; index < results.length; index += 1) {
			const content = (results[index] as RetrievalResult).content;
			if (content.trim() === "") continue;
			sent.push(index);
			if (content.length <= RERANK_MAX_DOCUMENT_CHARS) {
				documents.push(content);
				continue;
			}
			// Do not leave half of a surrogate pair at the cut.
			const lastKept = content.charCodeAt(RERANK_MAX_DOCUMENT_CHARS - 1);
			const cut =
				lastKept >= 0xd800 && lastKept <= 0xdbff ? RERANK_MAX_DOCUMENT_CHARS - 1 : RERANK_MAX_DOCUMENT_CHARS;
			documents.push(content.slice(0, cut));
		}
		if (documents.length === 0) {
			return options.topN === undefined ? [...results] : results.slice(0, options.topN);
		}

		const requestBody: {
			model: string;
			query: string;
			documents: string[];
			topN?: number;
		} = {
			model: this.model,
			query,
			documents,
		};
		if (options.topN !== undefined) requestBody.topN = options.topN;

		const response = await this.client.rerank({ requestBody }, { signal: options.signal });
		const ordered = Array.isArray(response?.results) ? response.results : [];
		const seen = new Set<number>();
		const reranked: RetrievalResult[] = [];
		for (const entry of ordered) {
			const originalIndex = sent[entry.index];
			if (originalIndex === undefined || seen.has(originalIndex)) continue;
			const original = results[originalIndex] as RetrievalResult;
			seen.add(originalIndex);
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
