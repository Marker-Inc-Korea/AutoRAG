import type { EmbedderEndpoint, ManagedEmbedder } from "../embedding/embedder.ts";

/**
 * Pointing an external CLI that embeds for itself (mailcrawl, lazykatok) at the same backend as the rest of
 * AutoRAG. The CLI keeps ownership of its archive and vector store; AutoRAG only tells it where the vectors
 * come from, through the environment variables the CLI documents. A remote key travels as the value of the
 * variable the operator's `minSync.embedder.apiKeyEnv` names, to that one child only.
 */

/** An endpoint whose vector length is known; the CLIs require it up front. */
export interface ResolvedEmbeddingEndpoint extends EmbedderEndpoint {
	readonly dimension: number;
}

/**
 * `index` is `refresh` work, where the shared embedder may start the `minsync serve` daemon. `query` is a
 * search, where nothing is ever started: an absent daemon leaves the CLI on its own configuration.
 */
export type EmbeddingPurpose = "index" | "query";

export type EmbeddingEndpointSource = (purpose: EmbeddingPurpose) => Promise<ResolvedEmbeddingEndpoint | undefined>;

const LOOPBACK_HOSTS: Record<string, true> = { "127.0.0.1": true, localhost: true, "[::1]": true, "::1": true };

function isLoopback(baseUrl: string): boolean {
	try {
		return LOOPBACK_HOSTS[new URL(baseUrl).hostname] === true;
	} catch {
		return false;
	}
}

/** The bearer token a remote endpoint needs, or a clear error: a CLI left keyless would only answer 401. */
function apiKeyValue(endpoint: EmbedderEndpoint, parentEnv: NodeJS.ProcessEnv): string | undefined {
	if (endpoint.apiKeyEnv === undefined) return undefined;
	const key = parentEnv[endpoint.apiKeyEnv];
	if (typeof key !== "string" || key.length === 0) {
		throw new Error(`embedding API key is not set: environment variable ${endpoint.apiKeyEnv} is empty or missing`);
	}
	return key;
}

/**
 * The shared embedder as an endpoint source. Probes only, except that `index` first lets the embedder start
 * its daemon. `undefined` means the embedder would run in-process, which an external CLI cannot reach.
 */
export function embeddingEndpointSource(embedder: ManagedEmbedder): EmbeddingEndpointSource {
	return async (purpose) => {
		if (purpose === "index") await embedder.ensureReady();
		const endpoint = await embedder.describeEndpoint();
		if (endpoint === undefined) return undefined;
		const dimension = endpoint.dimension ?? (await embedder.identity()).dimension;
		return { ...endpoint, dimension };
	};
}

/**
 * mailcrawl's `openai` and `tei` providers (0.2.2+). Identity is provider + model + dimension, so moving a
 * server keeps its vectors, while switching backend rebuilds the index on the next `mailcrawl index`.
 * mailcrawl accepts a non-loopback endpoint only over https with an explicit opt-in, which is set here
 * because the operator configured that remote themselves.
 */
export function mailcrawlEmbeddingEnv(
	endpoint: ResolvedEmbeddingEndpoint,
	parentEnv: NodeJS.ProcessEnv = process.env,
): Record<string, string> {
	const env: Record<string, string> = {
		MAILCRAWL_EMBEDDER_PROVIDER: endpoint.protocol,
		MAILCRAWL_EMBED_URL: endpoint.baseUrl,
		MAILCRAWL_EMBED_MODEL: endpoint.model,
		MAILCRAWL_EMBED_DIM: String(endpoint.dimension),
	};
	if (endpoint.backend === "remote" && !isLoopback(endpoint.baseUrl)) env.MAILCRAWL_EMBED_ALLOW_REMOTE = "1";
	const key = apiKeyValue(endpoint, parentEnv);
	if (key !== undefined && endpoint.apiKeyEnv !== undefined) {
		env.MAILCRAWL_EMBED_API_KEY_ENV = endpoint.apiKeyEnv;
		env[endpoint.apiKeyEnv] = key;
	}
	return env;
}

/**
 * lazykatok's OpenAI-compatible embedder (0.3.3+). It speaks `POST /v1/embeddings` only, so a TEI endpoint
 * cannot be offered and lazykatok keeps its own embedder, as discrawl does.
 */
export function lazykatokEmbeddingEnv(
	endpoint: ResolvedEmbeddingEndpoint,
	parentEnv: NodeJS.ProcessEnv = process.env,
): Record<string, string> | undefined {
	if (endpoint.protocol !== "openai") return undefined;
	const env: Record<string, string> = {
		KATOK_EMBEDDER: "openai-compatible",
		KATOK_EMBED_URL: endpoint.baseUrl,
		KATOK_EMBED_MODEL: endpoint.model,
		KATOK_EMBED_DIM: String(endpoint.dimension),
	};
	const key = apiKeyValue(endpoint, parentEnv);
	if (key !== undefined) env.KATOK_EMBED_API_KEY = key;
	return env;
}
