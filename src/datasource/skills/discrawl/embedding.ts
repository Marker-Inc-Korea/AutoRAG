import type { EmbedderEndpoint } from "../../../embedding/embedder.ts";
import type { DiscrawlEmbeddingRuntime } from "./types.ts";

/**
 * discrawl keeps `openai_compatible` embeddings off ("requires API key env ...") unless the variable its
 * config names is non-empty, while `minsync serve` has no authentication and ignores the value. For the
 * local daemon AutoRAG therefore names a variable of its own and supplies this fixed, non-secret placeholder
 * to the discrawl child only. It is never a credential and never read from the operator's environment.
 */
export const LOCAL_EMBEDDING_KEY_ENV = "AUTORAG_LOCAL_EMBEDDING_KEY";
export const LOCAL_EMBEDDING_KEY_PLACEHOLDER = "local";

/**
 * What discrawl's own `[search.embeddings]` section should say so that it embeds through the same backend
 * as the rest of AutoRAG. discrawl speaks OpenAI `POST {base_url}/embeddings` only (its provider name is
 * `openai_compatible`, with an underscore), so a TEI endpoint cannot be offered and discrawl keeps its own
 * configuration. A remote key travels as the name of an environment variable, never as a value.
 */
export function discrawlEmbeddingRuntimeFor(
	endpoint: EmbedderEndpoint | undefined,
): DiscrawlEmbeddingRuntime | undefined {
	if (endpoint === undefined || endpoint.protocol !== "openai") return undefined;
	const root = endpoint.baseUrl.replace(/\/+$/, "").replace(/\/v1$/, "");
	const local = endpoint.backend === "daemon";
	return {
		provider: "openai_compatible",
		model: endpoint.model,
		baseUrl: `${root}/v1`,
		...(endpoint.dimension === undefined ? {} : { dimensions: endpoint.dimension }),
		...(local
			? { apiKeyEnv: LOCAL_EMBEDDING_KEY_ENV, apiKeyValue: LOCAL_EMBEDDING_KEY_PLACEHOLDER }
			: endpoint.apiKeyEnv === undefined
				? {}
				: { apiKeyEnv: endpoint.apiKeyEnv }),
	};
}
