import { stripNativeEmbedderPrefix } from "../minsync/embedder-config.ts";
import type { MinSyncEmbedderConfig } from "../minsync/types.ts";
import { DEFAULT_DAEMON_MODEL } from "./daemon.ts";

/**
 * Qwen3-Embedding is instruction-aware: the query side is `Instruct: {task}\nQuery:{query}` and documents are
 * embedded raw (model card `get_detailed_instruct`; there is no space after `Query:`). The task stays in
 * English even for non-English queries, as the card recommends.
 */
export const QWEN3_QUERY_PREFIX =
	"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

/** EmbeddingGemma 2 retrieval prompts (model card). */
export const EMBEDDINGGEMMA2_QUERY_PREFIX = "task: search result | query: ";
export const EMBEDDINGGEMMA2_PASSAGE_PREFIX = "title: none | text: ";

/** Text prepended to an embedding input, by what the input is for. */
export interface EmbeddingPrefixes {
	readonly query: string;
	readonly passage: string;
}

/** True when `minSync.embedder` selects MinSync's own local Qwen3 (no id, or `native:` Qwen3; never an endpoint). */
function isLocalQwen3(embedder: MinSyncEmbedderConfig | undefined): boolean {
	if (embedder?.baseUrl !== undefined) return false;
	const id = embedder?.id;
	if (id === undefined) return true;
	return id.startsWith("native:") && id.slice("native:".length).toLowerCase() === DEFAULT_DAEMON_MODEL.toLowerCase();
}

/** True when `minSync.embedder` selects local EmbeddingGemma 2 (`native:` or `native-multimodal:`). */
function isLocalEmbeddingGemma2(embedder: MinSyncEmbedderConfig | undefined): boolean {
	if (embedder?.baseUrl !== undefined) return false;
	const id = embedder?.id;
	if (id === undefined) return false;
	if (!id.startsWith("native:") && !id.startsWith("native-multimodal:")) return false;
	return stripNativeEmbedderPrefix(id).toLowerCase() === "google/embeddinggemma-2";
}

/**
 * The query and passage prefixes for the embedder `minSync.embedder` describes. An operator value, including
 * `""`, always wins. Without one, MinSync's local Qwen3 gets its query instruction, local EmbeddingGemma 2
 * gets its Gemma retrieval prompts, and every other model gets none.
 */
export function resolveEmbeddingPrefixes(embedder: MinSyncEmbedderConfig | undefined): EmbeddingPrefixes {
	if (isLocalEmbeddingGemma2(embedder)) {
		return {
			query: embedder?.queryPrefix ?? EMBEDDINGGEMMA2_QUERY_PREFIX,
			passage: embedder?.passagePrefix ?? EMBEDDINGGEMMA2_PASSAGE_PREFIX,
		};
	}
	return {
		query: embedder?.queryPrefix ?? (isLocalQwen3(embedder) ? QWEN3_QUERY_PREFIX : ""),
		passage: embedder?.passagePrefix ?? "",
	};
}
