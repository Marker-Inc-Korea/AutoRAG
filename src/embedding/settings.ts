import type { MinSyncEmbedderConfig } from "../minsync/types.ts";
import type { RemoteEmbedderSettings } from "./managed.ts";

const OPENAI_DEFAULT_BASE_URL = "https://api.openai.com";

/**
 * The remote endpoint that `minSync.embedder` describes, if any. The same block configures MinSync itself
 * and every AutoRAG-owned consumer (memory, chat sidecars, gist, discrawl), so one remote setting reaches
 * all of them. `tei:` ids and a `baseUrl` mean a remote server; `openai:` ids default to api.openai.com;
 * `native:` ids, a missing id, and a missing endpoint mean MinSync's local Qwen3.
 */
export function remoteEmbedderFromMinSync(
	embedder: MinSyncEmbedderConfig | undefined,
): RemoteEmbedderSettings | undefined {
	const id = embedder?.id;
	if (embedder === undefined || id === undefined || id.startsWith("native:")) return undefined;
	const separator = id.indexOf(":");
	const kind = separator === -1 ? undefined : id.slice(0, separator);
	const model = separator === -1 ? id : id.slice(separator + 1);
	const protocol = kind === "tei" ? "tei" : kind === "openai" || embedder.baseUrl !== undefined ? "openai" : undefined;
	if (protocol === undefined) return undefined;
	const baseUrl = embedder.baseUrl ?? (kind === "openai" ? OPENAI_DEFAULT_BASE_URL : undefined);
	if (baseUrl === undefined) return undefined;
	return {
		baseUrl,
		model,
		protocol,
		provider: kind ?? protocol,
		...(embedder.dimension === undefined ? {} : { dimension: embedder.dimension }),
		...(embedder.apiKeyEnv === undefined ? {} : { apiKeyEnv: embedder.apiKeyEnv }),
		...(embedder.batchSize === undefined ? {} : { batchSize: embedder.batchSize }),
		...(embedder.timeoutMs === undefined ? {} : { timeoutMs: embedder.timeoutMs }),
	};
}
