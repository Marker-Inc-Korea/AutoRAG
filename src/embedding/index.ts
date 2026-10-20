import type { MinSyncEmbedderConfig } from "../minsync/types.ts";
import { DEFAULT_DAEMON_MODEL } from "./daemon.ts";
import type { ManagedEmbedder } from "./embedder.ts";
import { createManagedEmbedder } from "./managed.ts";
import { resolveEmbeddingPrefixes } from "./prefixes.ts";
import { remoteEmbedderFromMinSync } from "./settings.ts";

export {
	DEFAULT_DAEMON_MODEL,
	ensureMinSyncDaemon,
	MinSyncDaemonError,
	probeMinSyncDaemon,
	stopMinSyncDaemon,
} from "./daemon.ts";
export type {
	Embedder,
	EmbedderEndpoint,
	EmbedderHealth,
	EmbeddingBackend,
	EmbeddingIdentity,
	EmbedOptions,
	EmbedRole,
	ManagedEmbedder,
} from "./embedder.ts";
export { createHttpEmbedder, type EmbeddingProtocol } from "./http-embedder.ts";
export { loadNativeEmbedder, NATIVE_ADDON_PACKAGE, NativeAddonMissingError } from "./in-process.ts";
export { createManagedEmbedder, type ManagedEmbedderOptions, type RemoteEmbedderSettings } from "./managed.ts";
export { type EmbeddingPrefixes, QWEN3_QUERY_PREFIX, resolveEmbeddingPrefixes } from "./prefixes.ts";
export { remoteEmbedderFromMinSync } from "./settings.ts";

export interface EmbedderFromConfigOptions {
	/** The `minSync.embedder` block: it selects the embedding backend for MinSync and every other consumer. */
	readonly embedder?: MinSyncEmbedderConfig;
	readonly env?: NodeJS.ProcessEnv;
	/** User home that holds MinSync's `minsync serve` state; defaults to the real home. */
	readonly home?: string;
}

/** The local model a `native:` id names; MinSync's default Qwen3 when none is configured. */
function localModel(embedder: MinSyncEmbedderConfig | undefined): string {
	const id = embedder?.id;
	return id?.startsWith("native:") ? id.slice("native:".length) : DEFAULT_DAEMON_MODEL;
}

/**
 * The single embedder AutoRAG shares between its own semantic stores and the datasources: a configured
 * remote endpoint, else the `minsync serve` daemon the operator runs, else MinSync's in-process engine.
 * AutoRAG never starts the daemon for it.
 */
export function createEmbedderFromConfig(options: EmbedderFromConfigOptions): ManagedEmbedder {
	const remote = remoteEmbedderFromMinSync(options.embedder);
	const embedder = createManagedEmbedder({
		...(remote === undefined ? {} : { remote }),
		model: localModel(options.embedder),
		prefixes: resolveEmbeddingPrefixes(options.embedder),
		...(options.home === undefined ? {} : { home: options.home }),
		...(options.env === undefined ? {} : { env: options.env }),
	});
	created.add(embedder);
	return embedder;
}

const created = new Set<ManagedEmbedder>();

/**
 * Unload every in-process model this process loaded. A short-lived command calls this on the way out so the
 * addon's native threads cannot keep the process alive after its answer is written. A `minsync serve` daemon
 * is not touched: it belongs to the operator.
 */
export async function releaseEmbedders(): Promise<void> {
	const embedders = [...created];
	created.clear();
	await Promise.allSettled(embedders.map((embedder) => embedder.close()));
}
