import { existsSync } from "node:fs";
import { join } from "node:path";
import { resolveMinSyncBinary } from "../minsync/binary.ts";
import type { MinSyncEmbedderConfig } from "../minsync/types.ts";
import { DEFAULT_DAEMON_MODEL } from "./daemon.ts";
import type { ManagedEmbedder } from "./embedder.ts";
import { createManagedEmbedder } from "./managed.ts";
import { remoteEmbedderFromMinSync } from "./settings.ts";

export { DEFAULT_DAEMON_MODEL, ensureMinSyncDaemon, MinSyncDaemonError, probeMinSyncDaemon } from "./daemon.ts";
export type {
	Embedder,
	EmbedderEndpoint,
	EmbedderReadiness,
	EmbeddingBackend,
	EmbeddingIdentity,
	ManagedEmbedder,
} from "./embedder.ts";
export { createHttpEmbedder, type EmbeddingProtocol } from "./http-embedder.ts";
export { loadNativeEmbedder, NATIVE_ADDON_PACKAGE, NativeAddonMissingError } from "./in-process.ts";
export { createManagedEmbedder, type ManagedEmbedderOptions, type RemoteEmbedderSettings } from "./managed.ts";
export { remoteEmbedderFromMinSync } from "./settings.ts";

export interface EmbedderFromConfigOptions {
	/** The `minSync.embedder` block: it selects the embedding backend for MinSync and every other consumer. */
	readonly embedder?: MinSyncEmbedderConfig;
	/** Workspace project root (holds `.autorag/bin`); where a cached `minsync` is looked up. */
	readonly root: string;
	/** MinSync workspace directory; a daemon started here reads the same `[embedder]` settings `minsync sync` does. */
	readonly minSyncWorkspace?: string;
	readonly binaryPath?: string;
	readonly autoInstall?: boolean;
	readonly env?: NodeJS.ProcessEnv;
}

/** The local model a `native:` id names; MinSync's default Qwen3 when none is configured. */
function localModel(embedder: MinSyncEmbedderConfig | undefined): string {
	const id = embedder?.id;
	return id?.startsWith("native:") ? id.slice("native:".length) : DEFAULT_DAEMON_MODEL;
}

/**
 * The single embedder AutoRAG shares between its own semantic stores and the datasources: a configured
 * remote endpoint, else MinSync's `minsync serve` daemon, else MinSync's in-process engine.
 */
export function createEmbedderFromConfig(options: EmbedderFromConfigOptions): ManagedEmbedder {
	const remote = remoteEmbedderFromMinSync(options.embedder);
	// A daemon started here reads the same `[embedder]` settings `minsync sync` does, so both resolve to
	// one server instance. Before the first sync the workspace has no `.minsync/` and the defaults apply.
	const cwd =
		options.minSyncWorkspace !== undefined && existsSync(join(options.minSyncWorkspace, ".minsync"))
			? options.minSyncWorkspace
			: undefined;
	const embedder = createManagedEmbedder({
		...(remote === undefined ? {} : { remote }),
		model: localModel(options.embedder),
		...(cwd === undefined ? {} : { cwd }),
		...(options.env === undefined ? {} : { env: options.env }),
		resolveBinary: () =>
			resolveMinSyncBinary({
				root: options.root,
				...(options.binaryPath === undefined ? {} : { binaryPath: options.binaryPath }),
				...(options.autoInstall === undefined ? {} : { autoInstall: options.autoInstall }),
			}),
	});
	created.add(embedder);
	return embedder;
}

const created = new Set<ManagedEmbedder>();

/**
 * Unload every in-process model this process loaded. A short-lived command calls this on the way out so the
 * addon's native threads cannot keep the process alive after its answer is written. A `minsync serve` daemon
 * is not touched: it belongs to MinSync and stops itself when idle.
 */
export async function releaseEmbedders(): Promise<void> {
	const embedders = [...created];
	created.clear();
	await Promise.allSettled(embedders.map((embedder) => embedder.close()));
}
