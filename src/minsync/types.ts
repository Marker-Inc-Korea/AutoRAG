/** Embedder configuration for MinSync vector indexing. Shared by config and method layers. */

export interface MinSyncEmbedderConfig {
	/** Select an AutoRAG-owned embedding runtime profile. */
	readonly profile?: "qwen3-embedding-0.6b" | "embeddinggemma-300m";
	readonly id?: string;
	readonly baseUrl?: string;
	/** Environment variable name whose value holds the embedder API key. /^[A-Za-z_][A-Za-z0-9_]*$/ */
	readonly apiKeyEnv?: string;
	/** Positive integer embedding dimension. */
	readonly dimension?: number;
	readonly queryPrefix?: string;
	readonly passagePrefix?: string;
	readonly timeoutMs?: number;
	readonly batchSize?: number;
	readonly maxRetries?: number;
	readonly maxConcurrent?: number;
}

/** MinSync chunker settings exposed by AutoRAG's public configuration. */
export interface MinSyncChunkerConfig {
	readonly maxChunkSize?: number;
}

export interface MinSyncOptions {
	readonly root: string;
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly autoInstall?: boolean;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
}

export interface MinSyncSyncResult {
	readonly ok: boolean;
	readonly synced: number;
	readonly workspacePath: string;
	/** True when the unchanged workspace fingerprint avoided an external MinSync sync. */
	readonly skipped?: boolean;
	readonly reason?: string;
	readonly diagnostic?: MinSyncDiagnostic;
	/**
	 * Parsed-mirror ids that could not be staged for indexing because the file
	 * name has no canonical source-id form. Surfaced as diagnostics so an
	 * unindexable document is never dropped silently.
	 */
	readonly stagingExcluded?: readonly string[];
}

export type MinSyncDiagnosticCode =
	| "embedder-unavailable"
	| "sync-failed"
	| "no-hit"
	| "embedding-identity-mismatch"
	| "migration-required";

export interface MinSyncDiagnostic {
	readonly code: MinSyncDiagnosticCode;
	readonly message: string;
	readonly retryable?: boolean;
}

export interface MinSyncQueryHit {
	readonly path: string;
	readonly score: number;
	readonly text: string;
	/**
	 * MinSync's per-chunk identity (`doc_id` in its JSON). Several chunks of one
	 * parsed mirror share a `path`, so this is what distinguishes them; without
	 * it every passage of a document collapses into a single evidence id.
	 * Optional because older MinSync builds omit it.
	 */
	readonly docId?: string;
}
