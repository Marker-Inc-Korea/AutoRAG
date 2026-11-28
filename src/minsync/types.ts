/** Embedder configuration for MinSync vector indexing. Shared by config and method layers. */
export interface MinSyncEmbedderConfig {
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

/** SPLADE sparse encoder settings forwarded to MinSync's `[sparse]` section. */
export interface MinSyncSparseConfig {
	/** `native:nlpai-lab/KURE-Sparse-nano`, `tei:...`, `"none"` to disable SPLADE, or `"auto"` for hardware-based selection. */
	readonly id?: string;
	readonly baseUrl?: string;
	readonly batchSize?: number;
	readonly timeoutMs?: number;
	readonly maxRetries?: number;
}

/** BM25 lexical index settings forwarded to MinSync's `[bm25]` section. */
export interface MinSyncBm25Config {
	/** Whether MinSync builds and maintains `bm25-index.inv`. Default true. */
	readonly enabled?: boolean;
	/** Tokenizer language preset: `multilingual` (default: EN/KO/JA/ZH/AR), `simple`, `ko`, `ja`, `zh`, `ar`. */
	readonly language?: string;
}

/** Independent on/off retriever selection for `minsync query`. */
export interface MinSyncRetrievers {
	readonly vector?: boolean;
	readonly sparse?: boolean;
	readonly bm25?: boolean;
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
	readonly sparse?: MinSyncSparseConfig;
	readonly bm25?: MinSyncBm25Config;
	readonly maxChunkSize?: number;
}

export interface MinSyncSyncProgress {
	readonly processed: number;
	readonly total: number;
}

export interface MinSyncSyncResult {
	readonly ok: boolean;
	readonly synced: number;
	readonly workspacePath: string;
	readonly reason?: string;
	readonly diagnostic?: MinSyncDiagnostic;
	/**
	 * Parsed-mirror ids that could not be staged for indexing because the file
	 * name has no canonical source-id form. Surfaced as diagnostics so an
	 * unindexable document is never dropped silently.
	 */
	readonly stagingExcluded?: readonly string[];
	/**
	 * Parsed-mirror ids whose staged copy failed with a per-document path error
	 * (e.g. `ENAMETOOLONG` or `EINVAL`). Isolated so the rest of the refresh
	 * still succeeds and indexes.
	 */
	readonly stagingFailed?: readonly string[];
	/** Files MinSync skipped (exit 6, `partial`); the sync itself completed and the next sync retries them. */
	readonly filesFailed?: readonly MinSyncFileFailure[];
}

export interface MinSyncFileFailure {
	readonly path: string;
	readonly stage: string;
	readonly error: string;
}

export type MinSyncDiagnosticCode =
	| "embedder-unavailable"
	| "sync-failed"
	| "minsync-lock-held"
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
	/**
	 * MinSync's `heading_path`. It is also the first line of `text` (truncated by MinSync, so not always the
	 * whole `path`); that line is not part of the document. Absent in older builds.
	 */
	readonly headingPath?: string;
	readonly score: number;
	readonly text: string;
	/**
	 * MinSync's per-chunk identity (`doc_id` in its JSON). Several chunks of one
	 * parsed mirror share a `path`, so this is what distinguishes them; without
	 * it every passage of a document collapses into a single evidence id.
	 * Optional because older MinSync builds omit it.
	 */
	readonly docId?: string;
	readonly vectorRank?: number;
	readonly sparseRank?: number;
	readonly bm25Rank?: number;
	readonly vectorScore?: number;
	readonly sparseScore?: number;
	readonly bm25Score?: number;
}
