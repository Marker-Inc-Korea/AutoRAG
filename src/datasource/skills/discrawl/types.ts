import type { RetrievalOptions } from "../../../retrieval/types.ts";

/**
 * Search modes accepted by the external `discrawl search --mode` flag.
 *
 * - `fts`      — SQLite FTS5 lexical ranking over archived messages.
 * - `semantic` — dense vector similarity over locally stored message vectors.
 * - `hybrid`   — runs both, deduplicates by message id, falls back to FTS when
 *                semantic is unavailable.
 */
export type DiscrawlSearchMode = "fts" | "semantic" | "hybrid";

/**
 * Embedding models known to be English-only. Selecting one for a non-English
 * archive silently degrades semantic search to noise rather than failing, so
 * the skill emits a diagnostic instead of staying quiet.
 *
 * Measured on a real Korean archive: unrelated `nomic-embed-text` Korean
 * sentence pairs score 0.65-0.82 cosine (vs 0.30-0.46 for English), i.e. the
 * embedding space collapses and top-k ranking becomes random.
 *
 * See AutoRAG issue #1414.
 */
export const ENGLISH_ONLY_EMBEDDING_MODELS: ReadonlySet<string> = new Set([
	"nomic-embed-text",
	"nomic-embed-text:latest",
	"nomic-embed-text-v1",
	"nomic-embed-text-v1.5",
	"all-minilm",
	"all-minilm:latest",
	"mxbai-embed-large",
	"mxbai-embed-large:latest",
]);

/**
 * Default embedding model AutoRAG configures for workspace-managed discrawl
 * configs. `embeddinggemma`
 * (Gemma 3 300M, 768 dimensions) covers 100+ languages and matches the
 * EmbeddingGemma embedder lazykatok uses for KakaoTalk, so every CLI-backed
 * datasource shares one local embedding model (served through Ollama).
 */
export const DEFAULT_DISCRAWL_EMBEDDING_MODEL = "embeddinggemma";
export const DEFAULT_DISCRAWL_EMBEDDING_PROVIDER = "ollama";

/**
 * Configuration for the discrawl client. All fields optional; defaults mirror
 * the lazykatok client. The client spawns the `discrawl` binary as a child process
 * — it never opens the Discord archive database directly.
 *
 * AutoRAG drives discrawl's local archive only: `wiretap` to import the
 * Discord Desktop cache plus search/embed against that native store. The bot
 * API path belongs to the CLI, which owns its own credential, so no Discord
 * token is configured, required, or forwarded here.
 */
export interface DiscrawlEmbeddingRuntime {
	readonly provider: string;
	readonly model: string;
	readonly baseUrl: string;
	/** Omitted when the endpoint's dimension is not known; discrawl infers it from the first response. */
	readonly dimensions?: number;
	/** Name of the environment variable that holds the bearer token (remote endpoints). Never the token. */
	readonly apiKeyEnv?: string;
	/** A non-secret placeholder for a keyless local daemon, passed to the child only. Never set for a remote endpoint. */
	readonly apiKeyValue?: string;
	readonly identity?: string;
}

export interface DiscrawlEmbeddingConfigResult {
	readonly configured: boolean;
	readonly rebuildRequired: boolean;
	readonly diagnostic?: string;
}

export interface DiscrawlMetadataInfo {
	readonly embeddingProvider?: string;
	readonly embeddingModel?: string;
	readonly embeddingDimensions?: number;
	readonly embeddingIdentity?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface DiscrawlOptions {
	/** Explicit path to the `discrawl` binary. Defaults to a bare PATH lookup. */
	readonly binaryPath?: string;
	/** Spawn timeout in milliseconds. Default 30_000 (sync/embed are slow). */
	readonly timeoutMs?: number;
	/** Max stdout/stderr bytes retained. Default 1_048_576 (1 MiB). */
	readonly maxBufferBytes?: number;
	/**
	 * Explicit operator-owned discrawl config file passed as `--config`.
	 * AutoRAG never rewrites an explicit config.
	 */
	readonly configPath?: string;
	/** Retained for source compatibility; native discrawl defaults are used. */
	readonly root?: string;
	/** Explicit workspace directory overriding the computed default. */
	readonly workspacePath?: string;
	/** Restrict search to one guild id. */
	readonly guildId?: string;
	/** Embedding provider understood by the native discrawl configuration. */
	readonly embeddingProvider?: string;
	/**
	 * Where discrawl should embed, written into a managed workspace config. A function is evaluated on each
	 * `prepareEmbeddings`, because the endpoint of a `minsync serve` daemon exists only while it runs.
	 */
	readonly embeddingRuntime?: DiscrawlEmbeddingRuntime | (() => Promise<DiscrawlEmbeddingRuntime | undefined>);
	/**
	 * Embedding model understood by the native discrawl configuration.
	 * English-only models are accepted but reported through a diagnostic.
	 */
	readonly embeddingModel?: string;
	/**
	 * Default search mode. Defaults to `hybrid` so semantic retrieval covers
	 * terms FTS cannot reach: discrawl strips newlines without substituting a
	 * space, welding words across line breaks into a single unsearchable token
	 * (measured at ~47% of post-newline words on a real archive). See AutoRAG
	 * issue #1413.
	 */
	readonly defaultMode?: DiscrawlSearchMode;
	/**
	 * Spawn timeout for `discrawl attachments fetch`. Defaults to the shared connector sync budget, because a
	 * burst of attachments downloads far longer than one wiretap import.
	 */
	readonly attachmentTimeoutMs?: number;
	/** Environment overrides merged on top of `process.env` for the child. */
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export const DEFAULT_DISCRAWL_BINARY = "discrawl";
export const DEFAULT_DISCRAWL_TIMEOUT_MS = 30_000;
export const DEFAULT_DISCRAWL_MAX_BUFFER_BYTES = 1_048_576;
export const DEFAULT_DISCRAWL_MODE: DiscrawlSearchMode = "hybrid";

/**
 * Reasons a discrawl CLI invocation can fail. The client never throws for
 * these — every method returns a discriminated union with `ok: false`.
 */
export type DiscrawlFailureReason =
	| "binary-missing"
	| "user-token-rejected"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-json"
	| "invalid-shape";

/** Common failure payload shared by every method result union. */
export interface DiscrawlFailure {
	readonly ok: false;
	readonly reason: DiscrawlFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	/**
	 * The offending environment key when `reason === "user-token-rejected"`.
	 * A config key name (never a path, never the key's value).
	 */
	readonly violatingKey?: string;
	readonly hits?: readonly unknown[];
}

/** A single archived Discord message returned by the discrawl CLI. */
export interface DiscrawlMessage {
	readonly messageId: string;
	readonly content: string;
	readonly channelId?: string;
	readonly channelName?: string;
	readonly guildId?: string;
	readonly guildName?: string;
	readonly authorName?: string;
	readonly timestamp?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** A scored search hit. */
export interface DiscrawlSearchHit extends DiscrawlMessage {
	readonly score: number;
}

/** Result of `discrawl status`. */
export interface DiscrawlStatusInfo {
	readonly messages: number;
	readonly channels: number;
	readonly guilds: number;
	readonly databasePath?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `discrawl doctor`. */
export interface DiscrawlDoctorInfo {
	readonly ready: boolean;
	readonly configOk: boolean;
	readonly databaseOk: boolean;
	readonly ftsOk: boolean;
	readonly embeddingsOk: boolean;
	readonly embeddingModel?: string;
	readonly embeddingProvider?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `discrawl wiretap`. */
export interface DiscrawlSyncInfo {
	readonly messages: number;
	readonly guilds?: number;
	readonly channels?: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `discrawl embed`. */
export interface DiscrawlEmbedInfo {
	readonly processed: number;
	readonly succeeded: number;
	readonly failed: number;
	readonly remainingBacklog: number;
	readonly model?: string;
	readonly provider?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Discriminated ok-success shape carrying typed `data`. */
export interface DiscrawlOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface DiscrawlSearchOk {
	readonly ok: true;
	readonly hits: readonly DiscrawlSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
	readonly data: { readonly hits: readonly DiscrawlSearchHit[] };
}

export type DiscrawlDoctorResult = DiscrawlOk<DiscrawlDoctorInfo> | DiscrawlFailure;
export type DiscrawlStatusResult = DiscrawlOk<DiscrawlStatusInfo> | DiscrawlFailure;
export type DiscrawlMetadataResult = DiscrawlOk<DiscrawlMetadataInfo> | DiscrawlFailure;
export type DiscrawlSyncResult = DiscrawlOk<DiscrawlSyncInfo> | DiscrawlFailure;
export type DiscrawlEmbedResult = DiscrawlOk<DiscrawlEmbedInfo> | DiscrawlFailure;
export type DiscrawlSearchResult = DiscrawlSearchOk | DiscrawlFailure;

/** One row of `discrawl --json attachments`: metadata plus discrawl's own fetch bookkeeping. */
export interface DiscrawlAttachmentRow {
	readonly attachmentId: string;
	readonly messageId: string;
	readonly guildId: string;
	readonly guildName?: string;
	readonly channelId: string;
	readonly channelName?: string;
	readonly authorName?: string;
	readonly filename: string;
	readonly contentType?: string;
	readonly size: number;
	readonly url?: string;
	readonly proxyUrl?: string;
	/** Path relative to `<cache_dir>/media`, set once discrawl stored the file. */
	readonly mediaPath?: string;
	readonly contentSha256?: string;
	readonly contentSize?: number;
	/** `fetched`, `failed`, a skip reason such as `too_large`, or empty when never attempted. */
	readonly fetchStatus?: string;
	readonly fetchError?: string;
	/** When discrawl last attempted the download (RFC 3339). */
	readonly fetchedAt?: string;
	readonly createdAt?: string;
}

/** Result of `discrawl --json attachments fetch`. */
export interface DiscrawlAttachmentFetchStats {
	readonly attachments: number;
	readonly fetched: number;
	readonly reused: number;
	readonly skipped: number;
	readonly failed: number;
	readonly bytes: number;
}

export type DiscrawlAttachmentListResult = DiscrawlOk<readonly DiscrawlAttachmentRow[]> | DiscrawlFailure;
export type DiscrawlAttachmentFetchResult = DiscrawlOk<DiscrawlAttachmentFetchStats> | DiscrawlFailure;

/**
 * Search-specific options. Reuses the shared {@link RetrievalOptions} so the
 * retrieval methods can pass their options straight through.
 */
export type DiscrawlSearchOptions = RetrievalOptions;
