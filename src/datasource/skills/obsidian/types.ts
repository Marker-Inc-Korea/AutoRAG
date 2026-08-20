import type { RetrievalOptions } from "../../../retrieval/types.ts";

/**
 * Search modes accepted by the external `qmd` CLI.
 *
 * - `search`  — BM25 / FTS5 lexical search
 * - `vsearch` — dense vector semantic search
 */
export type QmdSearchMode = "search" | "vsearch";

/**
 * Configuration for {@link QmdClient}. The client spawns the `qmd` binary as a
 * child process and isolates config/cache under the AutoRAG workspace.
 */
export interface QmdOptions {
	/** Explicit path to the `qmd` binary. Defaults to a bare `qmd` PATH lookup. */
	readonly binaryPath?: string;
	/** Spawn timeout in milliseconds. Default 60_000 (embed can be slow). */
	readonly timeoutMs?: number;
	/** Max stdout/stderr bytes retained. Default 1_048_576 (1 MiB). */
	readonly maxBufferBytes?: number;
	/** Absolute path to the Obsidian vault. Required for ensure/update. */
	readonly vaultPath?: string;
	/** AutoRAG workspace root used to isolate qmd config/cache. */
	readonly workspaceRoot?: string;
	/** Instance id used for source paths and collection naming. Default `default`. */
	readonly instanceId?: string;
	/** Explicit collection name; defaults to a sanitized instance id. */
	readonly collectionName?: string;
	/** Environment overrides merged on top of a restricted process env. */
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export const DEFAULT_QMD_BINARY = "qmd";
export const DEFAULT_QMD_TIMEOUT_MS = 60_000;
export const DEFAULT_QMD_MAX_BUFFER_BYTES = 1_048_576;

export type QmdFailureReason =
	| "binary-missing"
	| "not-configured"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-json"
	| "invalid-shape";

export interface QmdFailure {
	readonly ok: false;
	readonly reason: QmdFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export interface QmdOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface QmdUpdateInfo {
	readonly indexed: number;
	readonly updated: number;
	readonly unchanged: number;
	readonly removed: number;
	readonly needsEmbedding?: boolean;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface QmdEmbedInfo {
	readonly embedded: boolean;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface QmdEnsureInfo {
	readonly collectionName: string;
	readonly vaultPath: string;
	readonly configDir: string;
}

/** A scored search hit from `qmd search` / `qmd vsearch`. */
export interface QmdSearchHit {
	readonly chunkId: string;
	readonly score: number;
	readonly content: string;
	readonly title?: string;
	readonly file?: string;
	readonly docid?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface QmdSearchOk {
	readonly ok: true;
	readonly hits: readonly QmdSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
	readonly data: { readonly hits: readonly QmdSearchHit[] };
}

export type QmdEnsureResult = QmdOk<QmdEnsureInfo> | QmdFailure;
export type QmdUpdateResult = QmdOk<QmdUpdateInfo> | QmdFailure;
export type QmdEmbedResult = QmdOk<QmdEmbedInfo> | QmdFailure;
export type QmdSearchResult = QmdSearchOk | QmdFailure;

export type QmdSearchOptions = RetrievalOptions;
