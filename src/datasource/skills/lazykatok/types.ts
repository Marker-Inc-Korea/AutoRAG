import type { RetrievalOptions } from "../../../retrieval/types.ts";

/**
 * Search modes accepted by the external `lazykatok search` subcommand.
 *
 * - `semantic` — dense vector similarity over indexed chunks.
 * - `keyword`  — exact / tokenized keyword matching.
 * - `hybrid`   — fused semantic + keyword scores.
 */
export type LazykatokSearchMode = "semantic" | "keyword" | "hybrid";

/**
 * Configuration for {@link LazykatokClient}. All fields optional; sensible defaults
 * mirror the jikji client. The client spawns the `lazykatok` binary as a child
 * process — it never opens the KakaoTalk database directly.
 */
export interface LazykatokOptions {
	/** Explicit path to the `lazykatok` binary. Defaults to a bare `lazykatok` PATH lookup. */
	readonly binaryPath?: string;
	/** Spawn timeout in milliseconds. Default 60_000 (sync/index over large archives can be slow). */
	readonly timeoutMs?: number;
	/** Max stdout/stderr bytes retained. Default 1_048_576 (1 MiB). */
	readonly maxBufferBytes?: number;
	/** Explicit native lazykatok data directory, passed as `--data-dir`. */
	readonly workspacePath?: string;
	/** Explicit operator-owned configuration/workspace transport. */
	readonly configPath?: string;
	/** Environment overrides merged on top of `process.env` for the child. */
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export const DEFAULT_LAZYKATOK_BINARY = "lazykatok";
export const DEFAULT_LAZYKATOK_TIMEOUT_MS = 60_000;
export const DEFAULT_LAZYKATOK_MAX_BUFFER_BYTES = 1_048_576;
export const DEFAULT_LAZYKATOK_OPTIONS = {
	binaryPath: DEFAULT_LAZYKATOK_BINARY,
	timeoutMs: DEFAULT_LAZYKATOK_TIMEOUT_MS,
	maxBufferBytes: DEFAULT_LAZYKATOK_MAX_BUFFER_BYTES,
} as const;

/**
 * Reasons a lazykatok CLI invocation can fail. The client never throws for these —
 * every method returns a discriminated union with `ok: false` and one of these
 * reasons.
 */
export type LazykatokFailureReason =
	| "binary-missing"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-json"
	| "invalid-shape";

/** Common failure payload shared by every method result union. */
export interface LazykatokFailure {
	readonly ok: false;
	readonly reason: LazykatokFailureReason;
	/** Raw stdout captured (may be partial / empty). Never a binary path. */
	readonly stdout: string;
	/** Raw stderr captured (may be partial / empty). Never a binary path. */
	readonly stderr: string;
	/** Exit code, or `null` when the process never exited normally. */
	readonly code: number | null;
	readonly hits?: readonly unknown[];
}

/** A single indexed chat chunk returned by the lazykatok CLI. */
export interface LazykatokChunk {
	readonly chunkId: string;
	readonly content: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** A scored search hit. */
export interface LazykatokSearchHit {
	readonly chunkId: string;
	readonly score: number;
	readonly content: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
	readonly source?: string;
}
export type LazykatokHit = LazykatokSearchHit;

/** Result of `lazykatok doctor`. */
export interface LazykatokDoctorInfo {
	readonly version?: string;
	readonly ready: boolean;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `lazykatok sync`. */
export interface LazykatokSyncInfo {
	readonly synced: boolean;
	readonly messageCount?: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `lazykatok index`. */
export interface LazykatokIndexInfo {
	readonly chunkCount: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Context payload: the chunks surrounding a target chunk. */
export interface LazykatokContext {
	readonly chunks: readonly LazykatokChunk[];
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Discriminated ok-success shape carrying typed `data`. */
export interface LazykatokOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}
export interface LazykatokSearchOk {
	readonly ok: true;
	readonly hits: readonly LazykatokSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
	readonly data: { readonly hits: readonly LazykatokSearchHit[] };
}

export type LazykatokDoctorResult = LazykatokOk<LazykatokDoctorInfo> | LazykatokFailure;
export type LazykatokSyncResult = LazykatokOk<LazykatokSyncInfo> | LazykatokFailure;
export type LazykatokIndexResult = LazykatokOk<LazykatokIndexInfo> | LazykatokFailure;
export type LazykatokSearchResult = LazykatokSearchOk | LazykatokFailure;
export type LazykatokChunkResult = LazykatokOk<LazykatokChunk> | LazykatokFailure;
export type LazykatokContextResult = LazykatokOk<LazykatokContext> | LazykatokFailure;
export type LazykatokParentResult = LazykatokOk<LazykatokChunk> | LazykatokFailure;

/**
 * Search-specific options. Reuses the shared {@link RetrievalOptions} so the
 * LazykatokSkill retrieval method can pass its options straight through. The
 * client maps `topK` to the native CLI limit flag; other fields are handled
 * by the retrieval layer or are not supported by lazykatok.
 */
export type LazykatokSearchOptions = RetrievalOptions;
