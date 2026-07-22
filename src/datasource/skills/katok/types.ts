import type { RetrievalOptions } from "../../../retrieval/types.ts";

/**
 * Search modes accepted by the external `katok search` subcommand.
 *
 * - `semantic` — dense vector similarity over indexed chunks.
 * - `keyword`  — exact / tokenized keyword matching.
 * - `hybrid`   — fused semantic + keyword scores.
 */
export type KatokSearchMode = "semantic" | "keyword" | "hybrid";

/**
 * Where the `katok` CLI reads KakaoTalk data from.
 *
 * - `macos`   — live KakaoTalk install on macOS (the CLI owns DB access; this
 *               client never touches the DB directly).
 * - `fixture` — a frozen fixture directory (used for tests / dry runs).
 */
export type KatokSourceKind = "macos" | "fixture";

/**
 * Configuration for {@link KatokClient}. All fields optional; sensible defaults
 * mirror the jikji client. The client spawns the `katok` binary as a child
 * process — it never opens the KakaoTalk database directly.
 */
export interface KatokOptions {
	/** Explicit path to the `katok` binary. Defaults to a bare `katok` PATH lookup. */
	readonly binaryPath?: string;
	/** Spawn timeout in milliseconds. Default 10_000. */
	readonly timeoutMs?: number;
	/** Max stdout/stderr bytes retained. Default 1_048_576 (1 MiB). */
	readonly maxBufferBytes?: number;
	/** Data source for the CLI. Default `macos`. */
	readonly source?: KatokSourceKind;
	/** Path to a fixture directory; required when `source === "fixture"`. */
	readonly fixturePath?: string;
	/**
	 * Explicit katok workspace directory (the `.autorag/datasources/katok`
	 * root). When omitted, computed from {@link KatokOptions.root} (or
	 * `process.cwd()`) via the paths helper.
	 */
	readonly workspacePath?: string;
	/** Workspace root used to compute the default katok workspace path. */
	readonly root?: string;
	/** Environment overrides merged on top of `process.env` for the child. */
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export const DEFAULT_KATOK_BINARY = "katok";
export const DEFAULT_KATOK_TIMEOUT_MS = 10_000;
export const DEFAULT_KATOK_MAX_BUFFER_BYTES = 1_048_576;
export const DEFAULT_KATOK_SOURCE: KatokSourceKind = "macos";
export const DEFAULT_KATOK_OPTIONS = {
	binaryPath: DEFAULT_KATOK_BINARY,
	timeoutMs: DEFAULT_KATOK_TIMEOUT_MS,
	maxBufferBytes: DEFAULT_KATOK_MAX_BUFFER_BYTES,
	source: DEFAULT_KATOK_SOURCE,
} as const;

/**
 * Reasons a katok CLI invocation can fail. The client never throws for these —
 * every method returns a discriminated union with `ok: false` and one of these
 * reasons.
 */
export type KatokFailureReason =
	| "binary-missing"
	| "remote-embedding-rejected"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-json"
	| "invalid-shape";

/** Common failure payload shared by every method result union. */
export interface KatokFailure {
	readonly ok: false;
	readonly reason: KatokFailureReason;
	/** Raw stdout captured (may be partial / empty). Never a binary path. */
	readonly stdout: string;
	/** Raw stderr captured (may be partial / empty). Never a binary path. */
	readonly stderr: string;
	/** Exit code, or `null` when the process never exited normally. */
	readonly code: number | null;
	/**
	 * The offending environment key when `reason === "remote-embedding-rejected"`.
	 * A config key name (never a path, never the key's value).
	 */
	readonly violatingKey?: string;
	readonly hits?: readonly unknown[];
}

/** A single indexed chat chunk returned by the katok CLI. */
export interface KatokChunk {
	readonly chunkId: string;
	readonly content: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** A scored search hit. */
export interface KatokSearchHit {
	readonly chunkId: string;
	readonly score: number;
	readonly content: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
	readonly source?: string;
}
export type KatokHit = KatokSearchHit;

/** Result of `katok doctor`. */
export interface KatokDoctorInfo {
	readonly version?: string;
	readonly ready: boolean;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `katok sync`. */
export interface KatokSyncInfo {
	readonly synced: boolean;
	readonly messageCount?: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Result of `katok index`. */
export interface KatokIndexInfo {
	readonly chunkCount: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Context payload: the chunks surrounding a target chunk. */
export interface KatokContext {
	readonly chunks: readonly KatokChunk[];
	readonly metadata?: Readonly<Record<string, unknown>>;
}

/** Discriminated ok-success shape carrying typed `data`. */
export interface KatokOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}
export interface KatokSearchOk {
	readonly ok: true;
	readonly hits: readonly KatokSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
	readonly data: { readonly hits: readonly KatokSearchHit[] };
}

export type KatokDoctorResult = KatokOk<KatokDoctorInfo> | KatokFailure;
export type KatokSyncResult = KatokOk<KatokSyncInfo> | KatokFailure;
export type KatokIndexResult = KatokOk<KatokIndexInfo> | KatokFailure;
export type KatokSearchResult = KatokSearchOk | KatokFailure;
export type KatokChunkResult = KatokOk<KatokChunk> | KatokFailure;
export type KatokContextResult = KatokOk<KatokContext> | KatokFailure;
export type KatokParentResult = KatokOk<KatokChunk> | KatokFailure;

/**
 * Search-specific options. Reuses the shared {@link RetrievalOptions} so the
 * KatokSkill retrieval method can pass its options straight through. The
 * client maps `topK` and `scope` to CLI flags; other fields are accepted but
 * not currently forwarded.
 */
export type KatokSearchOptions = RetrievalOptions;
