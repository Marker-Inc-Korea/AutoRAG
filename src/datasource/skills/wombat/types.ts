/** Retrieval modes of the `wombat` CLI: `keyword` is SPLADE sparse, `semantic` is dense, `hybrid` fuses both. */
export type WombatSearchMode = "keyword" | "semantic" | "hybrid";

export interface WombatSearchOptions {
	readonly topK?: number;
	readonly signal?: AbortSignal;
}

export interface WombatOptions {
	readonly binaryPath?: string;
	readonly instanceId?: string;
	/** `--data-dir`: holds the mirror, the MinSync index and the session catalog (default `~/.wombat`). */
	readonly dataDir?: string;
	/** `--config`: a wombat `config.toml`. */
	readonly configPath?: string;
	/** Restrict indexing and search to these coding agents (`codex`, `claude-code`, `opencode`, `pi`, `omp`). */
	readonly agents?: readonly string[];
	/** `--include-subagents`: index subagent / sidechain transcripts too. */
	readonly includeSubagents?: boolean;
	/** `--include-ephemeral`: index headless, benchmark and temp-directory sessions too. */
	readonly includeEphemeral?: boolean;
	/** Applies to search and `doctor`; `sync` honors `indexTimeoutMs` instead. */
	readonly timeoutMs?: number;
	/** Non-interactive budget for `sync --index`; defaults to one hour. */
	readonly indexTimeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export type WombatFailureReason =
	| "binary-missing"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-output"
	| "remote-embedding-rejected";

export interface WombatFailure {
	readonly ok: false;
	readonly reason: WombatFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export interface WombatDoctorInfo {
	readonly ready: boolean;
	readonly version?: string;
	readonly minsyncVersion?: string;
	readonly indexedSessions: number;
	readonly indexedTurns: number;
}

export interface WombatSyncInfo {
	readonly discovered?: number;
	readonly processed?: number;
	readonly written?: number;
	readonly skipped?: number;
	readonly deleted?: number;
	readonly failed?: number;
	readonly totalSessions: number;
	readonly totalMessages: number;
	/** Chunks MinSync added or updated while indexing; absent when MinSync reported nothing. */
	readonly chunkCount?: number;
	readonly indexOk: boolean;
}

/** One indexed chunk of a coding-agent session, with the pointers needed to reach the full trace. */
export interface WombatSearchHit {
	readonly chunkId: string;
	readonly snippet: string;
	readonly score: number;
	readonly mode: WombatSearchMode;
	readonly agent: string;
	readonly sessionId: string;
	/** `<agent>:<session id>`: what `wombat show` and `wombat trace` take. */
	readonly catalogKey: string;
	readonly title: string;
	readonly cwd?: string;
	readonly sourcePath?: string;
	readonly startTurn?: number;
	readonly endTurn?: number;
	readonly startedAt?: string;
	readonly endedAt?: string;
}

export interface WombatOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface WombatSearchOk {
	readonly ok: true;
	readonly hits: readonly WombatSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export type WombatDoctorResult = WombatOk<WombatDoctorInfo> | WombatFailure;
export type WombatSyncResult = WombatOk<WombatSyncInfo> | WombatFailure;
export type WombatSearchResult = WombatSearchOk | WombatFailure;
