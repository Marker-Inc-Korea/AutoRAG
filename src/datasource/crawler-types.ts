import type { RetrievalOptions } from "../retrieval/types.ts";

export type CrawlerFailureReason =
	| "binary-missing"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-output";

export interface CrawlerFailure {
	readonly ok: false;
	readonly reason: CrawlerFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export interface CrawlerHit {
	readonly id: string;
	readonly content: string;
	readonly score: number;
	readonly title?: string;
	readonly hierarchy?: readonly string[];
	readonly publishedAt?: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface CrawlerSyncOk {
	readonly ok: true;
	readonly count: number;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface CrawlerSearchOk {
	readonly ok: true;
	readonly hits: readonly CrawlerHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export type CrawlerSyncResult = CrawlerSyncOk | CrawlerFailure;
export type CrawlerSearchResult = CrawlerSearchOk | CrawlerFailure;
export type CrawlerSearchOptions = RetrievalOptions;

export interface CrawlerCliOptions {
	readonly binaryPath?: string;
	readonly databasePath?: string;
	readonly sourcePath?: string;
	readonly configPath?: string;
	readonly syncSource?: string;
	/**
	 * Spawn timeout for interactive search in milliseconds. Default 60_000.
	 * `sync` is non-interactive and uses {@link indexTimeoutMs} instead.
	 */
	readonly timeoutMs?: number;
	/**
	 * Spawn timeout for `sync` in milliseconds. Default 1_800_000 (30 min): a first
	 * full import over a real archive can take many minutes, and a timed-out sync
	 * restarts from scratch on the next refresh.
	 */
	readonly indexTimeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly env?: Readonly<Record<string, string | undefined>>;
	/** Working directory for the native CLI process, when explicitly needed. */
	readonly workspacePath?: string;
}

export interface CrawlerProfile {
	readonly binaryName: string;
	readonly allowedEnvPrefixes: readonly string[];
	readonly syncArgs: (options: CrawlerCliOptions) => readonly string[];
	readonly searchArgs: (options: CrawlerCliOptions, query: string, topK: number) => readonly string[];
	readonly parseSyncCount: (stdout: string) => number | undefined;
	readonly parseHits: (stdout: string) => readonly CrawlerHit[] | undefined;
}
