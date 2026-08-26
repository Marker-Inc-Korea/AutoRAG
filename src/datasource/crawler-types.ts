import type { ManagedCliConfigManager } from "../cli/managed-cli-config.ts";
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
	readonly timeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly env?: Readonly<Record<string, string | undefined>>;
	/** Workspace root for the shared managed CLI boundary. */
	readonly workspacePath?: string;
	/** Parent-owned managed configuration manager. */
	readonly managedCliConfigManager?: ManagedCliConfigManager;
}

export interface CrawlerProfile {
	readonly binaryName: string;
	readonly allowedEnvPrefixes: readonly string[];
	readonly syncArgs: (options: CrawlerCliOptions) => readonly string[];
	readonly searchArgs: (options: CrawlerCliOptions, query: string, topK: number) => readonly string[];
	readonly parseSyncCount: (stdout: string) => number | undefined;
	readonly parseHits: (stdout: string) => readonly CrawlerHit[] | undefined;
}
