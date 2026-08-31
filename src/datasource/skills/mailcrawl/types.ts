import type { ManagedCliConfigManager } from "../../../cli/managed-cli-config.ts";

export type MailcrawlSearchMode = "keyword" | "bm25" | "semantic" | "hybrid";

export interface MailcrawlSearchOptions {
	readonly topK?: number;
	readonly signal?: AbortSignal;
}

export interface MailcrawlOptions {
	readonly binaryPath?: string;
	readonly instanceId?: string;
	readonly dataDir?: string;
	readonly account?: string;
	readonly mailbox?: string;
	readonly backend?: string;
	readonly source?: string;
	readonly fixture?: string;
	readonly himalayaConfig?: string;
	readonly timeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly env?: Readonly<Record<string, string | undefined>>;
	readonly workspacePath?: string;
	readonly managedCliConfigManager?: ManagedCliConfigManager;
}

export type MailcrawlFailureReason =
	| "binary-missing"
	| "not-configured"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-output"
	| "remote-embedding-rejected";

export interface MailcrawlFailure {
	readonly ok: false;
	readonly reason: MailcrawlFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export interface MailcrawlSyncInfo {
	readonly added?: number;
	readonly updated?: number;
	readonly deleted?: number;
	readonly unchanged?: number;
	readonly chunksAdded?: number;
	readonly archiveRevision?: string;
	readonly messages: number;
}

export interface MailcrawlIndexInfo {
	readonly embedded?: number;
	readonly reused?: number;
	readonly generation?: string;
}

export interface MailcrawlSearchHit {
	readonly chunkId: string;
	readonly messageId: string;
	readonly threadId: string;
	readonly accountId: string;
	readonly mailbox: string;
	readonly subject: string;
	readonly from: string;
	readonly to: readonly string[];
	readonly date: string;
	readonly snippet: string;
	readonly score: number;
	readonly mode: MailcrawlSearchMode;
}

export interface MailcrawlOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface MailcrawlSearchOk {
	readonly ok: true;
	readonly hits: readonly MailcrawlSearchHit[];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export type MailcrawlSyncResult = MailcrawlOk<MailcrawlSyncInfo> | MailcrawlFailure;
export type MailcrawlSearchResult = MailcrawlSearchOk | MailcrawlFailure;
