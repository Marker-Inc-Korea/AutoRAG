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
	/**
	 * Applies to every operation. `sync`/`index` additionally honor
	 * `indexTimeoutMs`, because a 0.2.0 native index run is not interactive.
	 */
	readonly timeoutMs?: number;
	/** Non-interactive budget for `sync`/`index`; defaults to 30 minutes. */
	readonly indexTimeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly env?: Readonly<Record<string, string | undefined>>;
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
	/** 0.2.0 LanceDB index report: archive revision the vectors were built for. */
	readonly archiveRevision?: string;
	/** 0.2.0: the store was rebuilt instead of incrementally extended. */
	readonly rebuilt?: boolean;
	/** 0.2.0: embedder identity the vectors belong to, e.g. `native:Qwen/Qwen3-Embedding-0.6B:1024`. */
	readonly embedder?: string;
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
