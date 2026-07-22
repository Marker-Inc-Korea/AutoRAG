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

export interface MinSyncOptions {
	readonly root: string;
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly autoInstall?: boolean;
	readonly embedder?: MinSyncEmbedderConfig;
}

export interface MinSyncSyncResult {
	readonly ok: boolean;
	readonly synced: number;
	readonly workspacePath: string;
	readonly reason?: string;
}

export interface MinSyncQueryHit {
	readonly path: string;
	readonly score: number;
	readonly text: string;
}
