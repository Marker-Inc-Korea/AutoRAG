export interface MinSyncOptions {
	readonly root: string;
	readonly binaryPath?: string;
	readonly workspacePath?: string;
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
