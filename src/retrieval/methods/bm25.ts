import { existsSync, rmSync } from "node:fs";
import { join } from "node:path";

export type BM25ReadinessState =
	| "disabled"
	| "dependency_unavailable"
	| "index_missing"
	| "indexing"
	| "ready"
	| "error";

/** Local BM25 is always MinSync-backed when enabled. */
export type BM25Engine = "minsync" | "none";

export interface BM25SyncResult {
	readonly indexPath: string;
	readonly indexedChunks: number;
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
}

export interface BM25Status {
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
	readonly message?: string;
}

/** Legacy AutoRAG-local BM25 artifact directory. Never an active index. */
export const BM25_SUBDIR = join(".autorag", "bm25");

export class BM25UnavailableError extends Error {
	readonly readiness: BM25ReadinessState;

	constructor(readiness: BM25ReadinessState, message: string) {
		super(message);
		this.name = "BM25UnavailableError";
		this.readiness = readiness;
	}
}

/** Remove leftover `.autorag/bm25` artifacts from the pre-MinSync BM25 path. */
export function removeLegacyBm25Artifacts(root: string): void {
	rmSync(join(root, BM25_SUBDIR), { recursive: true, force: true });
}

export function hasLegacyBm25Artifacts(root: string): boolean {
	return existsSync(join(root, BM25_SUBDIR));
}
