import type { RetrievalOptions } from "../../../retrieval/types.ts";

export type ClawGallerySearchMode = "keyword" | "lexical" | "embedding" | "hybrid";
export type ClawGalleryVdrBackend = "mlx" | "jina-mlx" | "vsplade";

export interface ClawGalleryOptions {
	readonly binaryPath?: string;
	readonly timeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly configDir?: string;
	readonly path?: string;
	readonly syncVisual?: boolean;
	readonly vdrBackend?: ClawGalleryVdrBackend;
	readonly defaultMode?: ClawGallerySearchMode;
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export type ClawGalleryFailureReason =
	| "binary-missing"
	| "nonzero-exit"
	| "spawn-error"
	| "timeout"
	| "aborted"
	| "stdout-too-large"
	| "stderr-too-large"
	| "invalid-json"
	| "invalid-shape";

export interface ClawGalleryFailure {
	readonly ok: false;
	readonly reason: ClawGalleryFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
}

export interface ClawGalleryOk<T> {
	readonly ok: true;
	readonly data: T;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number;
}

export interface ClawGalleryHit {
	readonly imageId: string;
	readonly content: string;
	readonly score: number;
	readonly path?: string;
	readonly title?: string;
	readonly caption?: string;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ClawGalleryIndexInfo {
	readonly indexed: number;
	readonly skipped: number;
	readonly pruned: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export interface ClawGalleryVdrInfo {
	readonly processed: number;
	readonly skipped: number;
	readonly failed: number;
	readonly metadata?: Readonly<Record<string, unknown>>;
}

export type ClawGalleryIndexResult = ClawGalleryOk<ClawGalleryIndexInfo> | ClawGalleryFailure;
export type ClawGalleryVdrResult = ClawGalleryOk<ClawGalleryVdrInfo> | ClawGalleryFailure;
export type ClawGallerySearchResult =
	| (ClawGalleryOk<{ readonly hits: readonly ClawGalleryHit[] }> & {
			readonly hits: readonly ClawGalleryHit[];
	  })
	| ClawGalleryFailure;
export type ClawGallerySearchOptions = RetrievalOptions;
