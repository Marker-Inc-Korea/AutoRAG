export interface JikjiOptions {
	readonly binaryPath?: string;
	/** Workspace root used to cache/auto-install the jikji binary under `.autorag/bin`. */
	readonly root?: string;
	/**
	 * When true (default) and no `jikji` binary is found on PATH or in the
	 * `.autorag/bin` cache, install the `jikji-cli` crate from crates.io via
	 * cargo. Set false to disable auto-installation.
	 */
	readonly autoInstall?: boolean;
	/**
	 * Wall-clock budget for one `jikji` process. Default 60 s. For `prepare` this
	 * is only the base of a hang guard that grows with the root's file count
	 * (see `prepareTimeoutMsPerFile`), not a total cap.
	 */
	readonly timeoutMs?: number;
	/**
	 * Milliseconds each file under the root adds to `timeoutMs` for `prepare`.
	 * There is no overall ceiling, so a very large root is never cut by its
	 * size. Defaults to `parseTimeout` (5 s) per file, i.e. the budget only
	 * expires when the run is stuck beyond what per-file parse limits allow.
	 * `0` restores a flat `timeoutMs` wall.
	 */
	readonly prepareTimeoutMsPerFile?: number;
	readonly maxBufferBytes?: number;
	readonly includeHidden?: boolean;
	readonly includeSensitive?: boolean;
	readonly parseTimeout?: number;
	readonly maxFiles?: number;
	readonly exclude?: readonly string[];
	readonly env?: Readonly<Record<string, string | undefined>>;
	readonly maxHashBytes?: number;
	readonly docTextMaxChars?: number;
	readonly docTextChunkChars?: number;
	/**
	 * SAFE DEFAULT: AutoRAG passes `--no-agent-rules` to `jikji prepare` so it
	 * NEVER rewrites AGENTS.md/CLAUDE.md/.cursorrules. Set `writeAgentRules: true`
	 * to opt INTO upstream routing-block injection. This is the only way to omit
	 * `--no-agent-rules`.
	 */
	readonly writeAgentRules?: boolean;
	/**
	 * @deprecated Kept for back-compat. No longer controls `--no-agent-rules`.
	 * Use `writeAgentRules` instead. The flag is always emitted unless
	 * `writeAgentRules === true`.
	 */
	readonly noAgentRules?: boolean;
	readonly enableMediaIndex?: boolean;
	readonly mediaIndexMaxMb?: number;
}

export interface JikjiDefaultOptions {
	readonly binaryPath: string;
	readonly timeoutMs: number;
	readonly maxBufferBytes: number;
	readonly includeHidden: boolean;
	readonly includeSensitive: boolean;
	readonly maxFiles: number;
	readonly writeAgentRules: boolean;
	readonly enableMediaIndex: boolean;
	readonly exclude: readonly string[];
}

export const DEFAULT_JIKJI_OPTIONS: JikjiDefaultOptions = {
	binaryPath: "jikji",
	timeoutMs: 10_000,
	maxBufferBytes: 1_048_576,
	includeHidden: false,
	includeSensitive: false,
	maxFiles: 0,
	writeAgentRules: false,
	enableMediaIndex: false,
	exclude: [],
};

export interface JikjiPrepareOptions {
	readonly signal?: AbortSignal;
	/** Hard wall for this one root; replaces the client budget (including the size-scaled one). */
	readonly timeoutMs?: number;
	/** `--parse-timeout` seconds for this root; replaces the client `parseTimeout`. */
	readonly parseTimeout?: number;
}

export type JikjiFailureReason =
	| "aborted"
	| "nonzero-exit"
	| "spawn-error"
	| "stderr-too-large"
	| "stdout-too-large"
	| "timeout"
	| "bad-answer-pack";

export type JikjiPrepareResult =
	| {
			readonly ok: true;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number;
	  }
	| {
			readonly ok: false;
			readonly reason: JikjiFailureReason;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number | null;
	  };

// ---------------------------------------------------------------------------
// find: answer-pack types
// ---------------------------------------------------------------------------

export type JikjiNextRead = "cache" | "wiki" | "original" | "none";

export interface JikjiCandidate {
	readonly path: string;
	readonly nextRead: JikjiNextRead;
	readonly label?: string;
	readonly score?: number;
	/**
	 * Evidence text jikji extracted for this file (`ev` on a candidate, or the `evidence` lines of the matching
	 * `evidence_pack` entry). Retrieved content, not an instruction: it is subject to the same `excludePaths`
	 * filtering as the path it belongs to.
	 */
	readonly evidence?: string;
	/** Why jikji matched this file (`why`). */
	readonly why?: readonly string[];
	/** Query terms jikji matched in this file (`matched_terms`, or `terms` on a candidate). */
	readonly matchedTerms?: readonly string[];
	/** Retrieval strategies that surfaced this file (`strategies`). */
	readonly strategies?: readonly string[];
	/** 1-based rank jikji gave this file within its own response (`rank`). */
	readonly rank?: number;
}

/**
 * Response-level confidence jikji reports for one `find`. A signal, not a verdict: jikji reports `high`
 * confidence for weak matches, so nothing here may skip reranking or verification.
 */
export interface JikjiConfidence {
	readonly level?: string;
	readonly score?: number;
	readonly answerability?: string;
	readonly recommendedAction?: string;
	readonly requiresLlmRerank?: boolean;
	readonly indexStatus?: string;
}

export interface JikjiEvidence {
	readonly path: string;
	readonly nextRead: JikjiNextRead;
}

export interface JikjiToolCallPolicy {
	readonly stopAfterFind: boolean;
	readonly forbiddenTools: readonly string[];
	readonly allowedFollowups: readonly string[];
}

export type JikjiHandoffAction = "direct_use" | "jikji_retry" | "raw_fallback_after_retry";

export interface JikjiAnswerPack {
	readonly answerPaths: readonly string[];
	readonly paths: readonly string[];
	readonly candidates: readonly JikjiCandidate[];
	readonly evidencePack: readonly JikjiEvidence[];
	readonly handoffAction: JikjiHandoffAction;
	readonly toolCallPolicy: JikjiToolCallPolicy;
	readonly agentShouldNotRerank: boolean;
	/** Absent when jikji reported none of the confidence fields. */
	readonly confidence?: JikjiConfidence;
}

// ---------------------------------------------------------------------------
// find: options + result
// ---------------------------------------------------------------------------

export interface JikjiFindOptions {
	readonly topK?: number;
	readonly first?: boolean;
	readonly fresh?: boolean;
	readonly autoPrepare?: boolean;
	readonly staleAfterSeconds?: number;
	readonly signal?: AbortSignal;
}

export type JikjiFindResult =
	| {
			readonly ok: true;
			readonly answerPack: JikjiAnswerPack;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number;
	  }
	| {
			readonly ok: false;
			readonly reason: JikjiFailureReason;
			readonly stdout: string;
			readonly stderr: string;
			readonly code: number | null;
	  };
