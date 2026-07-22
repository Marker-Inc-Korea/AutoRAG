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
	readonly timeoutMs?: number;
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
