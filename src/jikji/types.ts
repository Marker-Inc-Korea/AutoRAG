export interface JikjiOptions {
	readonly binaryPath?: string;
	readonly timeoutMs?: number;
	readonly maxBufferBytes?: number;
	readonly includeHidden?: boolean;
	readonly includeSensitive?: boolean;
	readonly parseTimeout?: number;
	readonly maxFiles?: number;
	readonly staleAfterSeconds?: number;
	readonly exclude?: readonly string[];
	readonly env?: Readonly<Record<string, string | undefined>>;
}

export interface JikjiDefaultOptions {
	readonly binaryPath: string;
	readonly timeoutMs: number;
	readonly maxBufferBytes: number;
	readonly includeHidden: boolean;
	readonly includeSensitive: boolean;
	readonly parseTimeout: number;
	readonly maxFiles: number;
	readonly staleAfterSeconds: number;
	readonly exclude: readonly string[];
}

export const DEFAULT_JIKJI_OPTIONS: JikjiDefaultOptions = {
	binaryPath: "jikji",
	timeoutMs: 10_000,
	maxBufferBytes: 1_048_576,
	includeHidden: false,
	includeSensitive: false,
	parseTimeout: 5,
	maxFiles: 0,
	staleAfterSeconds: 86_400,
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
	| "timeout";

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
