/**
 * Types for the optional `jev` decision tool.
 *
 * Jev is TypeSafe's judgment model: it answers typed questions about a state
 * with calibrated probabilities instead of prose. The tool is a pi-agent-core
 * `AgentTool`; the transport lives behind {@link JevEvaluator} so unit tests
 * can inject a judge without touching the network.
 */

/** Jev backends the SDK ships with, in default preference order. */
export const JEV_BACKENDS = ["typesafe", "openrouter", "vercel", "cloudflare"] as const;

export type JevBackend = (typeof JEV_BACKENDS)[number];

export const JEV_TOOL_NAME = "jev";

/** Default provider-neutral Jev model id. */
export const DEFAULT_JEV_MODEL = "jev-latest";

/** Maximum questions per evaluation request (TypeSafe System One limit). */
export const MAX_JEV_QUESTIONS = 32;

/**
 * Tool options for the `jev` decision tool. Secrets never appear here: the
 * backend API key is read from the environment variable named by `apiKeyEnv`,
 * or from the backend's own environment variable when `apiKeyEnv` is omitted.
 */
export interface JevToolOptions {
	/** Jev backend to use. When omitted, the first authenticated backend wins. */
	readonly backend?: JevBackend;
	/** Provider-neutral model id, e.g. `jev-latest` or `jev-1.13`. */
	readonly model?: string;
	/** Environment variable holding the backend API key (never the key itself). */
	readonly apiKeyEnv?: string;
	/** Per-request timeout in milliseconds. SDK default 30_000. */
	readonly timeoutMs?: number;
	/** Retries for 408/429/5xx/connection/timeout failures. SDK default 2. */
	readonly maxRetries?: number;
}

/** Text, a JSON object, a JSON array, or null. Jev reads structure. */
export type JevState = string | Record<string, unknown> | unknown[] | null;

export interface JevNoulQuestion {
	readonly type: "noul";
	readonly instructions: string;
	/** Optional description of the yes outcome. Provide with `false` or not at all. */
	readonly true?: string;
	/** Optional description of the no outcome. Provide with `true` or not at all. */
	readonly false?: string;
}

export interface JevChoiceQuestion {
	readonly type: "choice";
	readonly instructions: string;
	/** Option id to description. The model picks one key. */
	readonly criteria: Readonly<Record<string, string>>;
}

export interface JevScoreQuestion {
	readonly type: "score";
	readonly instructions: string;
	/** Ordered rubric; index 0 is the lowest level. At least two levels. */
	readonly criteria: readonly string[];
}

export type JevQuestion = JevNoulQuestion | JevChoiceQuestion | JevScoreQuestion;

export interface JevAnswerSummary {
	readonly type: JevQuestion["type"];
	/** noul: probability of "yes", 0..1. */
	readonly noul?: number;
	/** choice: the chosen option id. */
	readonly choice?: string;
	/** score: expected rubric index; may be fractional. */
	readonly score?: number;
	/** choice/score agreement; omitted by backends that do not report it. */
	readonly confidence?: number;
	/** Option id or rubric index to probability. */
	readonly probabilities?: Readonly<Record<string, number>>;
}

export interface JevUsageSummary {
	readonly inputTokens: number;
	readonly outputTokens: number;
	readonly totalTokens: number;
	readonly costUsd: number;
}

export interface JevEvaluation {
	readonly provider: string;
	/** Model id as reported by the backend. */
	readonly model: string;
	readonly answers: Readonly<Record<string, JevAnswerSummary>>;
	readonly usage: JevUsageSummary;
}

/** Input to one Jev evaluation: a state plus named typed questions. */
export interface JevEvaluationInput {
	readonly state: JevState;
	readonly questions: Readonly<Record<string, JevQuestion>>;
	readonly signal?: AbortSignal;
}

/**
 * Injectable Jev judge. The tool owns schema validation and formatting; the
 * evaluator owns backend selection and transport.
 */
export interface JevEvaluator {
	evaluate(input: JevEvaluationInput): Promise<JevEvaluation>;
}

/** Thrown when the model passes a malformed tool request or no backend is configured. */
export class JevInputError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "JevInputError";
	}
}
