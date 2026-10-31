import { check, type Judgment, type Question } from "jev-use";
import { createJevJudge, type JevJudge, type JevJudgeOptions } from "./jev-extension.ts";
import { SllmDecisionBackend } from "./sllm-decision.ts";

/**
 * AutoRAG cannot answer without its decision model (Jev): routing, the follow-up check, citation and
 * evidence judgment, and the permission gate all ask it. The first question of every session therefore
 * puts one tiny probe to the decision model. When Jev cannot answer, the configured small chat LLM
 * (sLLM) is probed and serves the session in its place, reported as degraded; when neither answers,
 * the run ends with {@link DecisionModelUnavailableError} instead of a pipeline that guesses.
 */

/** Error code every surface reports when the session's decision model cannot be used. */
export const DECISION_MODEL_UNAVAILABLE_CODE = "decision-model-unavailable";

/** Question id of the availability probe. */
export const DECISION_MODEL_PROBE_QUESTION_ID = "decision_model_probe";

/** Why the decision model could not be used. */
export type DecisionModelFailureReason =
	/** The caller tried to turn the decision model off (`jev: false`). */
	| "disabled"
	/** No credential for the backend: an unset API key or a missing sign-in. */
	| "missing-credential"
	/** The backend rejected the credential (HTTP 401/403). */
	| "unauthorized"
	/** The backend account has no credit left (HTTP 402). */
	| "payment-required"
	/** The backend does not serve the configured model (HTTP 404, or "does not exist"). */
	| "model-not-found"
	/** The backend is throttling (HTTP 429). */
	| "rate-limited"
	/** Network failure, timeout, or a 5xx from the backend. */
	| "unreachable"
	/** The backend answered, but not with a usable verdict. */
	| "invalid-response";

const REMEDIES: Readonly<Record<DecisionModelFailureReason, string>> = {
	disabled: "Remove `jev: false`; the decision model cannot be turned off.",
	"missing-credential":
		"Set the backend's API key (OPENROUTER_API_KEY for openrouter, OPENAI_API_KEY for openai, TYPESAFE_API_KEY for typesafe, AI_GATEWAY_API_KEY for vercel-ai-gateway) or sign in to Dazzi Models with /login, then ask again.",
	unauthorized: "The backend rejected its credential; check or replace the API key, then ask again.",
	"payment-required": "The backend account has no credit left; top it up or switch `jev.backend`, then ask again.",
	"model-not-found": "The backend does not serve this model; set `jev.model` to a live decision model id.",
	"rate-limited": "The backend is rate limiting requests; wait a moment and ask again.",
	unreachable: "The backend could not be reached; check the network connection and ask again.",
	"invalid-response": "The backend answered without a usable verdict; check `jev.backend` and `jev.model`.",
};

export interface DecisionModelFailure {
	readonly reason: DecisionModelFailureReason;
	/** The backend's own error text, verbatim. */
	readonly detail: string;
	/** Backend name, when one was resolved. */
	readonly backend?: string;
	/** Model id the backend reported or was asked for. */
	readonly model?: string;
}

/** `backend / model`, or empty when neither is known. */
function describeTarget(failure: DecisionModelFailure): string {
	return [failure.backend, failure.model].filter((part) => part !== undefined).join(" / ");
}

/**
 * The session's decision model cannot be used, so AutoRAG cannot answer. Carries a stable `code`
 * ({@link DECISION_MODEL_UNAVAILABLE_CODE}) and the decision model's `reason` a host can branch on;
 * `attempts` lists every model tried, the decision model first and the sLLM fallback after it. The
 * message names the code, each failure with the backend's own error, and what to do about it.
 */
export class DecisionModelUnavailableError extends Error {
	readonly code = DECISION_MODEL_UNAVAILABLE_CODE;
	readonly reason: DecisionModelFailureReason;
	readonly detail: string;
	readonly backend?: string;
	readonly model?: string;
	readonly attempts: readonly DecisionModelFailure[];

	constructor(failure: DecisionModelFailure, fallback?: DecisionModelFailure) {
		const target = describeTarget(failure);
		const fallbackTarget = fallback === undefined ? "" : describeTarget(fallback);
		super(
			`[${DECISION_MODEL_UNAVAILABLE_CODE}] AutoRAG cannot be used: its decision model (Jev) is unavailable (${failure.reason}). ` +
				`${target === "" ? "" : `${target}: `}${failure.detail} ${REMEDIES[failure.reason]}` +
				(fallback === undefined
					? ""
					: ` The sLLM fallback${fallbackTarget === "" ? "" : ` (${fallbackTarget})`} is unavailable too (${fallback.reason}): ${fallback.detail}`),
		);
		this.name = "DecisionModelUnavailableError";
		this.reason = failure.reason;
		this.detail = failure.detail;
		if (failure.backend !== undefined) this.backend = failure.backend;
		if (failure.model !== undefined) this.model = failure.model;
		this.attempts = fallback === undefined ? [failure] : [failure, fallback];
	}
}

/** Map a backend's error text to a reason. HTTP status wins over wording. */
export function classifyDecisionModelFailure(detail: string): DecisionModelFailureReason {
	const text = detail.toLowerCase();
	if (/\b(401|403)\b|unauthori[sz]ed|forbidden/u.test(text)) return "unauthorized";
	if (/\b402\b|insufficient (credit|balance|fund)|payment required/u.test(text)) return "payment-required";
	if (/\b404\b|does not exist|model[^.]* not found|unknown model|no such model/u.test(text)) return "model-not-found";
	if (/\b429\b|rate.?limit/u.test(text)) return "rate-limited";
	if (/no jev credentials|needs [a-z_]+_api_key|needs a sign-in|not signed in|not configured|no api key/u.test(text)) {
		return "missing-credential";
	}
	if (/timed out|timeout|network error|fetch failed|econn|enotfound|eai_again|\b5\d\d\b|unreachable/u.test(text)) {
		return "unreachable";
	}
	return "invalid-response";
}

/** jev-use wraps a backend failure into this hint for the LLM; the backend's own text is inside it. */
const BACKEND_FAILED_HINT = /^Jev backend failed \((.*)\); answer this yourself\.$/su;

const PROBE_STATE = "AutoRAG starts a session and checks that its decision model answers.";
const PROBE_QUESTION: Question = {
	...check("Does the state say that AutoRAG checks its decision model?"),
	id: DECISION_MODEL_PROBE_QUESTION_ID,
};

/** The decision model that answered the probe. */
export interface DecisionModelStatus {
	readonly backend: string;
	readonly model?: string;
	/** Set when the sLLM fallback answered: why the decision model itself could not. */
	readonly fallbackFrom?: DecisionModelFailure;
}

/**
 * Put one probe question to the decision model. Resolves with the backend and model that answered;
 * throws {@link DecisionModelUnavailableError} when no backend could be built or it gave no number.
 * Whether the probe's answer is "right" does not matter: any number proves the model is reachable,
 * authorized, and served.
 */
export async function checkDecisionModel(judge: JevJudge): Promise<DecisionModelStatus> {
	let judgment: Judgment;
	try {
		judgment = await judge(PROBE_STATE, [PROBE_QUESTION]);
	} catch (error) {
		const detail = error instanceof Error ? error.message : String(error);
		throw new DecisionModelUnavailableError({ reason: classifyDecisionModelFailure(detail), detail });
	}
	const verdict = judgment.verdicts[0];
	const served = { backend: judgment.backend, ...(judgment.model !== undefined ? { model: judgment.model } : {}) };
	if (typeof verdict?.answer === "number") return served;
	const hint = verdict?.hint ?? "the decision model returned no verdict";
	const backendFailure = BACKEND_FAILED_HINT.exec(hint)?.[1];
	throw new DecisionModelUnavailableError({
		reason: backendFailure === undefined ? "invalid-response" : classifyDecisionModelFailure(backendFailure),
		detail: backendFailure ?? hint,
		...served,
	});
}

/** The session's decision model: the judge every caller asks, and the check that picks what it asks. */
export interface DecisionModels {
	/** Asks the model the latest {@link DecisionModels.check} selected: Jev, or the sLLM fallback when Jev could not answer. */
	readonly judge: JevJudge;
	/**
	 * Probe Jev; when it is unavailable and a fallback is configured, probe the fallback. The first that
	 * answers serves every later judgment until the next check. Throws {@link DecisionModelUnavailableError}
	 * with every failure when none answers.
	 */
	check(): Promise<DecisionModelStatus>;
}

/** Jev from `options`, with `options.fallback` (a small chat LLM, or any backend) behind it. */
export function createDecisionModels(options: JevJudgeOptions): DecisionModels {
	const { fallback, ...primaryOptions } = options;
	const primary = createJevJudge(primaryOptions);
	const secondary =
		fallback === undefined
			? undefined
			: createJevJudge({
					backend: "judge" in fallback ? fallback : new SllmDecisionBackend(fallback),
					...(options.confidenceThreshold !== undefined
						? { confidenceThreshold: options.confidenceThreshold }
						: {}),
				});
	let selected = primary;
	return {
		judge: (state, questions, callOptions) => selected(state, questions, callOptions),
		async check() {
			let primaryFailure: DecisionModelUnavailableError;
			try {
				const status = await checkDecisionModel(primary);
				selected = primary;
				return status;
			} catch (error) {
				if (!(error instanceof DecisionModelUnavailableError) || secondary === undefined) throw error;
				primaryFailure = error;
			}
			const [failure] = primaryFailure.attempts;
			if (failure === undefined) throw primaryFailure;
			try {
				const status = await checkDecisionModel(secondary);
				selected = secondary;
				return { ...status, fallbackFrom: failure };
			} catch (error) {
				if (!(error instanceof DecisionModelUnavailableError)) throw error;
				throw new DecisionModelUnavailableError(failure, error.attempts[0]);
			}
		},
	};
}
