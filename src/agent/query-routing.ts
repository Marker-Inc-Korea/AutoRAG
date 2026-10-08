import { check, pick, type Question, type Verdict } from "jev-use";
import type { JevJudge } from "./jev-extension.ts";

/**
 * Where a user question is answered from:
 * - `local`: information only the user can reach (their files, chats, mail).
 * - `web`: public information one internet search would answer.
 * - `direct`: general knowledge or small talk the model answers on its own.
 * - `config`: the user wants to inspect or change AutoRAG's own settings
 *   (model, providers, datasources). Offered only when the host enables
 *   agent self-configuration.
 */
export type QueryRoute = "local" | "web" | "direct" | "config";

/** Jev question id of the three-way branch. */
export const QUERY_ROUTE_QUESTION_ID = "route";
/** Jev question id of the "does this need decomposition?" check. */
export const DECOMPOSE_QUESTION_ID = "decompose";
/** Jev question id of the post-fast-answer "needs follow-up?" check. */
export const FOLLOW_UP_QUESTION_ID = "follow_up";

/** Route used whenever Jev gives no usable branch: today's local search. */
export const FALLBACK_QUERY_ROUTE: QueryRoute = "local";

/** A `noul` probability at or above this asks for question decomposition. */
export const DECOMPOSE_PROBABILITY_THRESHOLD = 0.5;

/** A `noul` probability at or above this sends the fast answer on to verification. */
export const FOLLOW_UP_PROBABILITY_THRESHOLD = 0.5;

/**
 * Wording calibrated against live Jev (OpenRouter `typesafe/jev-1.13`): asking
 * whether the answer "needs correction or further research before it can be
 * trusted" scored complete, evidence-backed answers at 0.75-0.86 and would have
 * sent nearly every answer to verification. Asking about unanswered, uncertain,
 * or possibly inaccurate parts separated the cases cleanly (final <= 0.37,
 * needs follow-up >= 0.98).
 */
const FOLLOW_UP_QUESTION = check(
	"Is any part of the user's question left unanswered, uncertain, possibly inaccurate, or asking the user to clarify?",
	{
		true: "Some part is missing, hedged, vague, contradictory, says it was not found or needs verification, or the answer asks the user to clarify.",
		false: "Every part of the question is answered with a concrete, specific, consistent fact (a name, date, number, or definite statement).",
	},
);

export interface FollowUpDecision {
	/** True when the run continues into the verification phase. */
	readonly followUp: boolean;
	/** Jev's P(follow-up needed), when it answered. */
	readonly probability?: number;
	/** Why the check fell back to verifying instead of following Jev; absent when Jev decided. */
	readonly fallbackReason?: string;
}

/**
 * Ask Jev whether the fast answer needs correction, clarification, or further
 * research. Never throws: a missing credential, an unreachable backend, or a
 * verdict with no probability keeps verifying, because ending early on an
 * unchecked answer is the costlier mistake.
 */
export async function needsFollowUp(judge: JevJudge, query: string, answer: string): Promise<FollowUpDecision> {
	let verdicts: readonly Verdict[];
	try {
		verdicts = (
			await judge(`User question: ${query}\n\nFirst answer:\n${answer}`, [
				{ ...FOLLOW_UP_QUESTION, id: FOLLOW_UP_QUESTION_ID },
			])
		).verdicts;
	} catch (error) {
		return { followUp: true, fallbackReason: error instanceof Error ? error.message : String(error) };
	}
	const verdict = verdicts.find((entry) => entry.id === FOLLOW_UP_QUESTION_ID);
	if (typeof verdict?.answer !== "number") {
		return { followUp: true, fallbackReason: verdict?.hint ?? "Jev returned no usable follow-up verdict." };
	}
	return { followUp: verdict.answer >= FOLLOW_UP_PROBABILITY_THRESHOLD, probability: verdict.answer };
}

const BASE_ROUTE_OPTIONS: Record<Exclude<QueryRoute, "config">, string> = {
	local: "Answering needs private information only the user can access and that is not on the public internet: files on their computer, their chats (Discord, KakaoTalk, Slack), their email, or their notes.",
	web: "Not answerable from general knowledge alone, but one public internet search would answer it: current events, recent releases, prices, or other public facts.",
	direct:
		"General common knowledge, a definition, simple reasoning, or small talk the assistant can answer from its own knowledge without searching anything.",
};

const CONFIG_ROUTE_OPTION =
	"The user asks to view, change, or test the settings of this AutoRAG agent itself: its model, model providers, API-key environment variables, Jev routing, search roots, or datasource setup. It is about configuring the assistant, not about the content of their documents.";

const BASE_ROUTE_QUESTION = pick("Where must the answer to this user question come from?", BASE_ROUTE_OPTIONS);
const SELF_CONFIG_ROUTE_QUESTION = pick("Where must the answer to this user question come from?", {
	...BASE_ROUTE_OPTIONS,
	config: CONFIG_ROUTE_OPTION,
});

const DECOMPOSE_QUESTION = check(
	"Does answering this question need several separate search queries instead of one search for the question as written?",
	{
		true: "It combines several sub-questions, compares items, or needs multiple independent facts confirmed, so one search would miss parts of it.",
		false: "One direct search for the question as written would already find the answer.",
	},
);

export interface QueryRouteDecision {
	/** Most probable branch; {@link FALLBACK_QUERY_ROUTE} when Jev gave none. */
	readonly route: QueryRoute;
	/** True when Jev judged the question needs decomposition (never for `direct`). */
	readonly decompose: boolean;
	/** Probability Jev gave the chosen branch, when it reported a distribution. */
	readonly routeProbability?: number;
	/** Jev's P(decomposition needed), when it answered. */
	readonly decomposeProbability?: number;
	/** Why the decision fell back instead of following Jev; absent when Jev decided. */
	readonly fallbackReason?: string;
}

function isQueryRoute(value: unknown, selfConfig: boolean): value is QueryRoute {
	return value === "local" || value === "web" || value === "direct" || (selfConfig && value === "config");
}

/** The highest-probability branch, even when Jev escalates the verdict as unsure. */
function mostProbableRoute(
	verdict: Verdict | undefined,
	selfConfig: boolean,
): { route: QueryRoute; probability?: number } | undefined {
	if (verdict === undefined) return undefined;
	let best: { route: QueryRoute; probability: number } | undefined;
	for (const [label, probability] of Object.entries(verdict.distribution ?? {})) {
		if (!isQueryRoute(label, selfConfig) || !Number.isFinite(probability)) continue;
		if (best === undefined || probability > best.probability) best = { route: label, probability };
	}
	if (best !== undefined) return best;
	return isQueryRoute(verdict.answer, selfConfig) ? { route: verdict.answer } : undefined;
}

export interface RouteQueryOptions {
	/** Offer the `config` branch: the agent may change its own settings. */
	readonly selfConfig?: boolean;
}

/**
 * Ask Jev, in one batched call about the user question, which branch answers
 * it and whether it needs decomposition. Never throws: a missing credential,
 * an unreachable backend, or an unusable verdict falls back to a single local
 * search, with the reason recorded.
 */
export async function routeQuery(
	judge: JevJudge,
	query: string,
	options: RouteQueryOptions = {},
): Promise<QueryRouteDecision> {
	const selfConfig = options.selfConfig === true;
	const questions: Question[] = [
		{ ...(selfConfig ? SELF_CONFIG_ROUTE_QUESTION : BASE_ROUTE_QUESTION), id: QUERY_ROUTE_QUESTION_ID },
		{ ...DECOMPOSE_QUESTION, id: DECOMPOSE_QUESTION_ID },
	];
	let verdicts: readonly Verdict[];
	try {
		verdicts = (await judge(`User question: ${query}`, questions)).verdicts;
	} catch (error) {
		return {
			route: FALLBACK_QUERY_ROUTE,
			decompose: false,
			fallbackReason: error instanceof Error ? error.message : String(error),
		};
	}
	const routeVerdict = verdicts.find((verdict) => verdict.id === QUERY_ROUTE_QUESTION_ID);
	const best = mostProbableRoute(routeVerdict, selfConfig);
	if (best === undefined) {
		return {
			route: FALLBACK_QUERY_ROUTE,
			decompose: false,
			fallbackReason: routeVerdict?.hint ?? "Jev returned no usable route verdict.",
		};
	}
	const decomposeVerdict = verdicts.find((verdict) => verdict.id === DECOMPOSE_QUESTION_ID);
	const decomposeProbability = typeof decomposeVerdict?.answer === "number" ? decomposeVerdict.answer : undefined;
	return {
		route: best.route,
		decompose:
			best.route !== "direct" &&
			best.route !== "config" &&
			decomposeProbability !== undefined &&
			decomposeProbability >= DECOMPOSE_PROBABILITY_THRESHOLD,
		...(best.probability !== undefined ? { routeProbability: best.probability } : {}),
		...(decomposeProbability !== undefined ? { decomposeProbability } : {}),
	};
}
