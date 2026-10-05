import { check, pick, type Question, type Verdict } from "jev-use";
import type { JevJudge } from "./jev-extension.ts";

/**
 * Where a user question is answered from:
 * - `local`: information only the user can reach (their files, chats, mail).
 * - `web`: public information one internet search would answer.
 * - `direct`: general knowledge or small talk the model answers on its own.
 */
export type QueryRoute = "local" | "web" | "direct";

/** Jev question id of the three-way branch. */
export const QUERY_ROUTE_QUESTION_ID = "route";
/** Jev question id of the "does this need decomposition?" check. */
export const DECOMPOSE_QUESTION_ID = "decompose";

/** Route used whenever Jev gives no usable branch: today's local search. */
export const FALLBACK_QUERY_ROUTE: QueryRoute = "local";

/** A `noul` probability at or above this asks for question decomposition. */
export const DECOMPOSE_PROBABILITY_THRESHOLD = 0.5;

const ROUTE_OPTIONS: Record<QueryRoute, string> = {
	local: "Answering needs private information only the user can access and that is not on the public internet: files on their computer, their chats (Discord, KakaoTalk, Slack), their email, or their notes.",
	web: "Not answerable from general knowledge alone, but one public internet search would answer it: current events, recent releases, prices, or other public facts.",
	direct:
		"General common knowledge, a definition, simple reasoning, or small talk the assistant can answer from its own knowledge without searching anything.",
};

const ROUTE_QUESTION = pick("Where must the answer to this user question come from?", ROUTE_OPTIONS);

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

function isQueryRoute(value: unknown): value is QueryRoute {
	return value === "local" || value === "web" || value === "direct";
}

/** The highest-probability branch, even when Jev escalates the verdict as unsure. */
function mostProbableRoute(verdict: Verdict | undefined): { route: QueryRoute; probability?: number } | undefined {
	if (verdict === undefined) return undefined;
	let best: { route: QueryRoute; probability: number } | undefined;
	for (const [label, probability] of Object.entries(verdict.distribution ?? {})) {
		if (!isQueryRoute(label) || !Number.isFinite(probability)) continue;
		if (best === undefined || probability > best.probability) best = { route: label, probability };
	}
	if (best !== undefined) return best;
	return isQueryRoute(verdict.answer) ? { route: verdict.answer } : undefined;
}

/**
 * Ask Jev, in one batched call about the user question, which branch answers
 * it and whether it needs decomposition. Never throws: a missing credential,
 * an unreachable backend, or an unusable verdict falls back to a single local
 * search, with the reason recorded.
 */
export async function routeQuery(judge: JevJudge, query: string): Promise<QueryRouteDecision> {
	const questions: Question[] = [
		{ ...ROUTE_QUESTION, id: QUERY_ROUTE_QUESTION_ID },
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
	const best = mostProbableRoute(routeVerdict);
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
			decomposeProbability !== undefined &&
			decomposeProbability >= DECOMPOSE_PROBABILITY_THRESHOLD,
		...(best.probability !== undefined ? { routeProbability: best.probability } : {}),
		...(decomposeProbability !== undefined ? { decomposeProbability } : {}),
	};
}
