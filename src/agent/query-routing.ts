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
 * Minimum Jev branch probability required to leave local search. A `direct` or
 * `web` branch at or below this floor — or one Jev reported without a
 * probability — keeps the run on local search, the corpus-grounded default.
 * Without the floor a weakly-favored `direct` verdict (observed at p 0.57-0.63
 * on personal-corpus questions) skips retrieval entirely and answers from
 * general knowledge, so the caller loses the evidence the question depends on.
 */
export const NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD = 0.75;

/** A `noul` probability at or above this searches that datasource before the fast answer. */
export const DATASOURCE_PROBABILITY_THRESHOLD = 0.5;

/** Jev question id of the "search this datasource?" check for one datasource. */
export function datasourceQuestionId(datasourceId: string): string {
	return `datasource:${datasourceId}`;
}

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
	local: "Answering needs private information only the user can access and that is not on the public internet: files on their computer, their chats (Discord, KakaoTalk, Slack), their email, their notes, their calendar, or their history. Prefer local whenever the question refers to the user's own life, situation, plans, or records — questions phrased with words like 'my', 'I', 'we', 'our', a named friend, family member, or colleague, or 'my case/hearing/appointment/routine' — even when a generic answer would also be possible.",
	web: "Not answerable from general knowledge alone and not from the user's private information either, but one public internet search would answer it: current events, recent releases, prices, or other public facts.",
	direct:
		"General common knowledge, a definition, simple reasoning, or small talk the assistant can answer from its own knowledge without searching anything. Never direct when the question refers to the user's own life, files, records, chats, calendar, or history (e.g. 'my', 'I', 'our', 'my girlfriend', 'my case', 'my routine').",
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
	/** Most probable branch; {@link FALLBACK_QUERY_ROUTE} when Jev gave none or its non-local branch missed {@link NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD}. */
	readonly route: QueryRoute;
	/** True when Jev judged the question needs decomposition (never for `direct`). */
	readonly decompose: boolean;
	/** Probability Jev gave the winning branch, including a rejected non-local branch whose floor moved the route to local. */
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
 * an unreachable backend, an unusable verdict, or a non-local branch that does
 * not clear {@link NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD} falls back to a
 * single local search, with the reason recorded.
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
	const decompose = decomposeProbability !== undefined && decomposeProbability >= DECOMPOSE_PROBABILITY_THRESHOLD;
	// Leaving local search drops corpus evidence, so it needs a confident branch:
	// a weak or probability-less `direct`/`web` verdict keeps the local default.
	if (
		best.route !== "local" &&
		(best.probability === undefined || best.probability <= NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD)
	) {
		return {
			route: FALLBACK_QUERY_ROUTE,
			decompose,
			fallbackReason:
				`Jev's ${best.route} route probability (${best.probability ?? "not reported"}) is not above the ` +
				`${NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD} confidence floor; using local search.`,
			...(best.probability !== undefined ? { routeProbability: best.probability } : {}),
			...(decomposeProbability !== undefined ? { decomposeProbability } : {}),
		};
	}
	return {
		route: best.route,
		decompose: best.route !== "direct" && best.route !== "config" && decompose,
		...(best.probability !== undefined ? { routeProbability: best.probability } : {}),
		...(decomposeProbability !== undefined ? { decomposeProbability } : {}),
	};
}

/** One configured datasource Jev may send the question to. */
export interface DatasourceCandidate {
	readonly datasourceId: string;
	/** Content kind, e.g. `chat`, `mail`, `docs`. */
	readonly type: string;
	/** Operator- or connector-authored description of what the datasource holds. */
	readonly description: string;
}

export interface DatasourceSelection {
	/** Datasource ids to search, in registration order. */
	readonly selected: readonly string[];
	/** Jev's P(search needed) per datasource id, for every datasource it answered. */
	readonly probabilities: Readonly<Record<string, number>>;
	/** Why no datasource is searched instead of following Jev; absent when Jev decided. */
	readonly fallbackReason?: string;
}

/** A similar question asked before, its results, and where each was found. */
export interface PastSearchHint {
	readonly query: string;
	/**
	 * Results it returned. `foundIn` names registered datasource ids, `local
	 * files`, or `web`; a title can also record that nothing was found there.
	 */
	readonly results: readonly { readonly title: string; readonly foundIn: readonly string[] }[];
}

function describeDatasources(
	query: string,
	datasources: readonly DatasourceCandidate[],
	pastSearches: readonly PastSearchHint[],
): string {
	const catalog = datasources
		.map((datasource) => `- ${datasource.datasourceId} (${datasource.type}): ${datasource.description}`)
		.join("\n");
	// Past questions and titles are model- or caller-written text: JSON-quote
	// them so a newline or a fake "User question:" line cannot forge the
	// state's structure.
	const history =
		pastSearches.length === 0
			? ""
			: `\n\nSimilar past questions, the results they returned, and where each result came from. A hint, ` +
				`not a rule: the same question can need other datasources this time. A result titled as not ` +
				`found / no result / negative means that datasource was searched and did NOT have the answer.\n${pastSearches
					.map(
						(past) =>
							`- ${JSON.stringify(past.query)}${past.results
								.map((result) => `\n  - ${JSON.stringify(result.title)} [${result.foundIn.join(", ")}]`)
								.join("")}`,
					)
					.join("\n")}`;
	return (
		`An assistant answers the user's question from their local files and a set of registered datasources. ` +
		`Local files are always searched. Each registered datasource is searched only when its content could hold ` +
		`part of the answer.\n\n` +
		`Each description below is only a short summary of what the operator knows is in that datasource, not a ` +
		`complete list: every datasource can also hold other topics, people, and conversations that are not ` +
		`mentioned. Do not rule a datasource out only because its description does not mention the question's ` +
		`topic; judge by what kind of content the datasource holds and what the question asks for.\n\n` +
		`Registered datasources:\n${catalog}${history}\n\nUser question: ${query}`
	);
}

/**
 * Ask Jev, in one batched call, one `noul` per registered datasource: does
 * answering the question need that datasource searched? The state describes
 * every datasource (descriptions are flagged as non-exhaustive summaries), so
 * Jev judges each one against the others, plus where similar past questions
 * were answered, when retrieval memory has any. Never throws: a missing
 * credential or an unreachable backend searches no datasource, which is the
 * behavior without Jev.
 */
export async function selectDatasources(
	judge: JevJudge,
	query: string,
	datasources: readonly DatasourceCandidate[],
	pastSearches: readonly PastSearchHint[] = [],
): Promise<DatasourceSelection> {
	if (datasources.length === 0) return { selected: [], probabilities: {} };
	let verdicts: readonly Verdict[];
	try {
		verdicts = (
			await judge(
				describeDatasources(query, datasources, pastSearches),
				datasources.map(({ datasourceId }) => ({
					...check(`Should the "${datasourceId}" datasource be searched to answer the user question?`, {
						true: `The "${datasourceId}" datasource's content could contain information that answers some part of the question.`,
						false: `The "${datasourceId}" datasource's content is unrelated to the question, or the question needs no datasource at all.`,
					}),
					id: datasourceQuestionId(datasourceId),
				})),
			)
		).verdicts;
	} catch (error) {
		return {
			selected: [],
			probabilities: {},
			fallbackReason: error instanceof Error ? error.message : String(error),
		};
	}
	const probabilities: Record<string, number> = {};
	for (const datasource of datasources) {
		const answer = verdicts.find((verdict) => verdict.id === datasourceQuestionId(datasource.datasourceId))?.answer;
		if (typeof answer === "number") probabilities[datasource.datasourceId] = answer;
	}
	if (Object.keys(probabilities).length === 0) {
		return {
			selected: [],
			probabilities,
			fallbackReason:
				verdicts.find((verdict) => verdict.hint !== undefined)?.hint ??
				"Jev returned no usable datasource verdict.",
		};
	}
	return {
		selected: datasources
			.map((datasource) => datasource.datasourceId)
			.filter((datasourceId) => (probabilities[datasourceId] ?? 0) >= DATASOURCE_PROBABILITY_THRESHOLD),
		probabilities,
	};
}
