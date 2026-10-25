import { check, pick, type Question, type Verdict } from "jev-use";
import type { JevJudge } from "./jev-extension.ts";

/**
 * Where a user question is answered from:
 * - `local`: anything needing information beyond the model's own knowledge —
 *   the user's files, chats, mail, or any fact the model cannot reliably recall,
 *   recent, specific, or changing public facts included. Local retrieval runs
 *   first; web search is left to the agent after the fast answer, when local
 *   evidence falls short.
 * - `organize`: the user wants files or folders acted on — move, rename in
 *   bulk, rename by content, convert, sort, tidy — rather than information
 *   found. No retrieval and no verification: one tool-capable turn does it.
 * - `extra`: neither of the other branches, and no search of the user's own
 *   records is needed first: the request is answerable from the model's own
 *   knowledge (stable general knowledge, definitions, simple reasoning, small
 *   talk), or it is a task for the agent's tools that is not file organization
 *   and not AutoRAG configuration (commit, PR, build, write code, run a
 *   script). No retrieval, decomposition, fast answer, follow-up check, or
 *   verification: one tool-capable turn does it, with every tool available.
 * - `config`: the user wants to inspect or change AutoRAG's own settings
 *   (model, providers, datasources). Offered only when the host enables
 *   agent self-configuration.
 */
export type QueryRoute = "local" | "organize" | "extra" | "config";

/** Jev question id of the intrinsic-knowledge / local branch. */
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
 * Minimum Jev branch probability required to leave local search. A non-local
 * branch (`organize`, `extra`, `config`) at or below this floor — or one Jev
 * reported without a probability — keeps the run on local search, the
 * corpus-grounded default, with a `query-route-fallback` diagnostic naming the
 * floor as the reason (the rejected branch's probability is still reported as
 * `routeProbability`). Without the floor a weakly-favored non-local verdict
 * skips retrieval entirely, so a question that depends on the user's own
 * records can be answered from general knowledge instead.
 *
 * The floor does not weaken a real non-local verdict: the fallback is a
 * full local plan (datasource selection and decomposition still run). Removed
 * by #1995 (982e464d) together with the intrinsic-knowledge branch, restored
 * here because the reasons above did not go away.
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
	local: "Answering needs additional information beyond the assistant's intrinsic knowledge: private information only the user can access (files on their computer, their chats such as Discord, KakaoTalk, Slack, their email, notes, calendar, or history), documents, or any fact that may be recent, specific, niche, or changing (current events, recent releases, prices, versions). Prefer local whenever the question refers to the user's own life, situation, plans, or records — questions phrased with words like 'my', 'I', 'we', 'our', a named friend, family member, or colleague, or 'my case/hearing/appointment/routine' — even when a generic answer would also be possible. When unsure whether intrinsic knowledge suffices, choose local.",
	extra: "None of the other options fits and no search of the user's own records is needed first. Either the request is answerable without looking anything up — stable common knowledge, a definition, simple reasoning, or small talk — or it is a task for the assistant's tools that is not file organization and not AutoRAG configuration: committing code, pushing a branch, opening a pull request, running tests or a build, installing a package, running a script, writing or editing code. Programming work is extra even when the code names the user's files or folders ('write a script that lowercases every PNG', 'add this to my parser'): the assistant is asked to produce or change code, not to act on those files now. So is a request to run something (tests, a build, a script, a command) and report the outcome, even when what it runs lives in the user's own repository: running it is a task for the tool, not a search of the user's records. Never extra when the request refers to the user's own life, files, records, chats, calendar, or history (e.g. 'my', 'I', 'our', 'my girlfriend', 'my case', 'my routine'), and never extra when the answer depends on facts that may be recent, specific, or changing (current events, recent releases, prices, versions): search those locally instead. If any other option fits, do not choose extra.",
	organize:
		"The user asks the assistant to DO something to files or folders instead of finding information: move or copy files into folders, tidy or sort a folder, bulk or pattern rename (for example by date or number), rename files according to what their content says, convert files to another format, create folder structures, group files by type or topic, archive, or move to the trash ('정리해줘', '폴더 이동', '파일 이름 일괄 변경', '내용에 맞게 파일명 바꿔줘', 'convert these to PDF', 'sort my Downloads folder'). It is about acting on files and folders, never about looking up what a document says or where something is. A question that only asks to find or read a file's content is local, not organize.",
};

const CONFIG_ROUTE_OPTION =
	"The user asks to view, change, or test the settings of this AutoRAG agent itself: its model, model providers, API-key environment variables, Jev routing, search roots, or datasource setup (config.json). It also covers everything about the assistant's file-triggered automations (자동화, 폴더 감시, 파일 기반 규칙): creating, listing, showing status or the last run of, editing, testing, deleting, or turning on or off the automation rules and the background daemon that run a command or an AutoRAG task by themselves whenever files in a folder are created, changed, or deleted (for example 'when a PDF lands in Downloads, summarize it', '새 파일이 생기면 자동으로 ... 해줘', 'watch this folder', 'show my automations', 'turn the folder-watch daemon off'). It is about configuring the assistant and its background rules, not about the content of the user's documents.";

const BASE_ROUTE_QUESTION = pick("How must the assistant handle this user question?", BASE_ROUTE_OPTIONS);
const SELF_CONFIG_ROUTE_QUESTION = pick("How must the assistant handle this user question?", {
	...BASE_ROUTE_OPTIONS,
	config: CONFIG_ROUTE_OPTION,
});

const DECOMPOSE_QUESTION = check(
	"Does answering this question need several separate search queries instead of one search for the question as written?",
	{
		true: "It combines several sub-questions, compares items, or needs multiple independent facts confirmed, so one search would miss parts of it.",
		false: "A single search for the question as written would already find the answer.",
	},
);

export interface QueryRouteDecision {
	/** Most probable branch; {@link FALLBACK_QUERY_ROUTE} when Jev gave none or its non-local branch missed {@link NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD}. */
	readonly route: QueryRoute;
	/** True when Jev judged the question needs decomposition; only a `local` route acts on it. */
	readonly decompose: boolean;
	/** Probability Jev gave the winning branch, including a rejected non-local branch whose floor moved the route to local. */
	readonly routeProbability?: number;
	/** Jev's P(decomposition needed), when it answered. */
	readonly decomposeProbability?: number;
	/** Why the decision fell back instead of following Jev; absent when Jev decided. */
	readonly fallbackReason?: string;
	/**
	 * True when a non-local branch won but did not clear
	 * {@link NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD}: `route` is `local` and the
	 * full local plan still runs (datasource selection and decomposition),
	 * unlike a fallback from an unavailable or unusable verdict, which keeps a
	 * bare single-query local search.
	 */
	readonly floorApplied?: boolean;
}

function isQueryRoute(value: unknown, selfConfig: boolean): value is QueryRoute {
	return value === "local" || value === "organize" || value === "extra" || (selfConfig && value === "config");
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
 * Ask Jev, in one batched call about the user question, which branch handles it
 * and whether it needs decomposition. The most probable branch wins once it
 * clears {@link NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD}: a weakly-favored or
 * probability-less `organize`, `extra`, or `config` verdict keeps the local
 * default and still runs the full local plan (`floorApplied`). Never throws: a
 * missing credential, an unreachable backend, or an unusable verdict falls back
 * to a single local search, with the reason recorded.
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
	const decomposeWanted =
		decomposeProbability !== undefined && decomposeProbability >= DECOMPOSE_PROBABILITY_THRESHOLD;
	// Leaving local search drops corpus evidence, so it needs a confident
	// branch: a weak or probability-less non-local verdict keeps the local
	// default, which still runs decomposition and the datasource check.
	if (
		best.route !== "local" &&
		(best.probability === undefined || best.probability <= NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD)
	) {
		return {
			route: FALLBACK_QUERY_ROUTE,
			decompose: decomposeWanted,
			floorApplied: true,
			fallbackReason:
				`Jev's ${best.route} route probability (${best.probability ?? "not reported"}) is not above the ` +
				`${NON_LOCAL_ROUTE_PROBABILITY_THRESHOLD} confidence floor; using local search.`,
			...(best.probability !== undefined ? { routeProbability: best.probability } : {}),
			...(decomposeProbability !== undefined ? { decomposeProbability } : {}),
		};
	}
	return {
		route: best.route,
		decompose: best.route === "local" && decomposeWanted,
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
