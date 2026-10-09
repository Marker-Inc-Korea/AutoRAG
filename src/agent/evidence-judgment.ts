import { check, type Question, type Verdict } from "jev-use";
import { EVIDENCE_SUPPORT_THRESHOLD } from "../memory/judged-evidence.ts";
import type { JevJudge } from "./jev-extension.ts";

/**
 * Retrieval memory stores only evidence that really supports the answer, so
 * every cited piece is put back to Jev: does this evidence support the
 * sentence it backs, and is answering the user's question impossible to trust
 * without it? One batched call covers all of a run's evidence at once.
 *
 * The judgment is scoped per evidence and never merges units: a run can cite
 * one result with several evidence pieces, each backing a different sentence,
 * and only some of them needed.
 */

/** One evidence a final answer cites. */
export interface EvidenceJudgmentUnit {
	/** Unique within one call, e.g. `${resultNumber}:${stableEvidenceId}`. */
	readonly id: string;
	/** The cited result [n] this evidence backs. */
	readonly resultNumber: number;
	/** That result's title. */
	readonly title: string;
	/** That result's summary. */
	readonly summary: string;
	/** The query that surfaced the evidence. */
	readonly searchQuery: string;
	/** Retrieval method that surfaced it. */
	readonly method: string;
	/** The evidence text. */
	readonly excerpt: string;
}

export interface EvidenceJudgmentInput {
	readonly question: string;
	readonly answer: string;
	readonly units: readonly EvidenceJudgmentUnit[];
}

export interface EvidenceJudgment {
	/** Jev's P(evidence supports the question), per unit id, for every unit Jev answered. */
	readonly probabilities: Readonly<Record<string, number>>;
	/** Unit ids with probability >= {@link EVIDENCE_SUPPORT_THRESHOLD}, in input order. */
	readonly kept: readonly string[];
	/** Set when Jev gave no usable verdict for ANY unit (judge threw, or no numeric verdicts); verbatim error / hint text. Absent when Jev decided. */
	readonly fallbackReason?: string;
}

/** Jev question id of one evidence support check. */
export const EVIDENCE_QUESTION_ID_PREFIX = "evidence:";

/** Jev question id of the support check for one evidence unit. */
export function evidenceQuestionId(unitId: string): string {
	return `${EVIDENCE_QUESTION_ID_PREFIX}${unitId}`;
}

/** Kept fallback wording when Jev answered, but with no usable number. */
const NO_VERDICT_FALLBACK = "Jev returned no usable evidence verdict.";

/**
 * Evidence text longer than this cannot add discrimination to a single yes/no
 * and would crowd the batch, so it is bounded.
 */
const MAX_EVIDENCE_CHARACTERS = 1500;

/**
 * The shared instruction in the state. It keeps every question honest: the
 * full answer is context for the judge, not the thing being judged, because a
 * correct-sounding neighbor sentence must not lift an unrelated evidence.
 */
const STATE_INSTRUCTION = [
	"Each question asks whether one piece of evidence really supports answering the user's question.",
	"Judge only that evidence against the sentence it backs; the full answer is shown for reference only.",
].join(" ");

const EVIDENCE_QUESTION = check(
	"Does the evidence directly support the cited sentence and is it needed to answer the user's question?",
	{
		true: "The evidence states or confirms what the cited sentence claims about the user's question, and leaving it out would leave that claim unsupported.",
		false: "The evidence is unrelated, only loosely related, redundant noise, or contradicts the cited sentence.",
	},
);

const SENTENCE_TERMINATORS: Record<string, true> = {
	".": true,
	"!": true,
	"?": true,
	"。": true,
	"！": true,
	"？": true,
};

/**
 * The sentences/lines of `answer` that cite result `number` with a bracketed
 * `[number]` marker (not a markdown link `[n](...)`). Splits on sentence
 * terminators followed by whitespace or end, so Korean and English bullet
 * lists both survive, and dedupes so one sentence backs a result once.
 */
export function answerSentencesCiting(answer: string, number: number): string[] {
	const marker = `[${number}]`;
	const seen = new Set<string>();
	const cited: string[] = [];
	for (const line of answer.split(/\r?\n/u)) {
		for (const sentence of splitSentences(line)) {
			const trimmed = sentence.trim();
			if (trimmed === "" || seen.has(trimmed) || !citesResult(trimmed, marker)) continue;
			seen.add(trimmed);
			cited.push(trimmed);
		}
	}
	return cited;
}

/** Split one line into sentences, keeping each terminator with its sentence. */
function splitSentences(line: string): string[] {
	const sentences: string[] = [];
	let start = 0;
	for (let index = 0; index < line.length; index += 1) {
		const character = line[index];
		if (character === undefined || SENTENCE_TERMINATORS[character] !== true) continue;
		const next = line[index + 1];
		if (next !== undefined && !/\s/u.test(next)) continue;
		sentences.push(line.slice(start, index + 1));
		start = index + 1;
	}
	if (start < line.length) sentences.push(line.slice(start));
	return sentences;
}

/** True when the sentence cites the result, skipping `[n](...)` markdown-link markers. */
function citesResult(sentence: string, marker: string): boolean {
	let index = sentence.indexOf(marker);
	while (index !== -1) {
		if (sentence[index + marker.length] !== "(") return true;
		index = sentence.indexOf(marker, index + 1);
	}
	return false;
}

function boundExcerpt(excerpt: string): string {
	if (excerpt.length <= MAX_EVIDENCE_CHARACTERS) return excerpt;
	return `${excerpt.slice(0, MAX_EVIDENCE_CHARACTERS - 1)}…`;
}

function describeState(input: EvidenceJudgmentInput): string {
	return [
		STATE_INSTRUCTION,
		`User question: ${JSON.stringify(input.question)}`,
		"Full answer (reference only):",
		input.answer,
	].join("\n");
}

function evidenceQuestion(unit: EvidenceJudgmentUnit, input: EvidenceJudgmentInput): Question {
	const sentences = answerSentencesCiting(input.answer, unit.resultNumber);
	const backed =
		sentences.length > 0
			? `the answer sentence it backs: ${sentences.map((sentence) => JSON.stringify(sentence)).join(" ")}`
			: `the claim that the answer never cites result [${unit.resultNumber}] (title ${JSON.stringify(unit.title)}, summary ${JSON.stringify(unit.summary)})`;
	return {
		...EVIDENCE_QUESTION,
		id: evidenceQuestionId(unit.id),
		question: [
			`Does the evidence below directly support ${backed} and is it needed to answer the user's question?`,
			`Search query that surfaced it: ${JSON.stringify(unit.searchQuery)}`,
			`Retrieval method: ${JSON.stringify(unit.method)}`,
			`Evidence: ${JSON.stringify(boundExcerpt(unit.excerpt))}`,
		].join("\n"),
	};
}

/**
 * Ask Jev, in one batched call, whether each cited evidence supports its backed
 * sentence. Never throws: a rejecting judge keeps nothing and reports why, and
 * a unit Jev left unanswered is simply absent from `probabilities`.
 */
export async function judgeEvidence(judge: JevJudge, input: EvidenceJudgmentInput): Promise<EvidenceJudgment> {
	if (input.units.length === 0) return { probabilities: {}, kept: [] };
	let verdicts: readonly Verdict[];
	try {
		verdicts = (
			await judge(
				describeState(input),
				input.units.map((unit) => evidenceQuestion(unit, input)),
			)
		).verdicts;
	} catch (error) {
		return { probabilities: {}, kept: [], fallbackReason: error instanceof Error ? error.message : String(error) };
	}
	const byId: Record<string, Verdict> = {};
	for (const verdict of verdicts) byId[verdict.id] = verdict;
	const probabilities: Record<string, number> = {};
	const kept: string[] = [];
	let firstHint: string | undefined;
	for (const unit of input.units) {
		const verdict = byId[evidenceQuestionId(unit.id)];
		if (typeof verdict?.answer !== "number") {
			firstHint ??= verdict?.hint;
			continue;
		}
		probabilities[unit.id] = verdict.answer;
		if (verdict.answer >= EVIDENCE_SUPPORT_THRESHOLD) kept.push(unit.id);
	}
	if (Object.keys(probabilities).length === 0) {
		return { probabilities, kept, fallbackReason: firstHint ?? NO_VERDICT_FALLBACK };
	}
	return { probabilities, kept };
}
