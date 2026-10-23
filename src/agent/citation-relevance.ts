import { check, type Question, type Verdict } from "jev-use";
import { citedChunkKey } from "./answer-citations.ts";
import { answerSentencesCiting } from "./evidence-judgment.ts";
import type { JevJudge } from "./jev-extension.ts";
import type { AutoRAGResultsDetails } from "./results.ts";

/**
 * The model sometimes cites a chunk that has nothing to do with the sentence it
 * backs or with the question: a table of contents, a navigation bar, a
 * copyright footer, a generic introduction. Before the answer is returned,
 * every cited chunk is put to Jev as one relevance question, and a chunk Jev
 * finds unrelated is removed from the answer's citations.
 *
 * This is deliberately weaker than the memory judgment in `evidence-judgment.ts`
 * ("does it directly support, and is it needed"): that gate decides what is
 * worth remembering and rejects any chunk that does not state the claim. Here a
 * chunk that is merely on topic, or backs only part of the claim, must stay, so
 * the only thing rejected is evidence with no information on the subject at all.
 */

/** A chunk is kept when Jev's P(relevant) is at or above this; below it the chunk is rejected. */
export const CITATION_RELEVANCE_FLOOR = 0.3;

/** Jev question id prefix of one citation relevance check. */
export const CITATION_QUESTION_ID_PREFIX = "citation:";

/** Evidence text longer than this adds no discrimination to one yes/no and would crowd the batch. */
const MAX_CHUNK_CHARACTERS = 1500;

const NO_VERDICT_FALLBACK = "Jev returned no usable citation relevance verdict.";

const STATE_INSTRUCTION =
	"Each question asks whether one cited piece of evidence is relevant to the answer sentence that cites it or to the user's question. Judge only that evidence; the full answer is shown for reference only.";

const RELEVANCE_QUESTION = check("Is the evidence relevant to the cited sentence or the user's question?", {
	true: "The evidence contains information on the same subject as the cited sentence or the user's question; it may back only part of the claim.",
	false: "The evidence has no information on the cited sentence's subject or the user's question: a table of contents, navigation, header or footer, copyright or boilerplate, a generic introduction, or another topic.",
});

export interface CitationRelevanceInput {
	readonly question: string;
	/** The derived answer: its `[n]` markers and `mapping` carry the cited chunks. */
	readonly details: AutoRAGResultsDetails;
}

export interface CitationRelevance {
	/** {@link citedChunkKey} of every chunk Jev found unrelated. */
	readonly rejected: ReadonlySet<string>;
	/** Set when Jev gave no usable verdict for ANY chunk; nothing was rejected. Absent when Jev decided. */
	readonly fallbackReason?: string;
}

interface CitationUnit {
	readonly key: string;
	readonly question: Question;
}

/**
 * Ask Jev, in one batched call, whether each cited chunk is relevant to the
 * sentence that cites it. Never throws: a rejecting judge rejects nothing and
 * reports why, and a chunk Jev left unanswered is kept.
 */
export async function judgeCitationRelevance(
	judge: JevJudge,
	input: CitationRelevanceInput,
): Promise<CitationRelevance> {
	const units: CitationUnit[] = [];
	const seen = new Set<string>();
	for (const entry of input.details.mapping) {
		const sentences = answerSentencesCiting(input.details.answer, entry.number);
		const backed =
			sentences.length > 0
				? `the answer sentence it backs: ${sentences.map((sentence) => JSON.stringify(sentence)).join(" ")}`
				: "the answer";
		for (const ref of entry.evidenceRefs) {
			const text = (ref.excerpt ?? ref.content ?? "").trim();
			// Evidence that is only a path (a file the model located, not read) has no text to judge.
			if (text === "" || text === ref.source) continue;
			const key = citedChunkKey(ref);
			if (seen.has(key)) continue;
			seen.add(key);
			const bounded = text.length <= MAX_CHUNK_CHARACTERS ? text : `${text.slice(0, MAX_CHUNK_CHARACTERS - 1)}…`;
			units.push({
				key,
				question: {
					...RELEVANCE_QUESTION,
					id: `${CITATION_QUESTION_ID_PREFIX}${units.length}`,
					question: [
						`Is the evidence below relevant to ${backed} or to the user's question?`,
						`Evidence: ${JSON.stringify(bounded)}`,
					].join("\n"),
				},
			});
		}
	}
	if (units.length === 0) return { rejected: new Set() };

	let verdicts: readonly Verdict[];
	try {
		verdicts = (
			await judge(
				[
					STATE_INSTRUCTION,
					`User question: ${JSON.stringify(input.question)}`,
					"Full answer (reference only):",
					input.details.answer,
				].join("\n"),
				units.map((unit) => unit.question),
			)
		).verdicts;
	} catch (error) {
		return { rejected: new Set(), fallbackReason: error instanceof Error ? error.message : String(error) };
	}

	const byId: Record<string, Verdict> = {};
	for (const verdict of verdicts) byId[verdict.id] = verdict;
	const rejected = new Set<string>();
	let decided = 0;
	let firstHint: string | undefined;
	for (const unit of units) {
		const verdict = byId[unit.question.id ?? ""];
		if (typeof verdict?.answer !== "number") {
			firstHint ??= verdict?.hint;
			continue;
		}
		decided += 1;
		if (verdict.answer < CITATION_RELEVANCE_FLOOR) rejected.add(unit.key);
	}
	if (decided === 0) return { rejected, fallbackReason: firstHint ?? NO_VERDICT_FALLBACK };
	return { rejected };
}
