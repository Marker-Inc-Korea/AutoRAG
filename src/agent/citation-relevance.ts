import { check, type Question, type Verdict } from "jev-use";
import type { CitedChunk } from "./answer-citations.ts";
import type { JevJudge } from "./jev-extension.ts";

/**
 * The model sometimes cites a chunk that has nothing to do with the sentence it
 * backs or with what the user asked: a table of contents, a navigation bar, a
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
	"Each question asks whether one cited piece of evidence is relevant to the answer sentence that cites it or to what the user asked. Judge only that evidence. The user's messages are the whole conversation so far, oldest first; the latest one is the question being answered.";

const RELEVANCE_QUESTION = check("Is the evidence relevant to the cited sentence or to what the user asked?", {
	true: "The evidence contains information on the same subject as the cited sentence or what the user asked; it may back only part of the claim.",
	false: "The evidence has no information on the cited sentence's subject or what the user asked: a table of contents, navigation, header or footer, copyright or boilerplate, a generic introduction, or another topic.",
});

export interface CitationRelevanceInput {
	/** Everything the user typed in the conversation, oldest first; the last is the question being answered. */
	readonly userMessages: readonly string[];
	/** The chunks the answer cites, with the sentences citing each. */
	readonly chunks: readonly CitedChunk[];
}

export interface CitationRelevance {
	/** {@link CitedChunk.key} of every chunk Jev found unrelated. */
	readonly rejected: ReadonlySet<string>;
	/** Set when Jev gave no usable verdict for ANY chunk; nothing was rejected. Absent when Jev decided. */
	readonly fallbackReason?: string;
}

/**
 * Ask Jev, in one batched call, whether each cited chunk is relevant to the
 * sentences that cite it and to what the user asked. Never throws: a rejecting
 * judge rejects nothing and reports why, and a chunk Jev left unanswered is kept.
 */
export async function judgeCitationRelevance(
	judge: JevJudge,
	input: CitationRelevanceInput,
): Promise<CitationRelevance> {
	const questions: Question[] = [];
	const keyById = new Map<string, string>();
	for (const chunk of input.chunks) {
		const text = chunk.content.trim();
		// Evidence that is only a path (a file the model located, not read) has no text to judge.
		if (text === "" || text === chunk.source) continue;
		const backed =
			chunk.sentences.length > 0
				? `the answer sentence it backs: ${chunk.sentences.map((sentence) => JSON.stringify(sentence)).join(" ")}`
				: "the answer";
		const bounded = text.length <= MAX_CHUNK_CHARACTERS ? text : `${text.slice(0, MAX_CHUNK_CHARACTERS - 1)}…`;
		const id = `${CITATION_QUESTION_ID_PREFIX}${questions.length}`;
		keyById.set(id, chunk.key);
		questions.push({
			...RELEVANCE_QUESTION,
			id,
			question: [
				`Is the evidence below relevant to ${backed} or to what the user asked?`,
				`Evidence: ${JSON.stringify(bounded)}`,
			].join("\n"),
		});
	}
	if (questions.length === 0) return { rejected: new Set() };

	let verdicts: readonly Verdict[];
	try {
		verdicts = (
			await judge(
				[
					STATE_INSTRUCTION,
					"User messages, oldest first:",
					...input.userMessages.map((message, index) => `${index + 1}. ${JSON.stringify(message)}`),
				].join("\n"),
				questions,
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
	for (const [id, key] of keyById) {
		const verdict = byId[id];
		if (typeof verdict?.answer !== "number") {
			firstHint ??= verdict?.hint;
			continue;
		}
		decided += 1;
		if (verdict.answer < CITATION_RELEVANCE_FLOOR) rejected.add(key);
	}
	if (decided === 0) return { rejected, fallbackReason: firstHint ?? NO_VERDICT_FALLBACK };
	return { rejected };
}
