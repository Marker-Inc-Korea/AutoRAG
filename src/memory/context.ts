import type { Embedder } from "../embedding-runtime/gateway-embedder.ts";
import type { JudgedEvidenceRecord } from "./judged-evidence.ts";
import type { RetrievalInsight, RetrievalMemory } from "./memory.ts";
import { CURRENT_CONVERSATION_RECORD_LIMIT, renderMemoryContext } from "./renderer.ts";
import { findSimilarQuestions, SIMILAR_QUESTION_LIMIT, type SimilarQuestion } from "./similar-queries.ts";

export interface MemoryContextOptions {
	readonly conversationId: string;
	/**
	 * Per-record gate applied to both the current conversation's evidence and
	 * the past evidence considered for similarity. Memory is shared across
	 * workspaces, so the agent uses this to drop evidence from datasources that
	 * are not configured for the current run.
	 */
	readonly isVisible?: (record: JudgedEvidenceRecord) => boolean;
	/**
	 * Per-insight gate applied to the matching long-term insights. Insights name
	 * sources and methods, so the same datasource-scope filter drops the ones
	 * whose sources the current run cannot read.
	 */
	readonly isInsightVisible?: (insight: RetrievalInsight) => boolean;
	readonly embedder?: Embedder;
	readonly vectorStorePath?: string;
}

export interface MemoryContext {
	/** The advisory markdown the librarian agent receives. */
	readonly text: string;
	/** True when there is nothing to show. */
	readonly empty: boolean;
	/** Current-conversation evidence after {@link MemoryContextOptions.isVisible}, before the render cap; reusable by the caller. */
	readonly current: readonly JudgedEvidenceRecord[];
	/**
	 * The same similarity search kept unfiltered, including questions from the
	 * current conversation. The agent uses it to hint at datasources, so a
	 * follow-up in the same session still finds the earlier question.
	 */
	readonly related: readonly SimilarQuestion[];
	/** Similar questions from other conversations only (current-conversation records removed); what is rendered. */
	readonly similar: readonly SimilarQuestion[];
	/** Records rendered from the current conversation (capped at {@link CURRENT_CONVERSATION_RECORD_LIMIT}). */
	readonly currentCount: number;
	/** Similar past questions found. */
	readonly similarCount: number;
	readonly insightCount: number;
	/** Why semantic matching was skipped, verbatim from {@link findSimilarQuestions}. */
	readonly fallbackReason?: string;
}

/**
 * Gathers this conversation's judged evidence, similar past questions, and
 * matching long-term insights into one advisory context. Never throws:
 * {@link findSimilarQuestions} degrades to keywords and reports why.
 */
export async function loadMemoryContext(
	memory: RetrievalMemory,
	query: string,
	options: MemoryContextOptions,
): Promise<MemoryContext> {
	const { isVisible } = options;
	const currentConversation = memory.getConversationEvidence(options.conversationId);
	const pastEvidence = memory.getJudgedEvidence();
	const current = isVisible === undefined ? currentConversation : currentConversation.filter(isVisible);
	const past = isVisible === undefined ? pastEvidence : pastEvidence.filter(isVisible);
	// The current conversation's questions may occupy result slots, so widen the
	// search by how many distinct current questions could show up.
	const currentQuestions = new Set(current.map((record) => record.question)).size;
	const similarResult = await findSimilarQuestions(past, query, {
		limit: SIMILAR_QUESTION_LIMIT + currentQuestions,
		...(options.embedder !== undefined ? { embedder: options.embedder } : {}),
		...(options.vectorStorePath !== undefined ? { vectorStorePath: options.vectorStorePath } : {}),
	});
	const related = similarResult.questions.slice(0, SIMILAR_QUESTION_LIMIT);
	const similar = similarResult.questions
		.map((question) => ({
			...question,
			records: question.records.filter((record) => record.conversationId !== options.conversationId),
		}))
		.filter((question) => question.records.length > 0)
		.slice(0, SIMILAR_QUESTION_LIMIT);
	const matchedInsights = memory.getInsights(query);
	const insights =
		options.isInsightVisible === undefined ? matchedInsights : matchedInsights.filter(options.isInsightVisible);
	const text = renderMemoryContext({ current, similar, insights });
	return {
		text,
		empty: current.length === 0 && similar.length === 0 && insights.length === 0,
		current,
		related,
		similar,
		currentCount: Math.min(current.length, CURRENT_CONVERSATION_RECORD_LIMIT),
		similarCount: similar.length,
		insightCount: insights.length,
		...(similarResult.fallbackReason !== undefined ? { fallbackReason: similarResult.fallbackReason } : {}),
	};
}
