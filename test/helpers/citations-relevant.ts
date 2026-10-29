import type { BackendRequest, JevBackend } from "jev-use";
import { CITATION_QUESTION_ID_PREFIX } from "../../src/agent/citation-relevance.ts";

/**
 * Answers every citation-relevance question as relevant and delegates the rest,
 * so tests that are not about citation filtering never lose a chunk to the
 * MockBackend's hash-derived answers for unscripted questions.
 */
export function citationsRelevant(inner: JevBackend): JevBackend {
	const isCitation = (id: string | undefined): boolean => id?.startsWith(CITATION_QUESTION_ID_PREFIX) === true;
	return {
		name: inner.name,
		async judge(request: BackendRequest) {
			if (!request.questions.some((question) => isCitation(question.id))) return inner.judge(request);
			const rest = await inner.judge({
				...request,
				questions: request.questions.filter((question) => !isCitation(question.id)),
			});
			let next = 0;
			return {
				...rest,
				answers: request.questions.map((question) =>
					isCitation(question.id) ? { answer: 0.9 } : (rest.answers[next++] ?? { answer: 0 }),
				),
			};
		},
	};
}
