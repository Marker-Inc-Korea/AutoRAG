import type { BackendRequest, BackendResponse, JevBackend, RawAnswer } from "jev-use";
import { CITATION_QUESTION_ID_PREFIX } from "../../src/agent/citation-relevance.ts";
import type { JevToolOptions } from "../../src/agent/jev-extension.ts";
import { PERMISSION_GATE_QUESTION_ID } from "../../src/agent/permission-judgment.ts";
import { FOLLOW_UP_QUESTION_ID, QUERY_ROUTE_QUESTION_ID } from "../../src/agent/query-routing.ts";

type Question = BackendRequest["questions"][number];

function answer(question: Question): RawAnswer {
	switch (question.type) {
		case "noul":
			// Always verify the fast answer and keep every citation; everything else is a firm "no":
			// no decomposition, no datasource or web search, no evidence stored in memory.
			if (question.id === FOLLOW_UP_QUESTION_ID || question.id.startsWith(CITATION_QUESTION_ID_PREFIX)) {
				return { answer: 1 };
			}
			// Every permission category is flagged, so the gate never approves a call on its own.
			if (question.id.startsWith("permission_")) return { answer: 1 };
			return { answer: 0 };
		case "choice": {
			const labels = Array.isArray(question.options) ? question.options : Object.keys(question.options ?? {});
			const winner =
				question.id === QUERY_ROUTE_QUESTION_ID
					? "local"
					: question.id === PERMISSION_GATE_QUESTION_ID
						? "deny"
						: (labels[0] ?? "");
			return { answer: winner, distribution: { [winner]: 1 }, confidence: 1 };
		}
		case "score":
			return { answer: 0, confidence: 1 };
	}
}

/**
 * A deterministic decision backend for tests that are not about decision-model behavior. It answers every
 * AutoRAG question the plain way: the local route, no decomposition, verification always, no datasource or
 * web search, every citation kept, nothing added to memory, and no file change approved on its own.
 */
export class BaselineDecisionBackend implements JevBackend {
	readonly name = "baseline";

	async judge(request: BackendRequest): Promise<BackendResponse> {
		return { answers: request.questions.map(answer), model: "baseline-decision" };
	}
}

/** `jev` agent option backed by {@link BaselineDecisionBackend}. */
export function baselineJev(): JevToolOptions {
	return { backend: new BaselineDecisionBackend() };
}
