import { type JevBackend, MockBackend } from "jev-use";
import { describe, expect, it } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	needsFollowUp,
	QUERY_ROUTE_QUESTION_ID,
	routeQuery,
} from "../../src/agent/query-routing.ts";

function judgeWith(backend: JevBackend) {
	return createJevJudge({ backend });
}

describe("routeQuery (Jev three-way branch + decomposition check)", () => {
	it("routes to the most probable branch even when Jev escalates it as unsure", async () => {
		const decision = await routeQuery(
			judgeWith(
				new MockBackend({
					[QUERY_ROUTE_QUESTION_ID]: {
						answer: "web",
						distribution: { local: 0.3, web: 0.38, direct: 0.32 },
						confidence: 0.2,
					},
					[DECOMPOSE_QUESTION_ID]: { answer: 0.2 },
				}),
			),
			"latest stable Node.js release",
		);
		expect(decision.route).toBe("web");
		expect(decision.decompose).toBe(false);
		expect(decision.fallbackReason).toBeUndefined();
	});

	it("asks for decomposition when Jev's yes-probability reaches one half", async () => {
		const decision = await routeQuery(
			judgeWith(
				new MockBackend({
					[QUERY_ROUTE_QUESTION_ID]: { answer: "local", distribution: { local: 0.9, web: 0.05, direct: 0.05 } },
					[DECOMPOSE_QUESTION_ID]: { answer: 0.5 },
				}),
			),
			"compare the Q3 and Q4 budget memos and who approved each",
		);
		expect(decision.route).toBe("local");
		expect(decision.decompose).toBe(true);
	});

	it("never decomposes a direct answer", async () => {
		const decision = await routeQuery(
			judgeWith(
				new MockBackend({
					[QUERY_ROUTE_QUESTION_ID]: { answer: "direct", distribution: { local: 0.1, web: 0.1, direct: 0.8 } },
					[DECOMPOSE_QUESTION_ID]: { answer: 0.99 },
				}),
			),
			"hi, how are you?",
		);
		expect(decision.route).toBe("direct");
		expect(decision.decompose).toBe(false);
	});

	it("asks Jev the branch and decomposition questions about the user question in one batch", async () => {
		const seen: { state: unknown; ids: string[] }[] = [];
		const backend: JevBackend = {
			name: "recording",
			async judge(request) {
				seen.push({ state: request.state, ids: request.questions.map((question) => question.id) });
				return { answers: [{ answer: "local", confidence: 0.9 }, { answer: 0.1 }] };
			},
		};
		await routeQuery(judgeWith(backend), "where is the signed lease?");
		expect(seen).toHaveLength(1);
		expect(String(seen[0]?.state)).toContain("where is the signed lease?");
		expect(seen[0]?.ids).toEqual([QUERY_ROUTE_QUESTION_ID, DECOMPOSE_QUESTION_ID]);
	});

	it("falls back to local search without decomposition when Jev is unreachable", async () => {
		const backend: JevBackend = {
			name: "down",
			async judge() {
				throw new Error("connection refused");
			},
		};
		const decision = await routeQuery(judgeWith(backend), "where is the signed lease?");
		expect(decision.route).toBe("local");
		expect(decision.decompose).toBe(false);
		expect(decision.fallbackReason).toMatch(/connection refused/u);
	});

	it("falls back to local search when no Jev credential resolves", async () => {
		const decision = await routeQuery(createJevJudge({ backend: "typesafe", env: {} }), "where is the signed lease?");
		expect(decision.route).toBe("local");
		expect(decision.fallbackReason).toMatch(/TYPESAFE_API_KEY/u);
	});
});

describe("needsFollowUp (Jev check after emit_fast_answer)", () => {
	const fastAnswer = "- The Q3 budget was approved by Mina Park on 2026-07-02 [1].";

	it("ends the run when Jev says the fast answer needs no correction, clarification, or further research", async () => {
		const decision = await needsFollowUp(
			judgeWith(new MockBackend({ [FOLLOW_UP_QUESTION_ID]: { answer: 0.1 } })),
			"who approved the Q3 budget?",
			fastAnswer,
		);
		expect(decision.followUp).toBe(false);
		expect(decision.probability).toBe(0.1);
		expect(decision.fallbackReason).toBeUndefined();
	});

	it("continues to verification when Jev's probability reaches one half", async () => {
		const decision = await needsFollowUp(
			judgeWith(new MockBackend({ [FOLLOW_UP_QUESTION_ID]: { answer: 0.5 } })),
			"who approved the Q3 budget?",
			"- The approver is not stated in the available evidence.",
		);
		expect(decision.followUp).toBe(true);
	});

	it("asks one noul about the question together with the fast answer", async () => {
		const seen: { state: unknown; questions: { id: string; type: string }[] }[] = [];
		const backend: JevBackend = {
			name: "recording",
			async judge(request) {
				seen.push({ state: request.state, questions: request.questions.map(({ id, type }) => ({ id, type })) });
				return { answers: [{ answer: 0.2 }] };
			},
		};
		await needsFollowUp(judgeWith(backend), "who approved the Q3 budget?", fastAnswer);
		expect(seen).toHaveLength(1);
		expect(seen[0]?.questions).toEqual([{ id: FOLLOW_UP_QUESTION_ID, type: "noul" }]);
		expect(String(seen[0]?.state)).toContain("who approved the Q3 budget?");
		expect(String(seen[0]?.state)).toContain(fastAnswer);
	});

	it("keeps verifying when Jev is unreachable", async () => {
		const backend: JevBackend = {
			name: "down",
			async judge() {
				throw new Error("connection refused");
			},
		};
		const decision = await needsFollowUp(judgeWith(backend), "q", fastAnswer);
		expect(decision.followUp).toBe(true);
		expect(decision.fallbackReason).toMatch(/connection refused/u);
	});

	it("keeps verifying when no Jev credential resolves", async () => {
		const decision = await needsFollowUp(createJevJudge({ backend: "typesafe", env: {} }), "q", fastAnswer);
		expect(decision.followUp).toBe(true);
		expect(decision.fallbackReason).toMatch(/TYPESAFE_API_KEY/u);
	});
});
