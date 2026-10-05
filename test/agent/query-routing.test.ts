import { type JevBackend, MockBackend } from "jev-use";
import { describe, expect, it } from "vitest";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { DECOMPOSE_QUESTION_ID, QUERY_ROUTE_QUESTION_ID, routeQuery } from "../../src/agent/query-routing.ts";

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
