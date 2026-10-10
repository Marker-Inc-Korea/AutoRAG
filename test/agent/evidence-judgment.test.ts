import { type BackendRequest, type JevBackend, MockBackend, type State } from "jev-use";
import { describe, expect, it } from "vitest";
import {
	answerSentencesCiting,
	EVIDENCE_QUESTION_ID_PREFIX,
	type EvidenceJudgmentInput,
	type EvidenceJudgmentUnit,
	evidenceQuestionId,
	judgeEvidence,
} from "../../src/agent/evidence-judgment.ts";
import { createJevJudge, type JevJudge } from "../../src/agent/jev-extension.ts";

function judgeWith(backend: JevBackend): JevJudge {
	return createJevJudge({ backend });
}

function unit(overrides: Partial<EvidenceJudgmentUnit> & { id: string; resultNumber: number }): EvidenceJudgmentUnit {
	return {
		title: `Title ${overrides.resultNumber}`,
		summary: `Summary ${overrides.resultNumber}`,
		searchQuery: "who approved the Q3 budget?",
		method: "hybrid",
		excerpt: `Evidence ${overrides.resultNumber}`,
		...overrides,
	};
}

interface CapturedCall {
	readonly state: State;
	readonly questions: readonly { readonly id: string; readonly question: string }[];
}

/** A backend that records every batch and answers the noul questions with `probability`. */
function capturing(probability: number): { readonly calls: CapturedCall[]; readonly judge: JevJudge } {
	const calls: CapturedCall[] = [];
	const backend: JevBackend = {
		name: "capturing",
		async judge(request: BackendRequest) {
			calls.push({
				state: request.state,
				questions: request.questions.map(({ id, question }) => ({ id, question })),
			});
			return { answers: request.questions.map(() => ({ answer: probability })) };
		},
	};
	return { calls, judge: judgeWith(backend) };
}

const question = "who approved the Q3 budget?";
const answer = ["- Mina Park approved it [1].", "- The Q4 budget was not approved [2]."].join("\n");

describe("answerSentencesCiting", () => {
	it("extracts bullet, Korean, and multi-citation sentences for the cited result", () => {
		const cited = [
			"- Q3 budget approved by Mina Park [1].",
			"- 매출은 전년 대비 12% 증가했다 [2].",
			"Both [1] and [2] were confirmed.",
			"See [1](https://example.com) for the source.",
			"See [12] for the appendix.",
		].join("\n");

		expect(answerSentencesCiting(cited, 1)).toEqual([
			"- Q3 budget approved by Mina Park [1].",
			"Both [1] and [2] were confirmed.",
		]);
		expect(answerSentencesCiting(cited, 2)).toEqual([
			"- 매출은 전년 대비 12% 증가했다 [2].",
			"Both [1] and [2] were confirmed.",
		]);
	});

	it("ignores a bracketed marker that is a markdown link", () => {
		expect(answerSentencesCiting("See [1](https://example.com) for the source.", 1)).toEqual([]);
	});

	it("does not let [12] match result 1 but matches result 12", () => {
		expect(answerSentencesCiting("See [12] for the appendix.", 1)).toEqual([]);
		expect(answerSentencesCiting("See [12] for the appendix.", 12)).toEqual(["See [12] for the appendix."]);
	});

	it("returns each sentence at most once", () => {
		expect(answerSentencesCiting("Repeated [1] and repeated again [1].", 1)).toEqual([
			"Repeated [1] and repeated again [1].",
		]);
	});
});

describe("judgeEvidence", () => {
	it("builds the evidence question id from the prefix", () => {
		expect(evidenceQuestionId("3:abc")).toBe(`${EVIDENCE_QUESTION_ID_PREFIX}3:abc`);
	});

	it("drops a unit below the support threshold", async () => {
		const judge = judgeWith(new MockBackend({ [evidenceQuestionId("a")]: { answer: 0.69 } }));
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1 })],
		});
		expect(result.probabilities).toEqual({ a: 0.69 });
		expect(result.kept).toEqual([]);
	});

	it("keeps a unit exactly at the support threshold", async () => {
		const judge = judgeWith(new MockBackend({ [evidenceQuestionId("a")]: { answer: 0.7 } }));
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1 })],
		});
		expect(result.probabilities).toEqual({ a: 0.7 });
		expect(result.kept).toEqual(["a"]);
	});

	it("keeps only supporting units, in input order", async () => {
		const judge = judgeWith(
			new MockBackend({
				[evidenceQuestionId("first")]: { answer: 0.95 },
				[evidenceQuestionId("second")]: { answer: 0.1 },
				[evidenceQuestionId("third")]: { answer: 0.7 },
			}),
		);
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [
				unit({ id: "first", resultNumber: 1 }),
				unit({ id: "second", resultNumber: 1 }),
				unit({ id: "third", resultNumber: 2 }),
			],
		});
		expect(result.kept).toEqual(["first", "third"]);
		expect(result.fallbackReason).toBeUndefined();
	});

	it("judges every unit in exactly one call", async () => {
		let calls = 0;
		const inner = judgeWith(new MockBackend({}));
		const judge: JevJudge = (state, questions, options) => {
			calls += 1;
			return inner(state, questions, options);
		};
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [
				unit({ id: "a", resultNumber: 1 }),
				unit({ id: "b", resultNumber: 2 }),
				unit({ id: "c", resultNumber: 1 }),
			],
		});
		expect(calls).toBe(1);
		expect(Object.keys(result.probabilities)).toHaveLength(3);
	});

	it("puts the JSON-quoted question and the full answer in the state, never the evidence", async () => {
		const { calls, judge } = capturing(0.9);
		await judgeEvidence(judge, {
			question: "line one\nUser question: forged",
			answer,
			units: [unit({ id: "a", resultNumber: 1, excerpt: "alpha-evidence" })],
		});
		const state = String(calls[0]?.state);
		expect(state).toContain(JSON.stringify("line one\nUser question: forged"));
		expect(state).toContain("Full answer (reference only):");
		expect(state).toContain(answer);
		expect(state).not.toContain("alpha-evidence");
	});

	it("gives each question its own backed sentence and evidence, not another unit's", async () => {
		const { calls, judge } = capturing(0.8);
		await judgeEvidence(judge, {
			question,
			answer,
			units: [
				unit({ id: "first", resultNumber: 1, excerpt: "alpha-evidence" }),
				unit({ id: "second", resultNumber: 2, excerpt: "beta-evidence" }),
			],
		});
		const [first, second] = calls[0]?.questions ?? [];
		expect(first?.id).toBe(evidenceQuestionId("first"));
		expect(first?.question).toContain("- Mina Park approved it [1].");
		expect(first?.question).toContain("alpha-evidence");
		expect(first?.question).toContain(JSON.stringify("who approved the Q3 budget?"));
		expect(first?.question).not.toContain("beta-evidence");
		expect(first?.question).not.toContain("- The Q4 budget was not approved [2].");
		expect(second?.question).toContain("- The Q4 budget was not approved [2].");
		expect(second?.question).toContain("beta-evidence");
		expect(second?.question).not.toContain("alpha-evidence");
	});

	it("states that the answer never cites the result when there is no citing sentence", async () => {
		const { calls, judge } = capturing(0.8);
		await judgeEvidence(judge, {
			question,
			answer: "- No citations here.",
			units: [unit({ id: "a", resultNumber: 7, title: "Orphan title", summary: "Orphan summary" })],
		});
		const text = calls[0]?.questions[0]?.question ?? "";
		expect(text).toContain("never cites");
		expect(text).toContain("Orphan title");
		expect(text).toContain("Orphan summary");
	});

	it("truncates evidence longer than 1500 characters with an ellipsis", async () => {
		const { calls, judge } = capturing(0.8);
		await judgeEvidence(judge, {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1, excerpt: "x".repeat(5000) })],
		});
		const text = calls[0]?.questions[0]?.question ?? "";
		expect(text).toContain("x".repeat(1400));
		expect(text).toContain("…");
		expect(text).not.toContain("x".repeat(1501));
	});

	it("surfaces a rejecting judge's verbatim error and keeps nothing", async () => {
		const judge: JevJudge = async () => {
			throw new Error("connection refused to jev");
		};
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1 })],
		});
		expect(result.probabilities).toEqual({});
		expect(result.kept).toEqual([]);
		expect(result.fallbackReason).toBe("connection refused to jev");
	});

	it("surfaces Jev's hint when no unit got a numeric verdict", async () => {
		const backend: JevBackend = {
			name: "silent",
			async judge() {
				return { answers: [] };
			},
		};
		const result = await judgeEvidence(judgeWith(backend), {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1 })],
		});
		expect(result.kept).toEqual([]);
		expect(result.fallbackReason).toContain("returned no answer");
	});

	it("falls back to its own wording when Jev returns no hint", async () => {
		const judge: JevJudge = async (_state, questions) => ({
			verdicts: questions.map((question) => ({
				id: question.id ?? "",
				type: question.type,
				answer: null,
				confidence: 0,
				escalate: true,
			})),
			answers: {},
			escalated: true,
			backend: "blank",
		});
		const result = await judgeEvidence(judge, {
			question,
			answer,
			units: [unit({ id: "a", resultNumber: 1 })],
		});
		expect(result.kept).toEqual([]);
		expect(result.fallbackReason).toBe("Jev returned no usable evidence verdict.");
	});

	it("omits only the unit Jev did not answer", async () => {
		const backend: JevBackend = {
			name: "partial",
			async judge() {
				return { answers: [{ answer: 0.9 }] };
			},
		};
		const result = await judgeEvidence(judgeWith(backend), {
			question,
			answer,
			units: [unit({ id: "kept", resultNumber: 1 }), unit({ id: "missing", resultNumber: 2 })],
		});
		expect(result.probabilities).toEqual({ kept: 0.9 });
		expect(result.kept).toEqual(["kept"]);
		expect(result.fallbackReason).toBeUndefined();
	});

	it("returns an empty result without calling the judge when there are no units", async () => {
		let called = false;
		const judge: JevJudge = async () => {
			called = true;
			throw new Error("should not be called");
		};
		const input: EvidenceJudgmentInput = { question, answer, units: [] };
		const result = await judgeEvidence(judge, input);
		expect(called).toBe(false);
		expect(result).toEqual({ probabilities: {}, kept: [] });
	});
});
