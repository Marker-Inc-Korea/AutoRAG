import { describe, expect, it } from "vitest";
import {
	createJevTool,
	JEV_TOOL_NAME,
	type JevEvaluation,
	type JevEvaluationInput,
	type JevEvaluator,
	JevInputError,
} from "../../src/jev/index.ts";

const EVALUATION: JevEvaluation = {
	provider: "openrouter",
	model: "jev-1.13",
	answers: {
		is_urgent: { type: "noul", noul: 0.95 },
		department: {
			type: "choice",
			choice: "billing",
			confidence: 0.8,
			probabilities: { billing: 0.8, technical: 0.2 },
		},
		frustration: { type: "score", score: 1.04, probabilities: { "0": 0.1, "1": 0.75, "2": 0.15 } },
	},
	usage: { inputTokens: 100, outputTokens: 10, totalTokens: 110, costUsd: 0.00012 },
};

function recordingEvaluator(seen: JevEvaluationInput[]): JevEvaluator {
	return {
		async evaluate(input) {
			seen.push(input);
			return EVALUATION;
		},
	};
}

describe("jev tool", () => {
	it("maps every question type, forwards the state, and formats the answers", async () => {
		const seen: JevEvaluationInput[] = [];
		const tool = createJevTool(recordingEvaluator(seen));
		const result = await tool.execute("call-1", {
			label: "inbox triage",
			state: { message: "Help! Payouts have failed for three days." },
			questions: {
				is_urgent: { type: "noul", instructions: "Does this convey urgency?" },
				department: {
					type: "choice",
					instructions: "Which team should handle this?",
					criteria: { billing: "Payments", technical: "Bugs" },
				},
				frustration: {
					type: "score",
					instructions: "How frustrated is the sender?",
					criteria: ["Calm", "Frustrated", "Very angry"],
				},
			},
		});

		expect(tool.name).toBe(JEV_TOOL_NAME);
		expect(seen).toHaveLength(1);
		expect(seen[0]?.state).toEqual({ message: "Help! Payouts have failed for three days." });
		expect(seen[0]?.questions).toEqual({
			is_urgent: { type: "noul", instructions: "Does this convey urgency?" },
			department: {
				type: "choice",
				instructions: "Which team should handle this?",
				criteria: { billing: "Payments", technical: "Bugs" },
			},
			frustration: {
				type: "score",
				instructions: "How frustrated is the sender?",
				criteria: ["Calm", "Frustrated", "Very angry"],
			},
		});

		const [firstContent] = result.content;
		const text = firstContent?.type === "text" ? (firstContent.text ?? "") : "";
		expect(text).toContain("openrouter/jev-1.13");
		expect(text).toContain("inbox triage");
		expect(text).toContain("noul=0.950");
		expect(text).toContain("choice=billing");
		expect(text).toContain("110 tokens");
		expect(result.details).toBe(EVALUATION);
	});

	it("treats an omitted state as null", async () => {
		const seen: JevEvaluationInput[] = [];
		const tool = createJevTool(recordingEvaluator(seen));
		await tool.execute("call-2", { questions: { ok: { type: "noul", instructions: "Is it fine?" } } });
		expect(seen[0]?.state).toBeNull();
	});

	it("forwards the abort signal", async () => {
		const seen: JevEvaluationInput[] = [];
		const tool = createJevTool(recordingEvaluator(seen));
		const controller = new AbortController();
		await tool.execute(
			"call-3",
			{ questions: { ok: { type: "noul", instructions: "Is it fine?" } } },
			controller.signal,
		);
		expect(seen[0]?.signal).toBe(controller.signal);
	});

	it("rejects malformed questions before calling Jev", async () => {
		const seen: JevEvaluationInput[] = [];
		const tool = createJevTool(recordingEvaluator(seen));
		await expect(
			tool.execute("call-4", { questions: { pick: { type: "choice", instructions: "Pick one" } } }),
		).rejects.toBeInstanceOf(JevInputError);
		await expect(
			tool.execute("call-5", {
				questions: { rate: { type: "score", instructions: "Rate it", criteria: ["Only"] } },
			}),
		).rejects.toBeInstanceOf(JevInputError);
		await expect(
			tool.execute("call-6", { questions: { ok: { type: "noul", instructions: "Fine?", true: "Yes" } } }),
		).rejects.toBeInstanceOf(JevInputError);
		await expect(tool.execute("call-7", { questions: {} })).rejects.toBeInstanceOf(JevInputError);
		expect(seen).toHaveLength(0);
	});
});
