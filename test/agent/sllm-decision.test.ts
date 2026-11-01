import { randomUUID } from "node:crypto";
import {
	type AssistantMessage,
	type Context,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import { checkDecisionModel, DecisionModelUnavailableError } from "../../src/agent/decision-model-check.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { SLLM_DECISION_TOOL_NAME, SllmDecisionBackend } from "../../src/agent/sllm-decision.ts";

const registrations: FauxProviderRegistration[] = [];

afterEach(() => {
	for (const registration of registrations.splice(0)) registration.unregister();
});

function sllm(...responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-sllm-${randomUUID()}`, models: [{ id: "small-1" }] });
	registration.setResponses(responses);
	registrations.push(registration);
	return registration.getModel();
}

/** Records the context each request carried, then answers with `step`. */
function recording(step: AssistantMessage, seen: Context[]): FauxResponseStep {
	return (context) => {
		seen.push(context);
		return step;
	};
}

const QUESTIONS = [
	{ id: "urgent", type: "noul" as const, question: "Is this urgent?" },
	{
		id: "team",
		type: "choice" as const,
		question: "Which team?",
		options: { billing: "Payments", technical: "Bugs" },
	},
	{ id: "anger", type: "score" as const, question: "How angry?", levels: ["calm", "upset", "furious"] },
];

describe("SllmDecisionBackend", () => {
	it("asks the small LLM through its decision tool and turns its probabilities into Jev verdicts", async () => {
		const seen: Context[] = [];
		const model = sllm(
			recording(
				fauxAssistantMessage(
					fauxToolCall(SLLM_DECISION_TOOL_NAME, {
						answers: [
							{ id: "urgent", probabilities: [0.9], confidence: 0.9 },
							{ id: "team", probabilities: [0.25, 0.75], confidence: 0.8 },
							{ id: "anger", probabilities: [0.1, 0.3, 0.6], confidence: 0.7 },
						],
					}),
					{ stopReason: "toolUse" },
				),
				seen,
			),
		);
		const judge = createJevJudge({ backend: new SllmDecisionBackend({ model, apiKey: "k" }) });

		const judgment = await judge("Payouts have failed for three days.", QUESTIONS);

		expect(judgment.backend).toBe("sllm");
		expect(judgment.model).toBe(`${model.provider}/small-1`);
		expect(judgment.answers.urgent?.answer).toBe(0.9);
		expect(judgment.answers.team?.answer).toBe("technical");
		expect(judgment.answers.team?.distribution).toEqual({ billing: 0.25, technical: 0.75 });
		expect(judgment.answers.anger?.answer).toBeCloseTo(1.5, 5);
		// The state and every question, with its options or levels in order, reach the model.
		const prompt = JSON.stringify(seen[0]?.messages);
		expect(prompt).toContain("Payouts have failed for three days.");
		expect(prompt).toContain("Which team?");
		expect(prompt).toContain("furious");
		expect(JSON.stringify(seen[0])).toContain(SLLM_DECISION_TOOL_NAME);
	});

	it("reads answers the model wrote as a JSON message instead of a tool call", async () => {
		const model = sllm(
			fauxAssistantMessage('```json\n{"answers":[{"id":"urgent","probabilities":[0.2],"confidence":0.8}]}\n```', {
				stopReason: "stop",
			}),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[0]]);
		expect(judgment.answers.urgent?.answer).toBe(0.2);
	});

	it("normalizes choice probabilities that do not sum to one", async () => {
		const model = sllm(
			fauxAssistantMessage(
				fauxToolCall(SLLM_DECISION_TOOL_NAME, {
					answers: [{ id: "team", probabilities: [2, 6], confidence: 0.6 }],
				}),
				{ stopReason: "toolUse" },
			),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[1]]);
		expect(judgment.answers.team?.distribution).toEqual({ billing: 0.25, technical: 0.75 });
	});

	it("is unavailable, with the reason, when the small LLM's request fails", async () => {
		const model = sllm(fauxAssistantMessage("", { stopReason: "error", errorMessage: "401 Unauthorized: bad key" }));
		const error = await checkDecisionModel(createJevJudge({ backend: new SllmDecisionBackend({ model }) })).then(
			() => undefined,
			(caught: unknown) => caught,
		);
		if (!(error instanceof DecisionModelUnavailableError))
			throw new Error(`expected a decision-model error, got ${error}`);
		expect(error.reason).toBe("unauthorized");
		expect(error.backend).toBe("sllm");
	});

	it("is unavailable as invalid-response when the reply holds no usable answers", async () => {
		const model = sllm(fauxAssistantMessage("I think it is probably fine.", { stopReason: "stop" }));
		const error = await checkDecisionModel(createJevJudge({ backend: new SllmDecisionBackend({ model }) })).then(
			() => undefined,
			(caught: unknown) => caught,
		);
		if (!(error instanceof DecisionModelUnavailableError))
			throw new Error(`expected a decision-model error, got ${error}`);
		expect(error.reason).toBe("invalid-response");
	});
});
