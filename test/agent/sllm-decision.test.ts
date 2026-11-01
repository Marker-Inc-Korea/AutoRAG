import { randomUUID } from "node:crypto";
import {
	type AssistantMessage,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	type SimpleStreamOptions,
	type Tool,
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

/** One request as the small LLM received it. */
interface Seen {
	readonly prompt: string;
	readonly tool: Tool | undefined;
	readonly options: SimpleStreamOptions | undefined;
}

/** Records each request, then answers with `step`. */
function recording(step: AssistantMessage, seen: Seen[]): FauxResponseStep {
	return (context, options) => {
		const tools = context.messages.flatMap((message) =>
			message.role === "system" ? (message.toolsAdded ?? []) : [],
		);
		seen.push({
			prompt: JSON.stringify(context.messages),
			tool: tools.find((tool) => tool.name === SLLM_DECISION_TOOL_NAME),
			options,
		});
		return step;
	};
}

function submit(answers: Parameters<typeof fauxToolCall>[1]["answers"]): AssistantMessage {
	return fauxAssistantMessage(fauxToolCall(SLLM_DECISION_TOOL_NAME, { answers }), { stopReason: "toolUse" });
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
	it("asks for exactly this request's answers: one required key per question, options and levels as named fields", async () => {
		const seen: Seen[] = [];
		const model = sllm(
			recording(
				submit({
					urgent: { yes: 0.9, confidence: 0.9 },
					team: { probabilities: { billing: 0.25, technical: 0.75 }, confidence: 0.8 },
					anger: { probabilities: { "0": 0.1, "1": 0.3, "2": 0.6 }, confidence: 0.7 },
				}),
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

		const schema = seen[0]?.tool?.parameters as {
			properties: {
				answers: {
					required: string[];
					properties: Record<string, { required: string[]; properties: Record<string, { required?: string[] }> }>;
				};
			};
		};
		const answers = schema.properties.answers;
		expect(answers.required).toEqual(["urgent", "team", "anger"]);
		expect(answers.properties.urgent?.required).toEqual(["yes", "confidence"]);
		expect(answers.properties.team?.properties.probabilities?.required).toEqual(["billing", "technical"]);
		expect(answers.properties.anger?.properties.probabilities?.required).toEqual(["0", "1", "2"]);
		expect(seen[0]?.tool?.constrainedSampling).toEqual({ type: "json_schema", strict: "prefer" });
		// The state, every question with its options or levels, and an example of the exact reply reach the model.
		expect(seen[0]?.prompt).toContain("Payouts have failed for three days.");
		expect(seen[0]?.prompt).toContain("Which team?");
		expect(seen[0]?.prompt).toContain("furious");
		expect(seen[0]?.prompt).toContain('\\"technical\\"');
		expect(seen[0]?.prompt).toMatch(/Example reply/u);
	});

	it("asks at temperature 0 with reasoning off and room for the whole reply", async () => {
		const seen: Seen[] = [];
		const model = sllm(recording(submit({ urgent: { yes: 0.2, confidence: 0.9 } }), seen));
		await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[0]]);
		expect(seen[0]?.options?.temperature).toBe(0);
		expect(seen[0]?.options?.reasoning).toBeUndefined();
		expect(seen[0]?.options?.maxTokens).toBeGreaterThanOrEqual(1024);
	});

	it("sends AutoRAG's colon ids under wire-safe keys and reads each answer back for its own question", async () => {
		const seen: Seen[] = [];
		const model = sllm(
			recording(
				submit({ datasource_web: { yes: 0.8, confidence: 0.9 }, citation_0: { yes: 0.1, confidence: 0.9 } }),
				seen,
			),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [
			{ id: "datasource:web", type: "noul", question: "Search the web?" },
			{ id: "citation:0", type: "noul", question: "Relevant?" },
		]);
		expect(judgment.answers["datasource:web"]?.answer).toBe(0.8);
		expect(judgment.answers["citation:0"]?.answer).toBe(0.1);
	});

	it.each([
		["a fenced JSON message", '```json\n{"answers":{"urgent":{"yes":0.2,"confidence":0.8}}}\n```'],
		[
			"JSON after prose, with a trailing comma",
			'Sure. {"answers": {"urgent": {"yes": 0.2, "confidence": 0.8,},}} Done.',
		],
		["the answers object without its wrapper", '{"urgent": {"yes": 0.2, "confidence": 0.8}}'],
		["a probability written as a percentage string", '{"answers":{"urgent":{"yes":"20%","confidence":"0.8"}}}'],
	])("reads %s instead of a tool call", async (_label, text) => {
		const model = sllm(fauxAssistantMessage(text, { stopReason: "stop" }));
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[0]]);
		expect(judgment.answers.urgent?.answer).toBeCloseTo(0.2, 5);
	});

	it("normalizes choice probabilities that do not sum to one", async () => {
		const model = sllm(submit({ team: { probabilities: { billing: 2, technical: 6 }, confidence: 0.6 } }));
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[1]]);
		expect(judgment.answers.team?.distribution).toEqual({ billing: 0.25, technical: 0.75 });
	});

	it("asks again, once, for only the answers a reply left out or got wrong, and keeps the ones it got right", async () => {
		const seen: Seen[] = [];
		const model = sllm(
			recording(
				submit({ urgent: { yes: 0.9, confidence: 0.9 }, team: { probabilities: { billing: 1 }, confidence: 0.5 } }),
				seen,
			),
			recording(
				submit({
					team: { probabilities: { billing: 0.4, technical: 0.6 }, confidence: 0.7 },
					anger: { probabilities: { "0": 0, "1": 1, "2": 0 }, confidence: 0.7 },
				}),
				seen,
			),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", QUESTIONS);
		expect(seen).toHaveLength(2);
		const retried = seen[1]?.tool?.parameters as { properties: { answers: { required: string[] } } };
		expect(retried.properties.answers.required).toEqual(["team", "anger"]);
		expect(judgment.answers.urgent?.answer).toBe(0.9);
		expect(judgment.answers.team?.answer).toBe("technical");
		expect(judgment.answers.anger?.answer).toBe(1);
	});

	it("asks again when the reply was cut off before any answer", async () => {
		const model = sllm(
			fauxAssistantMessage('{"answers":{"urg', { stopReason: "length" }),
			submit({ urgent: { yes: 0.3, confidence: 0.8 } }),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })("x", [QUESTIONS[0]]);
		expect(judgment.answers.urgent?.answer).toBe(0.3);
	});

	it("hands back only the questions still unanswered after the retry, as unsure, and keeps the rest", async () => {
		const model = sllm(
			submit({ urgent: { yes: 0.9, confidence: 0.9 } }),
			submit({ team: { probabilities: { nobody: 1 }, confidence: 0.9 } }),
		);
		const judgment = await createJevJudge({ backend: new SllmDecisionBackend({ model }) })(
			"x",
			QUESTIONS.slice(0, 2),
		);
		expect(judgment.answers.urgent?.answer).toBe(0.9);
		expect(judgment.answers.team?.answer).toBeNull();
		expect(judgment.answers.team?.escalate).toBe(true);
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

	it("is unavailable as invalid-response when neither attempt holds a usable answer", async () => {
		const model = sllm(
			fauxAssistantMessage("I think it is probably fine.", { stopReason: "stop" }),
			fauxAssistantMessage("Still fine.", { stopReason: "stop" }),
		);
		const error = await checkDecisionModel(createJevJudge({ backend: new SllmDecisionBackend({ model }) })).then(
			() => undefined,
			(caught: unknown) => caught,
		);
		if (!(error instanceof DecisionModelUnavailableError))
			throw new Error(`expected a decision-model error, got ${error}`);
		expect(error.reason).toBe("invalid-response");
	});
});
