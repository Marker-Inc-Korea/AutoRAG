import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ClassifierApi, ClassifierModel } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	DECISION_PROVIDER_DEFAULT_MODELS,
	type DecisionModelRuntime,
	PiDecisionBackend,
} from "../../src/agent/decision-backends.ts";
import { checkDecisionModel, DecisionModelUnavailableError } from "../../src/agent/decision-model-check.ts";
import { createJevJudge } from "../../src/agent/jev-extension.ts";
import { type FakeServer, startFakeServer } from "../cloud/helpers/fake-server.ts";

let agentDir: string;
let runtime: ModelRuntime;
let server: FakeServer | undefined;

beforeEach(async () => {
	agentDir = mkdtempSync(join(tmpdir(), "autorag-decision-backends-"));
	runtime = await ModelRuntime.create({
		authPath: join(agentDir, "auth.json"),
		modelsPath: join(agentDir, "models.json"),
		allowModelNetwork: false,
	});
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(agentDir, { recursive: true, force: true });
});

/** pi's real runtime and catalog, with every decision model's endpoint moved to the local fake server. */
function pointedAt(url: string): DecisionModelRuntime {
	return {
		getModelsOfType: (type, provider) =>
			runtime.getModelsOfType(type, provider).map((model) => ({ ...model, baseUrl: url })),
		classify: (model, context, options) => runtime.classify(model, context, options),
	};
}

async function unavailable(promise: Promise<unknown>): Promise<DecisionModelUnavailableError> {
	const error = await promise.then(
		() => undefined,
		(caught: unknown) => caught,
	);
	if (!(error instanceof DecisionModelUnavailableError))
		throw new Error(`expected a decision-model error, got ${error}`);
	return error;
}

describe("pi's decision-model catalog", () => {
	it("has a live decision model for every provider AutoRAG defaults, the GPT-6 Luna family included", () => {
		const ids = (provider: string) => runtime.getModelsOfType("classifier", provider).map((model) => model.id);
		for (const [provider, model] of Object.entries(DECISION_PROVIDER_DEFAULT_MODELS)) {
			expect(ids(provider), provider).toContain(model);
		}
		expect(ids("openai")).toContain("gpt-6-luna");
		expect(ids("openrouter")).toContain("openai/gpt-6-luna-decisions");
		expect(ids("vercel-ai-gateway")).toContain("openai/gpt-6-luna-decisions");
	});
});

describe("PiDecisionBackend over System One (OpenRouter)", () => {
	it("sends Jev's native questions to the provider's systemone endpoint and reads the answers back", async () => {
		server = await startFakeServer(() => ({
			body: {
				model: "typesafe/jev-1.13-20260917",
				answers: {
					urgent: { type: "noul", noul: 0.91 },
					team: {
						type: "choice",
						choice: "billing",
						probabilities: { billing: 0.8, technical: 0.2 },
						confidence: 0.8,
					},
					anger: { type: "score", score: 1.4, probabilities: { 0: 0.1, 1: 0.4, 2: 0.5 }, confidence: 0.7 },
				},
			},
		}));
		const judge = createJevJudge({
			backend: new PiDecisionBackend({
				provider: "openrouter",
				runtime: pointedAt(server.url),
				env: { OPENROUTER_API_KEY: "or-test-key" },
			}),
		});

		const judgment = await judge("Payouts have failed for three days.", [
			{ id: "urgent", type: "noul", question: "Is this urgent?" },
			{ id: "team", type: "choice", question: "Which team?", options: ["billing", "technical"] },
			{ id: "anger", type: "score", question: "How angry?", levels: ["calm", "upset", "furious"] },
		]);

		const request = server.requests[0];
		expect(request?.url).toBe("/systemone");
		expect(request?.headers.authorization).toBe("Bearer or-test-key");
		expect(request?.json).toEqual({
			model: DECISION_PROVIDER_DEFAULT_MODELS.openrouter,
			// The state goes out verbatim, and a noul without criteria carries none (an empty pair shifts Jev).
			state: "Payouts have failed for three days.",
			questions: {
				urgent: { type: "noul", instructions: "Is this urgent?" },
				team: {
					type: "choice",
					instructions: "Which team?",
					criteria: { billing: "billing", technical: "technical" },
				},
				anger: { type: "score", instructions: "How angry?", criteria: ["calm", "upset", "furious"] },
			},
		});
		expect(judgment.backend).toBe("openrouter");
		// pi-ai reports the catalog model it asked, not the provider's dated snapshot.
		expect(judgment.model).toBe(DECISION_PROVIDER_DEFAULT_MODELS.openrouter);
		expect(judgment.answers.urgent?.answer).toBe(0.91);
		expect(judgment.answers.team?.answer).toBe("billing");
		expect(judgment.answers.team?.distribution).toEqual({ billing: 0.8, technical: 0.2 });
		expect(judgment.answers.anger?.answer).toBe(1.4);
	});

	it("asks the configured decision model, e.g. GPT-6 Luna Decisions through OpenRouter", async () => {
		server = await startFakeServer(() => ({ body: { answers: { probe: { type: "noul", noul: 0.5 } } } }));
		const backend = new PiDecisionBackend({
			provider: "openrouter",
			model: "openai/gpt-6-luna-decisions",
			runtime: pointedAt(server.url),
			env: { OPENROUTER_API_KEY: "or-test-key" },
		});
		await createJevJudge({ backend })("x", [{ id: "probe", type: "noul", question: "ok?" }]);
		expect(server.requests[0]?.json).toMatchObject({ model: "openai/gpt-6-luna-decisions" });
	});

	it("fails as unauthorized when the provider rejects the key", async () => {
		server = await startFakeServer(() => ({ status: 401, body: { error: { message: "User not found" } } }));
		const backend = new PiDecisionBackend({
			provider: "openrouter",
			runtime: pointedAt(server.url),
			env: { OPENROUTER_API_KEY: "or-bad-key" },
		});
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend })));
		expect(error.reason).toBe("unauthorized");
		expect(error.backend).toBe("openrouter");
	});
});

describe("PiDecisionBackend over OpenAI's Decisions API", () => {
	it("sends the state as plain text input and maps a predicate back to a probability", async () => {
		server = await startFakeServer(() => ({
			body: {
				model: "gpt-6-luna",
				answers: [
					{ type: "predicate", name: "lease", probability: 0.97 },
					{
						type: "choice",
						name: "kind",
						choice: "contract",
						probabilities: [
							{ value: "contract", probability: 0.9 },
							{ value: "invoice", probability: 0.1 },
						],
						confidence: 0.9,
					},
				],
				usage: { input_tokens: 12, output_tokens: 0 },
			},
		}));
		const judge = createJevJudge({
			backend: new PiDecisionBackend({
				provider: "openai",
				runtime: pointedAt(server.url),
				env: { OPENAI_API_KEY: "sk-test" },
			}),
		});

		const judgment = await judge("The Seoul office lease\nrenews on 2027-03-31.", [
			{ id: "lease", type: "noul", question: "Does the state give the lease renewal date?" },
			{
				id: "kind",
				type: "choice",
				question: "What document is this?",
				options: { contract: "a contract", invoice: "a bill" },
			},
		]);

		const request = server.requests[0];
		expect(request?.url).toBe("/decisions");
		expect(request?.json).toEqual({
			model: DECISION_PROVIDER_DEFAULT_MODELS.openai,
			// Raw text, newlines intact: not a JSON-quoted string.
			input: "The Seoul office lease\nrenews on 2027-03-31.",
			questions: [
				{ type: "predicate", name: "lease", instructions: "Does the state give the lease renewal date?" },
				{
					type: "choice",
					name: "kind",
					instructions: "What document is this?",
					choices: [
						{ value: "contract", description: "a contract" },
						{ value: "invoice", description: "a bill" },
					],
				},
			],
		});
		expect(judgment.backend).toBe("openai");
		expect(judgment.answers.lease?.answer).toBe(0.97);
		expect(judgment.answers.kind?.answer).toBe("contract");
	});
});

describe("PiDecisionBackend failures", () => {
	it("fails as model-not-found for a model the provider's catalog lacks, listing what it has", async () => {
		const backend = new PiDecisionBackend({
			provider: "openrouter",
			model: "typesafe/jev-0.0-nope",
			runtime: pointedAt("http://127.0.0.1:9"),
			env: { OPENROUTER_API_KEY: "or-test-key" },
		});
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend })));
		expect(error.reason).toBe("model-not-found");
		expect(error.detail).toContain("typesafe/jev-0.0-nope");
		expect(error.detail).toContain("openai/gpt-6-luna-decisions");
	});

	it("fails as model-not-found for a provider with no decision models", async () => {
		const backend = new PiDecisionBackend({ provider: "anthropic", runtime: pointedAt("http://127.0.0.1:9") });
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend })));
		expect(error.reason).toBe("model-not-found");
		expect(error.detail).toMatch(/anthropic has no decision models/u);
	});

	it("fails as missing-credential when pi has no credential for the provider", async () => {
		const missing: DecisionModelRuntime = {
			getModelsOfType: (type, provider) => runtime.getModelsOfType(type, provider),
			classify: async (model: ClassifierModel<ClassifierApi>) => ({
				api: model.api,
				provider: model.provider,
				model: model.id,
				answers: {},
				stopReason: "error",
				errorMessage: `Provider is not configured: ${model.provider}`,
				timestamp: Date.now(),
			}),
		};
		const backend = new PiDecisionBackend({ provider: "cloudflare-workers-ai", runtime: missing });
		const error = await unavailable(checkDecisionModel(createJevJudge({ backend })));
		expect(error.reason).toBe("missing-credential");
		expect(error.backend).toBe("cloudflare-workers-ai");
	});
});
