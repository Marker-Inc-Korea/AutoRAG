import { join } from "node:path";
import type {
	ClassifierAnswer,
	ClassifierApi,
	ClassifierContext,
	ClassifierModel,
	ClassifierQuestion,
	ClassifierResult,
	JsonObject,
	ModelsClassifierOptions,
} from "@earendil-works/pi-ai";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { BackendError, type BackendRequest, type BackendResponse, type JevBackend, type RawAnswer } from "jev-use";

/**
 * Decision models come from pi's model catalog: pi-ai ships every provider's typed-question
 * ("classifier") models next to its chat models, with the transport for each wire dialect
 * (System One on TypeSafe, OpenRouter, Vercel AI Gateway, Cloudflare Workers AI, and OpenCode;
 * OpenAI's Decisions API for GPT-6 Luna), and pi's runtime adds the credentials in `auth.json` and
 * the provider `baseUrl` overrides in `models.json` (a proxy or gateway speaking the same dialect).
 * {@link PiDecisionBackend} puts one of those models behind jev-use's backend contract, so the
 * jev-use engine (screening, escalation) and every AutoRAG judgment stay unchanged.
 */

/**
 * The decision model asked on each provider when the config names none. OpenRouter stays on
 * `typesafe/jev-1.13`, the model AutoRAG's routing and permission thresholds were calibrated on.
 */
export const DECISION_PROVIDER_DEFAULT_MODELS: Readonly<Record<string, string>> = {
	openrouter: "typesafe/jev-1.13",
	openai: "gpt-6-luna",
	typesafe: "jev-latest",
	"vercel-ai-gateway": "typesafe-ai/jev-1.13.0",
	"cloudflare-workers-ai": "typesafe/jev",
	opencode: "jev-1.13",
};

/** The pi providers tried, in this order, when the caller names no decision backend. */
export const DECISION_PROVIDER_ORDER: readonly string[] = Object.keys(DECISION_PROVIDER_DEFAULT_MODELS);

/** What {@link PiDecisionBackend} reads from pi's model runtime. pi's `ModelRuntime` is one. */
export interface DecisionModelRuntime {
	getModelsOfType(type: "classifier", provider?: string): readonly ClassifierModel<ClassifierApi>[];
	classify(
		model: ClassifierModel<ClassifierApi>,
		context: ClassifierContext,
		options?: ModelsClassifierOptions,
	): Promise<ClassifierResult>;
}

export interface PiDecisionBackendOptions {
	/** pi provider id with decision models: `openrouter`, `openai`, `typesafe`, `vercel-ai-gateway`, `cloudflare-workers-ai`, or `opencode`. */
	readonly provider: string;
	/** Decision model id within the provider. Default: {@link DECISION_PROVIDER_DEFAULT_MODELS}, else the provider's first. */
	readonly model?: string;
	/** pi's runtime; built from `agentDir` on the first call when omitted. */
	readonly runtime?: DecisionModelRuntime;
	/** Pi agent home whose `auth.json` and `models.json` the runtime reads. Default: Pi's agent dir. */
	readonly agentDir?: string;
	/** Environment the provider credential is read from, ahead of `process.env`. */
	readonly env?: Record<string, string | undefined>;
}

/** One attempt's budget, matching jev-use's remote backends. */
const TIMEOUT_MS = 10_000;
const MAX_RETRIES = 2;

/** System One reads a question's `criteria` only when present; every other wire needs both meanings. */
const SYSTEM_ONE_APIS: Readonly<Record<string, true>> = {
	"typesafe-system-one": true,
	"cloudflare-workers-ai-system-one": true,
};

const runtimes = new Map<string, Promise<DecisionModelRuntime>>();

/** pi's runtime for one agent home, offline: the bundled catalog plus `models.json`, with `auth.json` credentials. */
function runtimeFor(agentDir: string): Promise<DecisionModelRuntime> {
	let runtime = runtimes.get(agentDir);
	if (runtime === undefined) {
		runtime = ModelRuntime.create({
			authPath: join(agentDir, "auth.json"),
			modelsPath: join(agentDir, "models.json"),
			allowModelNetwork: false,
		});
		runtimes.set(agentDir, runtime);
	}
	return runtime;
}

type BackendQuestion = BackendRequest["questions"][number];

function classifierQuestion(question: BackendQuestion, api: string): ClassifierQuestion {
	switch (question.type) {
		case "noul": {
			if (question.criteria !== undefined) {
				return { type: "bool", instructions: question.question, criteria: question.criteria };
			}
			if (SYSTEM_ONE_APIS[api] !== true) {
				return { type: "bool", instructions: question.question, criteria: { true: "", false: "" } };
			}
			// System One sends the question as given, and an empty criteria pair shifts Jev's calibration;
			// pi types `criteria` as required, so the question goes out without it.
			const bare = { type: "bool", instructions: question.question } as unknown as ClassifierQuestion;
			return bare;
		}
		case "choice": {
			const options = question.options ?? [];
			const criteria = Array.isArray(options)
				? Object.fromEntries(options.map((label) => [label, label]))
				: Object.fromEntries(Object.entries(options).map(([label, meaning]) => [label, meaning || label]));
			return { type: "choice", instructions: question.question, criteria };
		}
		case "score":
			return { type: "score", instructions: question.question, criteria: question.levels ?? [] };
	}
}

function rawAnswer(provider: string, question: BackendQuestion, answer: ClassifierAnswer | undefined): RawAnswer {
	if (answer === undefined) throw new BackendError(provider, `response missing answer for question "${question.id}"`);
	switch (answer.type) {
		case "bool":
			return { answer: answer.probability };
		case "choice":
			return { answer: answer.choice, distribution: answer.probabilities, confidence: answer.confidence };
		case "score":
			return { answer: answer.score, confidence: answer.confidence };
	}
}

/** HTTP status pi-ai put in its provider error text (`… error (401): …`). */
function statusOf(message: string): number | undefined {
	const status = /\((\d{3})\)/u.exec(message)?.[1];
	return status === undefined ? undefined : Number(status);
}

/**
 * One decision model from pi's catalog, as a jev-use backend. Missing credentials, an unknown model,
 * and provider errors surface as `BackendError`s with the provider's own message, which the jev-use
 * engine turns into `unreachable` escalations and the session check into a typed reason.
 */
export class PiDecisionBackend implements JevBackend {
	readonly name: string;
	private readonly modelId: string | undefined;
	private readonly runtime: DecisionModelRuntime | undefined;
	private readonly agentDir: string | undefined;
	private readonly env: Record<string, string> | undefined;

	constructor(options: PiDecisionBackendOptions) {
		this.name = options.provider;
		this.modelId = options.model;
		this.runtime = options.runtime;
		this.agentDir = options.agentDir;
		if (options.env !== undefined) {
			this.env = Object.fromEntries(
				Object.entries(options.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
			);
		}
	}

	async judge(request: BackendRequest): Promise<BackendResponse> {
		const runtime = this.runtime ?? (await runtimeFor(this.agentDir ?? getAgentDir()));
		const catalog = runtime.getModelsOfType("classifier", this.name);
		if (catalog.length === 0) {
			throw new BackendError(this.name, `HTTP 404: ${this.name} has no decision models in pi's model catalog`, 404);
		}
		const id = request.model ?? this.modelId ?? DECISION_PROVIDER_DEFAULT_MODELS[this.name] ?? catalog[0]?.id;
		const model = catalog.find((candidate) => candidate.id === id);
		if (model === undefined) {
			const known = catalog.map((candidate) => candidate.id).join(", ");
			throw new BackendError(
				this.name,
				`HTTP 404: ${this.name} has no decision model "${id}"; its decision models: ${known}`,
				404,
			);
		}
		// pi types the state as a JSON object; System One takes a string or an array verbatim, and OpenAI's
		// input is set to the raw text below, so jev-use's state is passed through as it is.
		const state = request.state as unknown as JsonObject;
		const text = typeof request.state === "string" ? request.state : undefined;
		const started = Date.now();
		const result = await runtime.classify(
			model,
			{
				state,
				questions: Object.fromEntries(
					request.questions.map((question) => [question.id, classifierQuestion(question, model.api)]),
				),
			},
			{
				timeoutMs: TIMEOUT_MS,
				maxRetries: MAX_RETRIES,
				...(this.env !== undefined ? { env: this.env } : {}),
				// pi-ai sends OpenAI's Decisions API the state as JSON text, which would quote a plain-text state and
				// escape its newlines; the text goes out as written instead.
				...(model.api === "openai-decisions" && text !== undefined
					? {
							onPayload: (payload: unknown) =>
								typeof payload === "object" && payload !== null ? { ...payload, input: text } : payload,
						}
					: {}),
			},
		);
		if (result.stopReason !== "stop") {
			const message = result.errorMessage ?? `classification ${result.stopReason}`;
			throw new BackendError(this.name, message, statusOf(message));
		}
		return {
			answers: request.questions.map((question) => rawAnswer(this.name, question, result.answers[question.id])),
			model: result.model,
			latencyMs: Date.now() - started,
			...(result.usage !== undefined
				? { usage: { inputTokens: result.usage.input, outputTokens: result.usage.output } }
				: {}),
		};
	}
}
