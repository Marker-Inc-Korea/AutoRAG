import { join } from "node:path";
import {
	type ClassifierAnswer,
	type ClassifierApi,
	type ClassifierContext,
	type ClassifierModel,
	type ClassifierQuestion,
	type ClassifierResult,
	createProvider,
	envApiKeyAuth,
	InMemoryCredentialStore,
	type JsonObject,
	type ModelsClassifierOptions,
	type Provider,
} from "@earendil-works/pi-ai";
import { typesafeSystemOneApi } from "@earendil-works/pi-ai/api/typesafe-system-one.lazy";
import { getAgentDir, ModelRuntime } from "@earendil-works/pi-coding-agent";
import { BackendError, type BackendRequest, type BackendResponse, type JevBackend, type RawAnswer } from "jev-use";
import { wireQuestionKeys } from "../cloud/jev-backend.ts";

/**
 * Decision models come from pi's model catalog plus AutoRAG-registered decision providers (such as
 * Articul8 Bivia): pi-ai ships every built-in provider's typed-question ("classifier") models next
 * to its chat models, with the transport for each wire dialect (System One on TypeSafe, OpenRouter,
 * Vercel AI Gateway, Cloudflare Workers AI, OpenCode, and Bivia; OpenAI's Decisions API for GPT-6
 * Luna), and pi's runtime adds the credentials in `auth.json` and the provider `baseUrl` overrides
 * in `models.json` (a proxy or gateway speaking the same dialect).
 * {@link PiDecisionBackend} puts one of those models behind jev-use's backend contract, so the
 * jev-use engine (screening, escalation) and every AutoRAG judgment stay unchanged.
 */

/** Articul8 Bivia System One provider id, default base URL, model id, and API key env vars. */
export const BIVIA_PROVIDER_ID = "bivia";
export const BIVIA_BASE_URL = "https://api.articul8.ai/bivia/v1";
export const BIVIA_DEFAULT_MODEL = "dec-1";
export const BIVIA_API_KEY_ENV_VARS: readonly string[] = ["BIVIA_API_KEY", "A8_API_KEY"];

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
	[BIVIA_PROVIDER_ID]: BIVIA_DEFAULT_MODEL,
};

/** The pi providers tried, in this order, when the caller names no decision backend. */
export const DECISION_PROVIDER_ORDER: readonly string[] = Object.keys(DECISION_PROVIDER_DEFAULT_MODELS);

/** Build the Articul8 Bivia native pi provider (System One dialect at `POST /v1/systemone`). */
export function biviaProvider(): Provider {
	const models: ClassifierModel<"typesafe-system-one">[] = [
		{
			type: "classifier",
			id: BIVIA_DEFAULT_MODEL,
			name: "Bivia (dec-1)",
			api: "typesafe-system-one",
			provider: BIVIA_PROVIDER_ID,
			baseUrl: BIVIA_BASE_URL,
			input: ["text", "image"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			contextWindow: 64000,
			compat: { imageInput: "assets", sendImageOptInHeader: false },
		},
	];
	return createProvider({
		id: BIVIA_PROVIDER_ID,
		name: "Bivia",
		baseUrl: BIVIA_BASE_URL,
		auth: {
			apiKey: envApiKeyAuth("Bivia API key", BIVIA_API_KEY_ENV_VARS),
		},
		models,
		classifiers: {
			"typesafe-system-one": typesafeSystemOneApi(),
		},
	});
}

/**
 * Register AutoRAG's extra decision-model providers on a pi `ModelRuntime`.
 * Skips any provider already shipped by pi-ai so future upstream additions win cleanly.
 */
export function registerDecisionProviders(runtime: ModelRuntime): void {
	if (runtime.getModelsOfType("classifier", BIVIA_PROVIDER_ID).length === 0) {
		runtime.registerNativeProvider(biviaProvider());
	}
}

/** First non-empty env value for a Bivia API key (`BIVIA_API_KEY`, then `A8_API_KEY`). */
export function getBiviaEnvApiKey(env: Record<string, string | undefined> = process.env): string | undefined {
	for (const key of BIVIA_API_KEY_ENV_VARS) {
		const value = env[key];
		if (value !== undefined && value.length > 0) return value;
	}
	return undefined;
}

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
	/** pi provider id with decision models: `openrouter`, `openai`, `typesafe`, `vercel-ai-gateway`, `cloudflare-workers-ai`, `opencode`, or `bivia`. */
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
	const environmentOnly = process.env.AUTORAG_AUTH_MODE === "environment";
	const cacheKey = `${environmentOnly ? "environment" : "stored"}:${agentDir}`;
	let runtime = runtimes.get(cacheKey);
	if (runtime === undefined) {
		runtime = ModelRuntime.create({
			...(environmentOnly
				? { credentials: new InMemoryCredentialStore(), modelsPath: null, refreshOnCreate: false }
				: { authPath: join(agentDir, "auth.json"), modelsPath: join(agentDir, "models.json") }),
			allowModelNetwork: false,
		}).then((created) => {
			registerDecisionProviders(created);
			return created;
		});
		runtimes.set(cacheKey, runtime);
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
		const template = catalog[0];
		if (template === undefined) {
			throw new BackendError(this.name, `HTTP 404: ${this.name} has no decision models in pi's model catalog`, 404);
		}
		const id = request.model ?? this.modelId ?? DECISION_PROVIDER_DEFAULT_MODELS[this.name] ?? template.id;
		// A model released after this pi-ai build is not in its catalog yet; it is asked on the provider's own
		// wire anyway, and a provider that does not serve it answers with its own "model not found".
		const model = catalog.find((candidate) => candidate.id === id) ?? {
			...template,
			id,
			name: id,
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		};
		// pi types the state as a JSON object; System One takes a string or an array verbatim, and OpenAI's
		// input is set to the raw text below, so jev-use's state is passed through as it is.
		const state = request.state as unknown as JsonObject;
		const text = typeof request.state === "string" ? request.state : undefined;
		const started = Date.now();
		const keys = wireQuestionKeys(request.questions.map((question) => question.id));
		const result = await runtime.classify(
			model,
			{
				state,
				questions: Object.fromEntries(
					request.questions.map((question, index) => [keys[index], classifierQuestion(question, model.api)]),
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
			answers: request.questions.map((question, index) =>
				rawAnswer(this.name, question, result.answers[keys[index] ?? question.id]),
			),
			model: result.model,
			latencyMs: Date.now() - started,
			...(result.usage !== undefined
				? { usage: { inputTokens: result.usage.input, outputTokens: result.usage.output } }
				: {}),
		};
	}
}
