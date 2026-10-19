/**
 * Jev backend for the hosted Dazzi Models plan.
 *
 * The server answers Jev's native dialect at `POST <root>/v1/jev` behind the
 * Dazzi login token (stored by `/login`), returning the same `answers` map the
 * TypeSafe and OpenRouter backends do. `jev-use`'s package index does not
 * export its native-dialect helpers, so the small mapping and the retrying JSON
 * transport are reimplemented here against that documented dialect (covered by
 * tests against a local HTTP server).
 *
 * Failures surface as `BackendError(name, "HTTP <status>: <message>", status)`
 * so the engine degrades to an escalation with a readable hint instead of
 * blocking the loop.
 */

import {
	BackendError,
	type BackendRequest,
	type BackendResponse,
	type JevBackend,
	type Question,
	type RawAnswer,
	type State,
} from "jev-use";
import { API_PATH, resolveBaseUrl } from "./config.ts";
import { DAZZI_LOGIN_REQUIRED_MESSAGE, readDazziLoginToken } from "./credential.ts";

/** API path appended to the normalized server root. */
export const JEV_PATH = `${API_PATH}/jev`;

/** Live Jev model id on the hosted Dazzi Models plan. */
export const DAZZI_DEFAULT_JEV_MODEL = "jev-1.13";

/** Abort one attempt after this long, matching `jev-use`'s remote backends. */
const DEFAULT_TIMEOUT_MS = 10_000;
/** Retries on 408/429/5xx and network failures, matching `jev-use`. */
const DEFAULT_MAX_RETRIES = 2;
const RETRY_BASE_MS = 500;
const RETRY_MAX_MS = 5_000;

/** Options for {@link DazziJevBackend}. */
export interface DazziJevBackendOptions {
	/** Environment holding `AUTORAG_BASE_URL`. Default `process.env`. */
	readonly env?: Record<string, string | undefined>;
	/** Pi agent home whose `auth.json` holds the `/login` credential. Default: Pi's agent dir. */
	readonly agentDir?: string;
	/** Explicit bearer token (test seam); when omitted, the stored login token is used. */
	readonly apiKey?: string;
	/** Override the server root (test seam); defaults to `resolveBaseUrl(env)`. */
	readonly baseUrl?: string;
	/** Model used when a call names none. @default DAZZI_DEFAULT_JEV_MODEL */
	readonly defaultModel?: string;
	/** Abort one attempt after this long. @default 10_000 */
	readonly timeoutMs?: number;
	/** Retries on 408/429/5xx and network failures. @default 2 */
	readonly maxRetries?: number;
	/** Injection seam for tests. */
	readonly fetchImpl?: typeof fetch;
}

interface NativeAnswer {
	type: string;
	noul?: number;
	choice?: string;
	score?: number;
	confidence?: number;
	probabilities?: Record<string, number>;
	legend?: Record<string, string>;
}

interface NativeResponse {
	model?: string;
	answers?: unknown;
	usage?: { input_tokens?: number; output_tokens?: number };
}

type NativeWireQuestion =
	| { type: "noul"; instructions: string; criteria?: { true: string; false: string } }
	| { type: "choice"; instructions: string; criteria: Record<string, string> }
	| { type: "score"; instructions: string; criteria: string[] };

interface NativeBody {
	model: string;
	state: string;
	questions: Record<string, NativeWireQuestion>;
}

/**
 * Jev backend served by the hosted Dazzi Models plan.
 *
 * The constructor throws a named error when no account key is configured — a
 * missing credential must be loud, never a silent fake judgment — and the retry
 * policy mirrors `jev-use`'s own remote backends (408/429/5xx, `Retry-After`).
 */
export class DazziJevBackend implements JevBackend {
	readonly name = "dazzi";
	private readonly apiKey: string;
	private readonly baseUrl: string;
	private readonly defaultModel: string;
	private readonly timeoutMs: number;
	private readonly maxRetries: number;
	private readonly fetchImpl: typeof fetch | undefined;

	constructor(options: DazziJevBackendOptions = {}) {
		const env = options.env ?? process.env;
		const apiKey = options.apiKey ?? readDazziLoginToken(options.agentDir);
		if (apiKey === undefined || apiKey.trim() === "") {
			throw new Error(`jev.backend "dazzi" needs a sign-in. ${DAZZI_LOGIN_REQUIRED_MESSAGE}`);
		}
		this.apiKey = apiKey;
		this.baseUrl = options.baseUrl ?? resolveBaseUrl(env);
		this.defaultModel = options.defaultModel ?? DAZZI_DEFAULT_JEV_MODEL;
		this.timeoutMs = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		this.maxRetries = options.maxRetries ?? DEFAULT_MAX_RETRIES;
		this.fetchImpl = options.fetchImpl;
	}

	async judge(request: BackendRequest): Promise<BackendResponse> {
		const model = request.model ?? this.defaultModel;
		const body = toNativeBody(request.state, request.questions, model);
		const started = Date.now();
		const response = (await this.postJson(`${this.baseUrl}${JEV_PATH}`, body)) as NativeResponse;
		return {
			answers: parseNativeAnswers(this.name, response.answers, request.questions),
			model: response.model ?? model,
			latencyMs: Date.now() - started,
			usage: {
				inputTokens: response.usage?.input_tokens,
				outputTokens: response.usage?.output_tokens,
			},
		};
	}

	/** POST JSON with a per-attempt timeout and jev-use's retry policy. */
	private async postJson(url: string, body: unknown): Promise<unknown> {
		const doFetch = this.fetchImpl ?? fetch;
		let lastError: BackendError | undefined;
		for (let attempt = 0; attempt <= this.maxRetries; attempt += 1) {
			if (attempt > 0) await sleep(retryDelay(attempt, lastError));
			const controller = new AbortController();
			const timer = setTimeout(() => controller.abort(), this.timeoutMs);
			try {
				const response = await doFetch(url, {
					method: "POST",
					headers: {
						"Content-Type": "application/json",
						Accept: "application/json",
						Authorization: `Bearer ${this.apiKey}`,
					},
					body: JSON.stringify(body),
					signal: controller.signal,
				});
				if (response.ok) return (await response.json()) as unknown;
				const message = await extractErrorMessage(response);
				lastError = new BackendError(this.name, `HTTP ${response.status}: ${message}`, response.status);
				lastError.retryAfterMs = parseRetryAfter(response);
				if (!isRetryable(response.status) || attempt === this.maxRetries) throw lastError;
			} catch (error) {
				if (error instanceof BackendError) {
					if (attempt === this.maxRetries || !isRetryable(error.status)) throw error;
					continue;
				}
				const aborted = (error as Error)?.name === "AbortError";
				lastError = new BackendError(
					this.name,
					aborted ? `request timed out after ${this.timeoutMs}ms` : `network error: ${String(error)}`,
					undefined,
					error,
				);
				if (attempt === this.maxRetries) throw lastError;
			} finally {
				clearTimeout(timer);
			}
		}
		throw lastError ?? new BackendError(this.name, "unreachable");
	}
}

/** The state as the server requires it: a string, JSON-encoded when not already one. */
function serializeState(state: State): string {
	return typeof state === "string" ? state : JSON.stringify(state);
}

/** Translate one question into the server's native dialect. */
function toNativeQuestion(question: Question): NativeWireQuestion {
	switch (question.type) {
		case "noul":
			return {
				type: "noul",
				instructions: question.question,
				...(question.criteria ? { criteria: question.criteria } : {}),
			};
		case "choice":
			return {
				type: "choice",
				instructions: question.question,
				criteria: Object.fromEntries(
					optionEntries(question.options ?? []).map(([label, meaning]) => [label, meaning || label]),
				),
			};
		case "score":
			return {
				type: "score",
				instructions: question.question,
				criteria: question.levels ?? [],
			};
	}
}

/** Build the whole request body: model, a string state, and the questions by id. */
export function toNativeBody(state: State, questions: (Question & { id: string })[], model: string): NativeBody {
	return {
		model,
		state: serializeState(state),
		questions: Object.fromEntries(questions.map((question) => [question.id, toNativeQuestion(question)])),
	};
}

/**
 * Read the provider's answer map back into raw answers, in question order.
 * Anything missing or mistyped is a `BackendError` — a silently dropped answer
 * would read as a low-confidence judgment.
 */
export function parseNativeAnswers(
	backend: string,
	answers: unknown,
	questions: (Question & { id: string })[],
): RawAnswer[] {
	if (!answers || typeof answers !== "object") {
		throw new BackendError(backend, "response has no answers object");
	}
	const byId = answers as Record<string, NativeAnswer>;
	return questions.map((question) => parseNativeAnswer(backend, question, byId[question.id]));
}

/** Shape one provider answer into a raw answer for its question's primitive. */
function parseNativeAnswer(
	backend: string,
	question: Question & { id: string },
	answer: NativeAnswer | undefined,
): RawAnswer {
	if (!answer) {
		throw new BackendError(backend, `response missing answer for question "${question.id}"`);
	}
	switch (question.type) {
		case "noul":
			return {
				answer: numberOrThrow(backend, question.id, answer.noul),
				confidence: answer.confidence,
			};
		case "choice":
			return {
				answer: stringOrThrow(backend, question.id, answer.choice),
				confidence: answer.confidence,
				distribution: answer.probabilities,
			};
		case "score":
			return {
				answer: numberOrThrow(backend, question.id, answer.score),
				confidence: answer.confidence,
				distribution: answer.probabilities,
				legend: answer.legend,
			};
	}
}

/** Normalize choice options to label → meaning (labels-only become their own meaning). */
function optionEntries(options: string[] | Record<string, string>): [string, string][] {
	return Array.isArray(options) ? options.map((label) => [label, ""] as [string, string]) : Object.entries(options);
}

function numberOrThrow(backend: string, id: string, value: unknown): number {
	if (typeof value !== "number" || !Number.isFinite(value)) {
		throw new BackendError(backend, `answer "${id}" has no numeric value`);
	}
	return value;
}

function stringOrThrow(backend: string, id: string, value: unknown): string {
	if (typeof value !== "string" || !value) {
		throw new BackendError(backend, `answer "${id}" has no choice value`);
	}
	return value;
}

function isRetryable(status?: number): boolean {
	if (status === undefined) return true; // network / timeout
	return status === 408 || status === 429 || status >= 500;
}

function retryDelay(attempt: number, lastError?: BackendError): number {
	if (lastError?.retryAfterMs !== undefined) return Math.min(lastError.retryAfterMs, 60_000);
	const base = Math.min(RETRY_BASE_MS * 2 ** (attempt - 1), RETRY_MAX_MS);
	return base * (1 + Math.random() * 0.25);
}

function parseRetryAfter(response: Response): number | undefined {
	const milliseconds = response.headers.get("retry-after-ms");
	if (milliseconds && Number.isFinite(Number(milliseconds))) return Number(milliseconds);
	const seconds = response.headers.get("retry-after");
	if (seconds && Number.isFinite(Number(seconds))) return Number(seconds) * 1000;
	return undefined;
}

/** Dig the provider's own message out of an OpenAI-shaped error body. */
async function extractErrorMessage(response: Response): Promise<string> {
	try {
		const data = (await response.json()) as Record<string, unknown>;
		for (const key of ["error", "message", "detail"]) {
			const value = data[key];
			if (typeof value === "string") return value;
			if (value && typeof value === "object") {
				const message = (value as Record<string, unknown>).message;
				if (typeof message === "string") return message;
			}
			if (Array.isArray(value) && value.length > 0) {
				const first = value[0] as Record<string, unknown>;
				if (typeof first?.msg === "string") return first.msg;
			}
		}
		return JSON.stringify(data).slice(0, 300);
	} catch {
		return response.statusText || "request failed";
	}
}

function sleep(ms: number): Promise<void> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise((resolve) => setTimeout(resolve, ms));
}
