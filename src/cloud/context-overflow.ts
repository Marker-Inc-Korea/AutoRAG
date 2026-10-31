/**
 * Context-overflow recognition for the hosted Dazzi Models API.
 *
 * Pi recovers from an overflowing request (compact, then retry once) only when
 * the provider error reads like an overflow. When the upstream provider behind
 * OpenRouter rejects an oversized request, the hosted server relays just
 * OpenRouter's generic wording and drops the reason:
 *
 *   dazzi API error (400): {"message":"Provider returned error","type":"invalid_request_error","code":"upstream_rejected_request"}
 *
 * Pi cannot classify that, so the session stayed stuck on the same 400. This
 * wrapper adds the missing diagnosis, but only when the request is large enough
 * for the rejection to be an overflow at all.
 */

import {
	type Api,
	type AssistantMessageEvent,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	isContextOverflow,
	type Model,
	type SimpleStreamOptions,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { getApiProvider } from "@earendil-works/pi-ai/compat";

/**
 * Most input tokens one character of request payload can cost. Plain text runs
 * near a quarter token per character, Korean text near one, and binary garbage
 * (U+FFFD, NUL, control characters) was measured at 0.72. Two leaves room above
 * all of them, so a request below `contextWindow / 2` characters cannot have
 * overflowed the window.
 */
const MAX_TOKENS_PER_CHAR = 2;

/** The relay's generic wording for an upstream rejection it knows no reason for. */
const REASONLESS_REJECTION = /^dazzi API error \(400\): (\{.*\})\s*$/su;

/** True for the hosted server's 400 `upstream_rejected_request` that carries only OpenRouter's generic message. */
export function isReasonlessUpstreamRejection(errorMessage: string): boolean {
	const body = REASONLESS_REJECTION.exec(errorMessage)?.[1];
	if (body === undefined) return false;
	try {
		const parsed: unknown = JSON.parse(body);
		if (typeof parsed !== "object" || parsed === null) return false;
		return (
			"code" in parsed &&
			"message" in parsed &&
			parsed.code === "upstream_rejected_request" &&
			parsed.message === "Provider returned error"
		);
	} catch {
		return false;
	}
}

/** Whether a request of `payloadChars` characters could have exceeded a `contextWindow`-token window. */
export function couldExceedContextWindow(payloadChars: number, contextWindow: number): boolean {
	return Number.isFinite(contextWindow) && contextWindow > 0 && payloadChars * MAX_TOKENS_PER_CHAR >= contextWindow;
}

/** The size of the request body Pi is about to send; computed only once a request has already failed. */
function payloadSize(payload: unknown): number {
	if (payload === undefined) return 0;
	try {
		return JSON.stringify(payload)?.length ?? 0;
	} catch {
		return 0;
	}
}

const lastEstimates = new Map<string, number>();
const ceilings = new Map<string, number>();
const modelKey = (model: Pick<Model<Api>, "provider" | "id">): string => `${model.provider}/${model.id}`;

/** The context guard reports the estimated size (its chars/4 unit, system prompt and tools included) of each request it releases. */
export function recordRequestEstimate(model: Pick<Model<Api>, "provider" | "id">, estimatedTokens: number): void {
	lastEstimates.set(modelKey(model), estimatedTokens);
}

/**
 * Remember that `model` just overflowed, so later requests are held to half the
 * estimated size of the one that failed. Pi compacts by summarising older turns
 * and keeping recent ones whole, which leaves an oversized recent tool result in
 * place; the context guard trims to this ceiling so the retry fits. The ceiling
 * only ever ratchets down, and is not set for a model whose requests the guard
 * never sized.
 */
export function noteContextOverflow(model: Pick<Model<Api>, "provider" | "id">): void {
	const key = modelKey(model);
	const failed = lastEstimates.get(key);
	if (failed === undefined) return;
	ceilings.set(key, Math.min(Math.floor(failed / 2), ceilings.get(key) ?? Number.POSITIVE_INFINITY));
}

/** The most estimated tokens a request to `model` may carry after an overflow, if it has overflowed. */
export function contextOverflowCeiling(model: Pick<Model<Api>, "provider" | "id">): number | undefined {
	return ceilings.get(modelKey(model));
}

/**
 * Pi's built-in `openai-responses` stream, with a reasonless upstream rejection
 * of a request that could not fit the model's window relabelled as a context
 * overflow so Pi compacts and retries it.
 */
export function streamDazziSimple(
	model: Model<Api>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const api = getApiProvider(model.api);
	if (api === undefined) throw new Error(`No API provider registered for api: ${model.api}`);
	let sent: unknown;
	const inner = api.streamSimple(model, context, {
		...options,
		onPayload: async (payload, target) => {
			const replacement = await options?.onPayload?.(payload, target);
			sent = replacement ?? payload;
			return replacement;
		},
	});
	const stream = createAssistantMessageEventStream();
	void (async () => {
		try {
			for await (const event of inner) {
				stream.push(event.type === "error" ? relabel(event, sent, model) : event);
			}
		} finally {
			stream.end();
		}
	})();
	return stream;
}

function relabel(
	event: Extract<AssistantMessageEvent, { type: "error" }>,
	payload: unknown,
	model: Model<Api>,
): AssistantMessageEvent {
	const { error } = event;
	const message = error.errorMessage;
	if (error.stopReason !== "error" || message === undefined) return event;
	const reasonless = isReasonlessUpstreamRejection(message);
	if (!reasonless && !isContextOverflow(error)) return event;
	const chars = payloadSize(payload);
	if (reasonless && !couldExceedContextWindow(chars, model.contextWindow)) return event;
	noteContextOverflow(model);
	if (!reasonless) return event;
	return {
		...event,
		error: {
			...error,
			errorMessage: `${message}\n[AutoRAG] No reason was given, and the request is ~${chars} characters against a ${model.contextWindow}-token window: treating this as context_length_exceeded.`,
		},
	};
}
