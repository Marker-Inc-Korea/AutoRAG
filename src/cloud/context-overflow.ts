/**
 * Context-overflow recovery for the hosted Dazzi Models API.
 *
 * The hosted server reports a request that exceeds the model's window as
 * `400 context_length_exceeded` (or HTTP 413 for an oversized body), which Pi
 * already recognises: it compacts, then retries once.
 *
 * That recovery is not enough on its own. Pi compacts by summarising older
 * turns and keeping the latest ones whole, so an oversized tool result that is
 * the newest turn survives compaction and the retry is rejected again. This
 * module remembers which model just overflowed and how big the rejected request
 * was, so the context guard can hold the retry to a fraction of that size.
 */

import {
	type Api,
	type AssistantMessageEventStream,
	createAssistantMessageEventStream,
	isContextOverflow,
	type Model,
	type SimpleStreamOptions,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { getApiProvider } from "@earendil-works/pi-ai/compat";

const lastEstimates = new Map<string, number>();
const ceilings = new Map<string, number>();
const modelKey = (model: Pick<Model<Api>, "provider" | "id">): string => `${model.provider}/${model.id}`;

/** The context guard reports the estimated size (its chars/4 unit, system prompt and tools included) of each request it releases. */
export function recordRequestEstimate(model: Pick<Model<Api>, "provider" | "id">, estimatedTokens: number): void {
	lastEstimates.set(modelKey(model), estimatedTokens);
}

/**
 * Remember that `model` just overflowed, so later requests are held to half the
 * estimated size of the one that failed. The ceiling only ever ratchets down,
 * and is not set for a model whose requests the guard never sized.
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
 * Pi's built-in `openai-responses` stream, unchanged except that a context
 * overflow it reports is noted for {@link contextOverflowCeiling}.
 */
export function streamDazziSimple(
	model: Model<Api>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
): AssistantMessageEventStream {
	const api = getApiProvider(model.api);
	if (api === undefined) throw new Error(`No API provider registered for api: ${model.api}`);
	const inner = api.streamSimple(model, context, options);
	const stream = createAssistantMessageEventStream();
	void (async () => {
		try {
			for await (const event of inner) {
				if (event.type === "error" && event.error.stopReason === "error" && isContextOverflow(event.error)) {
					noteContextOverflow(model);
				}
				stream.push(event);
			}
		} finally {
			stream.end();
		}
	})();
	return stream;
}
