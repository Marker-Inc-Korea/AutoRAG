import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Api, type AssistantMessage, isContextOverflow, type Model } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { contextOverflowCeiling, recordRequestEstimate } from "../../src/cloud/context-overflow.ts";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
import { registerDazziProvider } from "../../src/cloud/provider.ts";
import { type FakeServer, startFakeServer } from "./helpers/fake-server.ts";
import { responsesTextStream, SSE_HEADERS } from "./helpers/responses-sse.ts";

/** What the hosted server answers when the upstream provider says the input is longer than the model's window. */
const CONTEXT_EXCEEDED = {
	error: {
		message:
			"This model's maximum context length is 1048576 tokens, but the request resolved to 1244118 input tokens.",
		type: "invalid_request_error",
		code: "context_length_exceeded",
	},
};
/** A rejection with no reason at all: nothing says the request was too big, so none is assumed. */
const GENERIC_REJECTION = {
	error: { message: "Provider returned error", type: "invalid_request_error", code: "upstream_rejected_request" },
};
const CONTEXT_WINDOW = 4_000;
/** The size the context guard reported for the request that fails in these tests. */
const REQUEST_ESTIMATE = 3_000;

let root: string;
let server: FakeServer | undefined;
let modelCounter = 0;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-dazzi-overflow-"));
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

/**
 * A signed-in runtime whose only model has a {@link CONTEXT_WINDOW}-token window and talks to the fake server.
 * Every call gets its own model id: overflow ceilings are remembered per model for the life of the process.
 */
async function dazziRuntime(handler: Parameters<typeof startFakeServer>[0]): Promise<{
	runtime: ModelRuntime;
	model: Model<Api>;
}> {
	server = await startFakeServer(handler);
	const modelId = `vendor/model-${++modelCounter}`;
	writeFileSync(
		join(root, "auth.json"),
		JSON.stringify({ dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER } }),
	);
	const models = mapServerModels({
		object: "list",
		data: [
			{
				id: modelId,
				name: "Model",
				context_window: CONTEXT_WINDOW,
				max_output_tokens: 512,
				input_modalities: ["text"],
				reasoning: false,
				pricing: { input: 1, output: 2, cache_read: 0, cache_write: 0 },
			},
		],
	});
	writeFileSync(
		join(root, "models-store.json"),
		JSON.stringify({ dazzi: buildStoreEntry(models, "dazzi", `${server.url}/v1`) }),
	);
	const runtime = await ModelRuntime.create({
		authPath: join(root, "auth.json"),
		modelsPath: join(root, "models.json"),
		allowModelNetwork: false,
	});
	await registerDazziProvider(runtime, { AUTORAG_BASE_URL: server.url });
	const model = runtime.getModel("dazzi", modelId);
	if (model === undefined) throw new Error("the stored Dazzi catalog did not load");
	return { runtime, model };
}

/** Sends one request the way the session does: the guard sizes it, then the stream carries it. */
async function ask(runtime: ModelRuntime, model: Model<Api>, text: string): Promise<AssistantMessage> {
	recordRequestEstimate(model, REQUEST_ESTIMATE);
	return runtime.completeSimple(model, { messages: [{ role: "user", content: text, timestamp: Date.now() }] });
}

describe("Dazzi context overflow handling", () => {
	it("recognises the server's context_length_exceeded and holds later requests to half the failed size", async () => {
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: CONTEXT_EXCEEDED }));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(reply.stopReason).toBe("error");
		expect(isContextOverflow(reply)).toBe(true);
		expect(contextOverflowCeiling(model)).toBe(REQUEST_ESTIMATE / 2);
	});

	it("holds later requests down further when the retry overflows again", async () => {
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: CONTEXT_EXCEEDED }));

		await ask(runtime, model, "x".repeat(20_000));
		recordRequestEstimate(model, REQUEST_ESTIMATE / 2);
		await runtime.completeSimple(model, { messages: [{ role: "user", content: "again", timestamp: Date.now() }] });

		expect(contextOverflowCeiling(model)).toBe(REQUEST_ESTIMATE / 4);
	});

	it("treats an HTTP 413 as a context overflow", async () => {
		const { runtime, model } = await dazziRuntime(() => ({
			status: 413,
			body: {
				error: { message: "Request Entity Too Large", type: "invalid_request_error", code: "request_too_large" },
			},
		}));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(isContextOverflow(reply)).toBe(true);
		expect(contextOverflowCeiling(model)).toBe(REQUEST_ESTIMATE / 2);
	});

	it("does not guess that a reasonless 400 was an overflow, however large the request", async () => {
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: GENERIC_REJECTION }));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(reply.stopReason).toBe("error");
		expect(isContextOverflow(reply)).toBe(false);
		expect(reply.errorMessage).toContain("Provider returned error");
		expect(contextOverflowCeiling(model)).toBeUndefined();
	});

	it("leaves a 400 that states another reason alone", async () => {
		const reason = {
			error: {
				message: "Reasoning is mandatory for this endpoint",
				type: "invalid_request_error",
				code: "upstream_rejected_request",
			},
		};
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: reason }));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(isContextOverflow(reply)).toBe(false);
		expect(reply.errorMessage).toContain("Reasoning is mandatory");
		expect(contextOverflowCeiling(model)).toBeUndefined();
	});

	it("streams a successful reply through unchanged and sets no ceiling", async () => {
		const { runtime, model } = await dazziRuntime(() => ({
			headers: SSE_HEADERS,
			rawBody: responsesTextStream("안녕하세요"),
		}));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(reply.stopReason).toBe("stop");
		expect(reply.content).toEqual([expect.objectContaining({ type: "text", text: "안녕하세요" })]);
		expect(contextOverflowCeiling(model)).toBeUndefined();
	});
});
