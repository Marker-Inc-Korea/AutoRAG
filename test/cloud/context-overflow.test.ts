import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type Api, type AssistantMessage, isContextOverflow, type Model } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
import { registerDazziProvider } from "../../src/cloud/provider.ts";
import { type FakeServer, startFakeServer } from "./helpers/fake-server.ts";
import { responsesTextStream, SSE_HEADERS } from "./helpers/responses-sse.ts";

/** What the hosted server returns when OpenRouter's upstream rejects a request without saying why. */
const GENERIC_REJECTION = {
	error: { message: "Provider returned error", type: "invalid_request_error", code: "upstream_rejected_request" },
};
const CONTEXT_WINDOW = 4_000;

let root: string;
let server: FakeServer | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-dazzi-overflow-"));
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

/** A signed-in runtime whose only model has a {@link CONTEXT_WINDOW}-token window and talks to the fake server. */
async function dazziRuntime(handler: Parameters<typeof startFakeServer>[0]): Promise<{
	runtime: ModelRuntime;
	model: Model<Api>;
}> {
	server = await startFakeServer(handler);
	writeFileSync(
		join(root, "auth.json"),
		JSON.stringify({ dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER } }),
	);
	const models = mapServerModels({
		object: "list",
		data: [
			{
				id: "vendor/model-alpha",
				name: "Model Alpha",
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
	const model = runtime.getModel("dazzi", "vendor/model-alpha");
	if (model === undefined) throw new Error("the stored Dazzi catalog did not load");
	return { runtime, model };
}

async function ask(runtime: ModelRuntime, model: Model<Api>, text: string): Promise<AssistantMessage> {
	return runtime.completeSimple(model, { messages: [{ role: "user", content: text, timestamp: Date.now() }] });
}

describe("Dazzi context overflow classification", () => {
	it("tells pi a reasonless 400 on a request too large for the window is a context overflow", async () => {
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: GENERIC_REJECTION }));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(reply.stopReason).toBe("error");
		expect(isContextOverflow(reply)).toBe(true);
		// The server's own words stay in front of the added diagnosis.
		expect(reply.errorMessage).toContain("dazzi API error (400)");
		expect(reply.errorMessage).toContain("Provider returned error");
	});

	it("leaves a reasonless 400 on a request that fits the window as an ordinary error", async () => {
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: GENERIC_REJECTION }));

		const reply = await ask(runtime, model, "안녕하세요");

		expect(reply.stopReason).toBe("error");
		expect(isContextOverflow(reply)).toBe(false);
		expect(reply.errorMessage).not.toMatch(/context_length_exceeded/);
	});

	it("leaves a 400 that states its reason untouched, however large the request", async () => {
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
		expect(reply.errorMessage).not.toMatch(/context_length_exceeded/);
	});

	it("still recognises a server that reports context_length_exceeded itself", async () => {
		const exceeded = {
			error: { message: "Provider returned error", type: "invalid_request_error", code: "context_length_exceeded" },
		};
		const { runtime, model } = await dazziRuntime(() => ({ status: 400, body: exceeded }));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(isContextOverflow(reply)).toBe(true);
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
	});

	it("streams a successful reply through unchanged", async () => {
		const { runtime, model } = await dazziRuntime(() => ({
			headers: SSE_HEADERS,
			rawBody: responsesTextStream("안녕하세요"),
		}));

		const reply = await ask(runtime, model, "x".repeat(20_000));

		expect(reply.stopReason).toBe("stop");
		expect(reply.content).toEqual([expect.objectContaining({ type: "text", text: "안녕하세요" })]);
	});
});
