import { afterEach, describe, expect, it } from "vitest";
import { AutoragHttpError } from "../../src/cloud/errors.ts";
import {
	buildStoreEntry,
	DEFAULT_CONTEXT_WINDOW,
	DEFAULT_MAX_TOKENS,
	fetchModels,
	mapModelsError,
	mapServerModel,
	mapServerModels,
	modelsFromStore,
	resolveCredentialKey,
} from "../../src/cloud/models.ts";
import { rejectionOf } from "./helpers/assertions.ts";
import { type FakeServer, startFakeServer } from "./helpers/fake-server.ts";

let server: FakeServer | undefined;

afterEach(async () => {
	await server?.close();
	server = undefined;
});

const catalogEntry = {
	id: "vendor/model-alpha",
	object: "model",
	created: 1791397883,
	owned_by: "vendor",
	name: "Model Alpha",
	context_window: 1_000_000,
	max_output_tokens: 32_000,
	input_modalities: ["text", "image"],
	reasoning: true,
	pricing: { input: 0.1, output: 0.5, cache_read: 0.01, cache_write: 0.125 },
};

describe("mapServerModel", () => {
	it("maps a full catalog record to a Pi chat model", () => {
		expect(mapServerModel(catalogEntry)).toEqual({
			type: "chat",
			id: "vendor/model-alpha",
			name: "Model Alpha",
			input: ["text", "image"],
			cost: { input: 0.1, output: 0.5, cacheRead: 0.01, cacheWrite: 0.125 },
			reasoning: true,
			contextWindow: 1_000_000,
			maxTokens: 32_000,
		});
	});

	it("applies defaults and drops unknown modalities", () => {
		expect(mapServerModel({ id: "vendor/model-beta", input_modalities: ["audio", "text"] })).toEqual({
			type: "chat",
			id: "vendor/model-beta",
			name: "vendor/model-beta",
			input: ["text"],
			cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
			reasoning: false,
			contextWindow: DEFAULT_CONTEXT_WINDOW,
			maxTokens: DEFAULT_MAX_TOKENS,
		});
	});

	it("returns undefined without an id and skips invalid entries", () => {
		expect(mapServerModel({ name: "no id" })).toBeUndefined();
		expect(mapServerModels({ object: "list", data: [{ id: "ok" }, null, 5, {}] })).toHaveLength(1);
		expect(mapServerModels(null)).toEqual([]);
		expect(mapServerModels({ object: "list" })).toEqual([]);
	});
});

describe("fetchModels", () => {
	it("lists models with the bearer token and maps them", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));

		const models = await fetchModels({ baseUrl: server.url, apiKey: "dz_key" });

		expect(models).toHaveLength(1);
		expect(models[0]).toMatchObject({ id: "vendor/model-alpha" });
		const sent = server.requests[0]!;
		expect(sent.url).toBe("/v1/models");
		expect(sent.headers.authorization).toBe("Bearer dz_key");
	});

	it("maps a 401 to a re-login hint", async () => {
		server = await startFakeServer(() => ({
			status: 401,
			body: { error: { message: "Unauthorized", code: "unauthorized" } },
		}));
		const failure = await rejectionOf(fetchModels({ baseUrl: server.url, apiKey: "dz_stale" }));
		expect(failure).toBeInstanceOf(AutoragHttpError);
		expect(failure.message).toMatch(/HTTP 401/);
		expect((failure as AutoragHttpError).status).toBe(401);
	});

	it("wraps a connection failure", async () => {
		await expect(fetchModels({ baseUrl: "http://127.0.0.1:1", apiKey: "dz" })).rejects.toThrow(/could not reach/);
	});

	it("maps non-401 errors with the server message", () => {
		const error = mapModelsError(500, { error: { message: "boom" } });
		expect(error.message).toBe("AutoRAG model listing failed (HTTP 500): boom");
	});
});

describe("resolveCredentialKey", () => {
	it("reads the oauth access token or the api key", () => {
		expect(resolveCredentialKey({ type: "oauth", access: "dz_oauth", refresh: "", expires: 1 })).toBe("dz_oauth");
		expect(resolveCredentialKey({ type: "api_key", key: "dz_env" })).toBe("dz_env");
	});

	it("returns undefined when no usable secret is present", () => {
		expect(resolveCredentialKey(undefined)).toBeUndefined();
		expect(resolveCredentialKey({ type: "oauth", access: "", refresh: "", expires: 1 })).toBeUndefined();
	});
});

describe("catalog persistence round trip", () => {
	it("restores models from a stored snapshot", () => {
		const models = mapServerModels({ object: "list", data: [catalogEntry] });
		const entry = buildStoreEntry(models, "autorag", "https://api.dazziapp.com/v1");

		expect(entry.models).toHaveLength(1);
		expect(entry.models[0]).toMatchObject({
			type: "chat",
			provider: "autorag",
			api: "openai-responses",
			baseUrl: "https://api.dazziapp.com/v1",
			contextWindow: 1_000_000,
		});

		expect(modelsFromStore(entry)).toEqual(models);
	});

	it("returns nothing for a missing snapshot", () => {
		expect(modelsFromStore(undefined)).toEqual([]);
		expect(modelsFromStore({ models: [] })).toEqual([]);
	});
});
