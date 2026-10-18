import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ModelsPublication, ModelsStoreEntry, RefreshModelsContext } from "@earendil-works/pi-ai";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildStoreEntry, mapServerModels } from "../../src/cloud/models.ts";
import { createDazziProviderConfig, refreshDazziModels, registerDazziProvider } from "../../src/cloud/provider.ts";
import { expectConnectionRefused, type FakeServer, startFakeServer } from "./helpers/fake-server.ts";

let server: FakeServer | undefined;
let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-cloud-provider-"));
});

afterEach(async () => {
	await server?.close();
	server = undefined;
	rmSync(root, { recursive: true, force: true });
});

interface FakeRefreshContext extends RefreshModelsContext {
	publications: ModelsPublication[];
}

function createRefreshContext(overrides: Partial<RefreshModelsContext> = {}): FakeRefreshContext {
	const publications: ModelsPublication[] = [];
	return {
		allowNetwork: true,
		signal: new AbortController().signal,
		publish: async (publication) => {
			publications.push(publication);
			return true;
		},
		publications,
		...overrides,
	};
}

const catalogEntry = {
	id: "vendor/model-alpha",
	name: "Model Alpha",
	context_window: 200_000,
	max_output_tokens: 8_192,
	input_modalities: ["text", "image"],
	reasoning: true,
	pricing: { input: 1, output: 2, cache_read: 0.1, cache_write: 0.2 },
};

function storedCatalog(): ModelsStoreEntry {
	const models = mapServerModels({ object: "list", data: [catalogEntry] });
	return buildStoreEntry(models, "dazzi", "https://api.dazziapp.com/v1");
}

function endpoints(baseUrl: string): { baseUrl: string; apiBaseUrl: string } {
	return { baseUrl, apiBaseUrl: `${baseUrl}/v1` };
}

describe("createDazziProviderConfig", () => {
	it("describes the provider without hardcoding any model", () => {
		const config = createDazziProviderConfig({});

		expect(config.name).toBe("Dazzi Models");
		expect(config.api).toBe("openai-responses");
		expect(config.baseUrl).toBe("https://api.dazziapp.com/v1");
		expect(config.apiKey).toBeUndefined();
		expect(config.models).toBeUndefined();
		expect(config.oauth?.name).toBe("Dazzi Models");
	});

	it("never declares an API-key reference, even when AUTORAG_API_KEY is set", () => {
		expect(createDazziProviderConfig({ AUTORAG_API_KEY: "dz_env" }).apiKey).toBeUndefined();
	});

	it("honours AUTORAG_BASE_URL and strips a trailing slash", () => {
		const config = createDazziProviderConfig({ AUTORAG_BASE_URL: "https://staging.example.com/" });
		expect(config.baseUrl).toBe("https://staging.example.com/v1");
	});

	it("exposes the access token as the API key and does not rotate it", async () => {
		const config = createDazziProviderConfig({});
		const credentials = { access: "dz_access", refresh: "", expires: 1 };

		expect(config.oauth?.getApiKey(credentials)).toBe("dz_access");
		await expect(config.oauth?.refreshToken(credentials, new AbortController().signal)).resolves.toBe(credentials);
	});

	it("runs the loopback login against AUTORAG_BASE_URL", async () => {
		server = await startFakeServer(() => ({ body: { access_token: "dz_env_login" } }));
		const config = createDazziProviderConfig({ AUTORAG_BASE_URL: server.url });

		let resolveAuthorized!: (url: string) => void;
		const authorized = new Promise<string>((resolve) => {
			resolveAuthorized = resolve;
		});
		const login = config.oauth!.login({
			onAuth: (info) => resolveAuthorized(info.url),
			onDeviceCode: () => {},
			onProgress: () => {},
			onPrompt: () => new Promise<string>(() => {}),
			onSelect: async () => undefined,
		});

		const authorize = new URL(await authorized);
		expect(authorize.origin).toBe(server.url);
		expect(authorize.searchParams.get("client")).toBe("agent");
		const redirectUri = authorize.searchParams.get("redirect_uri")!;
		const state = authorize.searchParams.get("state")!;

		const callback = await fetch(`${redirectUri}?code=env-code&state=${state}`);
		expect(await callback.text()).toContain("Signed in to Dazzi Models");

		const credentials = await login;
		expect(credentials.access).toBe("dz_env_login");
		await expectConnectionRefused(redirectUri);
	});
});

describe("refreshDazziModels", () => {
	it("restores the persisted catalog without network access", async () => {
		const context = createRefreshContext({ allowNetwork: false, stored: storedCatalog() });

		const models = await refreshDazziModels(context, endpoints("https://api.dazziapp.com"));

		expect(models.map((model) => model.id)).toEqual(["vendor/model-alpha"]);
		expect(context.publications).toHaveLength(0);
	});

	it("returns no models and makes no request when nothing is configured", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		const context = createRefreshContext();

		const models = await refreshDazziModels(context, endpoints(server.url));

		expect(models).toEqual([]);
		expect(server.requests).toHaveLength(0);
	});

	it("discovers models with the resolved API key and persists the catalog", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		const context = createRefreshContext({ credential: { type: "api_key", key: "dz_env_key" } });

		const models = await refreshDazziModels(context, endpoints(server.url));

		expect(models).toHaveLength(1);
		expect(models[0]).toMatchObject({ id: "vendor/model-alpha", reasoning: true, maxTokens: 8_192 });
		expect(server.requests[0]!.headers.authorization).toBe("Bearer dz_env_key");
		expect(server.requests[0]!.url).toBe("/v1/models");
		expect(context.publications).toHaveLength(1);
		expect(context.publications[0]!.persist).toMatchObject({
			models: [{ provider: "dazzi", api: "openai-responses", id: "vendor/model-alpha" }],
		});
	});

	it("uses the OAuth access token as the bearer token", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		const context = createRefreshContext({
			credential: { type: "oauth", access: "dz_oauth_token", refresh: "", expires: 1 },
		});

		await refreshDazziModels(context, endpoints(server.url));

		expect(server.requests[0]!.headers.authorization).toBe("Bearer dz_oauth_token");
	});

	it("keeps the stored catalog when the fetch fails", async () => {
		const context = createRefreshContext({
			credential: { type: "api_key", key: "dz_env_key" },
			stored: storedCatalog(),
		});

		const models = await refreshDazziModels(context, endpoints("http://127.0.0.1:1"));

		expect(models.map((model) => model.id)).toEqual(["vendor/model-alpha"]);
		expect(context.publications).toHaveLength(0);
	});

	it("keeps the stored catalog when the plan returns none", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [] } }));
		const context = createRefreshContext({
			credential: { type: "api_key", key: "dz_env_key" },
			stored: storedCatalog(),
		});

		const models = await refreshDazziModels(context, endpoints(server.url));

		expect(models.map((model) => model.id)).toEqual(["vendor/model-alpha"]);
		expect(context.publications).toHaveLength(0);
	});
});

describe("registerDazziProvider", () => {
	it("registers the provider on a ModelRuntime without a network call when signed out", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		const runtime = await ModelRuntime.create({
			authPath: join(root, "auth.json"),
			modelsPath: join(root, "models.json"),
			allowModelNetwork: false,
		});

		await registerDazziProvider(runtime, { AUTORAG_BASE_URL: server.url });

		const provider = runtime.getProvider("dazzi");
		expect(provider?.name).toBe("Dazzi Models");
		expect(provider?.baseUrl).toBe(`${server.url}/v1`);
		expect(provider?.auth.oauth?.name).toBe("Dazzi Models");
		expect(runtime.getModels("dazzi")).toHaveLength(0);
		expect(server.requests).toHaveLength(0);
	});

	it("refreshes a signed-out runtime without errors and keeps the persisted catalog", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		writeFileSync(join(root, "models-store.json"), JSON.stringify({ dazzi: storedCatalog() }));
		const runtime = await ModelRuntime.create({
			authPath: join(root, "auth.json"),
			modelsPath: join(root, "models.json"),
		});
		await registerDazziProvider(runtime, { AUTORAG_BASE_URL: server.url });

		const result = await runtime.refresh({ providers: ["dazzi"] });

		expect([...result.errors.keys()]).toEqual([]);
		expect(runtime.getModels("dazzi").map((model) => model.id)).toEqual(["vendor/model-alpha"]);
		expect(server.requests).toHaveLength(0);
	});

	it("refreshes with the stored login credential and persists the catalog", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		writeFileSync(
			join(root, "auth.json"),
			JSON.stringify({
				dazzi: { type: "oauth", access: "dz_login", refresh: "", expires: Number.MAX_SAFE_INTEGER },
			}),
		);
		const runtime = await ModelRuntime.create({
			authPath: join(root, "auth.json"),
			modelsPath: join(root, "models.json"),
		});
		await registerDazziProvider(runtime, { AUTORAG_BASE_URL: server.url });

		const result = await runtime.refresh({ providers: ["dazzi"] });

		expect([...result.errors.keys()]).toEqual([]);
		expect(server.requests[0]!.headers.authorization).toBe("Bearer dz_login");
		expect(runtime.getModels("dazzi").map((model) => model.id)).toEqual(["vendor/model-alpha"]);
	});

	it("ignores AUTORAG_API_KEY: it neither signs in nor makes a request", async () => {
		server = await startFakeServer(() => ({ body: { object: "list", data: [catalogEntry] } }));
		const previous = process.env.AUTORAG_API_KEY;
		process.env.AUTORAG_API_KEY = "dz_env_key";
		try {
			const runtime = await ModelRuntime.create({
				authPath: join(root, "auth.json"),
				modelsPath: join(root, "models.json"),
			});
			await registerDazziProvider(runtime, { AUTORAG_BASE_URL: server.url });

			const result = await runtime.refresh({ providers: ["dazzi"] });

			expect([...result.errors.keys()]).toEqual([]);
			expect(server.requests).toHaveLength(0);
			expect(runtime.getProviderAuthStatus("dazzi").configured).toBe(false);
		} finally {
			if (previous === undefined) delete process.env.AUTORAG_API_KEY;
			else process.env.AUTORAG_API_KEY = previous;
		}
	});

	it("loads the persisted catalog snapshot offline", async () => {
		writeFileSync(join(root, "models-store.json"), JSON.stringify({ dazzi: storedCatalog() }));
		const runtime = await ModelRuntime.create({
			authPath: join(root, "auth.json"),
			modelsPath: join(root, "models.json"),
			allowModelNetwork: false,
		});

		await registerDazziProvider(runtime);

		expect(runtime.getModel("dazzi", "vendor/model-alpha")).toMatchObject({
			provider: "dazzi",
			id: "vendor/model-alpha",
			api: "openai-responses",
		});
	});
});
