import { describe, expect, it, vi } from "vitest";
import {
	createEmbeddingRuntime,
	type EmbeddingRuntimeCache,
	type EmbeddingRuntimeSupervisor,
} from "../../src/embedding-runtime/index.ts";
import type { SupervisorStatus } from "../../src/embedding-runtime/supervisor.ts";

function deps() {
	const ready: SupervisorStatus = { state: "ready", port: 43123, backend: "auto", model: "model.gguf" };
	const supervisor: EmbeddingRuntimeSupervisor = {
		ensureRunning: vi.fn(async () => ready),
		shutdown: vi.fn(async () => {}),
		status: vi.fn(() => ready),
	};
	const cache: EmbeddingRuntimeCache = {
		downloadAsset: vi.fn(async (asset) => `/cache/${asset.filename}`),
		importAsset: vi.fn(async (_path, asset) => `/cache/${asset.filename}`),
		verifyCacheEntry: vi.fn(async (path) => path),
	};
	return { supervisor, cache };
}

describe("embedding runtime public API", () => {
	it("resolves the default profile and ensures verified assets before starting", async () => {
		const { supervisor, cache } = deps();
		const runtime = createEmbeddingRuntime({
			supervisor,
			cache,
			platform: "darwin-arm64-metal",
			gatewayFactory: async ({ upstreamUrl }) => ({
				url: upstreamUrl.replace(/:\d+$/, ":43123"),
				close: async () => {},
			}),
		});
		const result = await runtime.ensureRuntime();
		expect(result.profile.profileId).toBe("qwen3-embedding-0.6b");
		expect(result.baseUrl).toBe("http://127.0.0.1:43123");
		expect(cache.downloadAsset).toHaveBeenCalledTimes(2);
		expect(supervisor.ensureRunning).toHaveBeenCalledTimes(1);
	});

	it("prefetches/imports/verifies the selected model without starting the runtime", async () => {
		const { supervisor, cache } = deps();
		const runtime = createEmbeddingRuntime({ supervisor, cache, platform: "darwin-arm64-metal" });
		await runtime.prefetchModel();
		await runtime.importModel("embeddinggemma-300m", "/tmp/model.gguf");
		await runtime.verifyModel();
		expect(cache.importAsset).toHaveBeenCalledTimes(1);
		expect(cache.verifyCacheEntry).toHaveBeenCalledTimes(1);
		expect(supervisor.ensureRunning).not.toHaveBeenCalled();
	});

	it("stops and reports sanitized gateway state", async () => {
		const { supervisor, cache } = deps();
		const fetch = vi.fn(async () => new Response(JSON.stringify({ status: "ok" }), { status: 200 }));
		const runtime = createEmbeddingRuntime({ supervisor, cache, fetch, platform: "darwin-arm64-metal" });
		const status = await runtime.runtimeStatus();
		expect(status.state).toBe("ready");
		await runtime.stopRuntime();
		expect(supervisor.shutdown).toHaveBeenCalledTimes(1);
	});

	it("recognizes a live gateway /healthz payload as healthy", async () => {
		const { supervisor, cache } = deps();
		const fetch = vi.fn(
			async () =>
				new Response(
					JSON.stringify({
						status: "ok",
						backend: "auto",
						model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
						dimension: 1024,
						runtimeBuild: "b10951",
						profileId: "qwen3-embedding-0.6b",
					}),
					{ status: 200 },
				),
		);
		const runtime = createEmbeddingRuntime({
			supervisor,
			cache,
			fetch,
			platform: "darwin-arm64-metal",
			gatewayFactory: async ({ upstreamUrl }) => ({
				url: upstreamUrl.replace(/:\d+$/, ":43123"),
				close: async () => {},
			}),
		});
		await runtime.ensureRuntime();
		const status = await runtime.runtimeStatus();
		expect(status.health).toEqual({
			ok: true,
			profileId: "qwen3-embedding-0.6b",
			dimension: 1024,
			runtimeBuild: "b10951",
		});
	});

	it("reports a non-ok gateway health payload as a failure", async () => {
		const { supervisor, cache } = deps();
		const fetch = vi.fn(async () => new Response(JSON.stringify({ status: "starting" }), { status: 200 }));
		const runtime = createEmbeddingRuntime({
			supervisor,
			cache,
			fetch,
			platform: "darwin-arm64-metal",
			gatewayFactory: async ({ upstreamUrl }) => ({
				url: upstreamUrl.replace(/:\d+$/, ":43123"),
				close: async () => {},
			}),
		});
		await runtime.ensureRuntime();
		const status = await runtime.runtimeStatus();
		expect(status.health.ok).toBe(false);
		expect(status.health).toMatchObject({ code: "incompatible" });
	});
});
