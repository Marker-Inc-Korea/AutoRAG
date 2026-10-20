import { describe, expect, it } from "vitest";
import { remoteEmbedderFromMinSync } from "../../src/embedding/settings.ts";

describe("remoteEmbedderFromMinSync", () => {
	it("treats no embedder, the native default, and an empty block as local", () => {
		expect(remoteEmbedderFromMinSync(undefined)).toBeUndefined();
		expect(remoteEmbedderFromMinSync({})).toBeUndefined();
		expect(remoteEmbedderFromMinSync({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 })).toBeUndefined();
	});

	it("reads a TEI server from the id and base URL", () => {
		expect(
			remoteEmbedderFromMinSync({ id: "tei:BAAI/bge-m3", baseUrl: "http://gpu-box:8080", dimension: 1024 }),
		).toMatchObject({ protocol: "tei", model: "BAAI/bge-m3", baseUrl: "http://gpu-box:8080", dimension: 1024 });
	});

	it("sends an openai: id to api.openai.com unless a gateway URL is configured", () => {
		expect(
			remoteEmbedderFromMinSync({ id: "openai:text-embedding-3-small", apiKeyEnv: "OPENAI_API_KEY" }),
		).toMatchObject({
			protocol: "openai",
			model: "text-embedding-3-small",
			baseUrl: "https://api.openai.com",
			apiKeyEnv: "OPENAI_API_KEY",
		});
		expect(remoteEmbedderFromMinSync({ id: "openai:Qwen3", baseUrl: "https://router.example/api" })?.baseUrl).toBe(
			"https://router.example/api",
		);
	});

	it("treats an id with no known kind but a base URL as an OpenAI-compatible gateway", () => {
		expect(
			remoteEmbedderFromMinSync({ id: "Qwen/Qwen3-Embedding-0.6B", baseUrl: "http://127.0.0.1:8089/v1" }),
		).toMatchObject({
			protocol: "openai",
			model: "Qwen/Qwen3-Embedding-0.6B",
		});
	});

	it("cannot guess an endpoint for a bare id without a URL", () => {
		expect(remoteEmbedderFromMinSync({ id: "Qwen3-Embedding-0.6B-Q8_0.gguf" })).toBeUndefined();
	});
});
