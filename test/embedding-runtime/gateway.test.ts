import { afterEach, describe, expect, it } from "vitest";
import { type EmbeddingGateway, GatewayError, startEmbeddingGateway } from "../../src/embedding-runtime/gateway.ts";
import { resolveProfile } from "../../src/embedding-runtime/manifest.ts";

const profile = resolveProfile("embeddinggemma-300m");
const gateways: EmbeddingGateway[] = [];

afterEach(async () => {
	await Promise.all(gateways.splice(0).map((gateway) => gateway.close()));
});

async function gateway(
	fetchImpl: typeof fetch = async () =>
		new Response(JSON.stringify({ data: [] }), { headers: { "content-type": "application/json" } }),
) {
	const server = await startEmbeddingGateway({
		profile,
		upstreamUrl: "http://127.0.0.1:9999",
		port: 0,
		fetch: fetchImpl,
	});
	gateways.push(server);
	return server;
}

function upstream(rows: readonly number[][], indices?: readonly number[]): typeof fetch {
	return async () => {
		return new Response(
			JSON.stringify({ data: rows.map((embedding, i) => ({ index: indices?.[i] ?? i, embedding })) }),
			{
				status: 200,
				headers: { "content-type": "application/json" },
			},
		);
	};
}

async function post(server: EmbeddingGateway, path: string, body: unknown): Promise<Response> {
	return fetch(`${server.url}${path}`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify(body),
	});
}

const row = (n: number) => Array.from({ length: profile.dimension }, (_, i) => (i === 0 ? n : 0));

describe("embedding gateway", () => {
	it("rejects non-loopback listen host before binding", async () => {
		await expect(
			startEmbeddingGateway({ profile, host: "0.0.0.0", upstreamUrl: "http://127.0.0.1:1" }),
		).rejects.toMatchObject({ code: "non-loopback-host" });
	});

	it("rejects non-loopback upstream URL", async () => {
		await expect(startEmbeddingGateway({ profile, upstreamUrl: "https://example.com" })).rejects.toBeInstanceOf(
			GatewayError,
		);
	});

	it("serves healthz profile payload", async () => {
		const server = await gateway();
		expect(await (await fetch(`${server.url}/healthz`)).json()).toEqual({
			status: "ok",
			backend: profile.backend,
			model: profile.model,
			dimension: profile.dimension,
			runtimeBuild: profile.runtimeBuild,
			profileId: profile.profileId,
		});
	});

	it("returns ordered embeddings with exact dimensions", async () => {
		const server = await gateway(upstream([row(1), row(2)]));
		const response = await post(server, "/embed", { inputs: ["one", "two"] });
		expect(response.status).toBe(200);
		expect(await response.json()).toEqual([row(1), row(2)]);
	});

	it("orders upstream rows by index", async () => {
		const server = await gateway(upstream([row(2), row(1)], [1, 0]));
		expect(await (await post(server, "/embed", { inputs: ["one", "two"] })).json()).toEqual([row(1), row(2)]);
	});

	it.each([[row(1).map(() => Number.NaN)], [row(1).slice(0, 2)]])(
		"returns 502 for invalid upstream rows",
		async (bad) => {
			const server = await gateway(upstream([bad]));
			expect((await post(server, "/embed", { inputs: ["one"] })).status).toBe(502);
		},
	);

	it("rejects empty inputs and batch limits", async () => {
		const server = await gateway();
		expect((await post(server, "/embed", { inputs: [] })).status).toBe(400);
		const limited = await startEmbeddingGateway({
			profile,
			upstreamUrl: "http://127.0.0.1:9999",
			port: 0,
			maxInputs: 1,
			fetch: upstream([row(1)]),
		});
		gateways.push(limited);
		expect((await post(limited, "/embed", { inputs: ["a", "b"] })).status).toBe(413);
	});

	it("returns 504 for an upstream timeout", async () => {
		const server = await startEmbeddingGateway({
			profile,
			upstreamUrl: "http://127.0.0.1:9999",
			port: 0,
			timeoutMs: 10,
			fetch: (_input, init) =>
				new Promise((_resolve, reject) =>
					init?.signal?.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError"))),
				),
		});
		gateways.push(server);
		expect((await post(server, "/embed", { inputs: ["one"] })).status).toBe(504);
	});

	it("supports OpenAI and Ollama route shapes", async () => {
		const server = await gateway(upstream([row(3)]));
		const openai = (await (await post(server, "/v1/embeddings", { model: "ignored", input: ["one"] })).json()) as {
			data: Array<{ embedding: number[] }>;
		};
		expect(openai.data[0].embedding).toEqual(row(3));
		expect(await (await post(server, "/api/embeddings", { model: "ignored", prompt: "one" })).json()).toEqual({
			embedding: row(3),
		});
	});
});
