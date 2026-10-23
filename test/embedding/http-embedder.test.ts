import { createServer, type IncomingMessage, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { createHttpEmbedder } from "../../src/embedding/http-embedder.ts";

interface Captured {
	readonly url: string;
	readonly authorization: string | undefined;
	readonly body: Record<string, unknown>;
}

let servers: Server[] = [];
afterEach(() => {
	for (const server of servers) server.close();
	servers = [];
});

async function readBody(request: IncomingMessage): Promise<string> {
	let text = "";
	for await (const chunk of request) text += chunk;
	return text;
}

/** Real loopback server so the wire format, not a fetch stub, is what the test pins. */
async function serve(
	handler: (captured: Captured) => { status?: number; body: unknown },
): Promise<{ readonly baseUrl: string; readonly calls: Captured[] }> {
	const calls: Captured[] = [];
	const server = createServer(async (request, response) => {
		const captured: Captured = {
			url: request.url ?? "",
			authorization: request.headers.authorization,
			body: JSON.parse(await readBody(request)) as Record<string, unknown>,
		};
		calls.push(captured);
		const reply = handler(captured);
		response.statusCode = reply.status ?? 200;
		response.setHeader("content-type", "application/json");
		response.end(typeof reply.body === "string" ? reply.body : JSON.stringify(reply.body));
	});
	servers.push(server);
	await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
	return { baseUrl: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, calls };
}

const vectorFor = (text: string): number[] => [text.length, 1, 0];

describe("createHttpEmbedder openai protocol", () => {
	it("posts to /v1/embeddings once per batch, keeps input order, and reorders by response index", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({
			body: {
				data: (body.input as string[]).map((text, index) => ({ index, embedding: vectorFor(text) })).reverse(),
			},
		}));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			batchSize: 2,
			provider: "openai",
		});
		const vectors = await embedder.embed(["a", "bb", "ccc"]);
		expect(vectors).toEqual([vectorFor("a"), vectorFor("bb"), vectorFor("ccc")]);
		expect(calls.map((call) => call.url)).toEqual(["/v1/embeddings", "/v1/embeddings"]);
		expect(calls.map((call) => (call.body.input as string[]).length)).toEqual([2, 1]);
		expect(calls[0]?.body.model).toBe("m");
	});

	it("does not double a trailing /v1 on the base URL", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({
			body: { data: (body.input as string[]).map((text, index) => ({ index, embedding: vectorFor(text) })) },
		}));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl: `${baseUrl}/v1/`,
			model: "m",
			dimension: 3,
			provider: "openai",
		});
		await embedder.embed(["a"]);
		expect(calls[0]?.url).toBe("/v1/embeddings");
	});

	it("sends the bearer key read from the named environment variable at call time", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({
			body: { data: (body.input as string[]).map((text, index) => ({ index, embedding: vectorFor(text) })) },
		}));
		const env: NodeJS.ProcessEnv = { REMOTE_EMBED_KEY: "sk-live" };
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "openai",
			apiKeyEnv: "REMOTE_EMBED_KEY",
			env,
		});
		await embedder.embed(["a"]);
		expect(calls[0]?.authorization).toBe("Bearer sk-live");
	});

	it("names the missing variable instead of sending an unauthenticated request", async () => {
		const { baseUrl, calls } = await serve(() => ({ body: { data: [] } }));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "openai",
			apiKeyEnv: "REMOTE_EMBED_KEY",
			env: {},
		});
		await expect(embedder.embed(["a"])).rejects.toThrow(/REMOTE_EMBED_KEY/);
		expect(calls).toHaveLength(0);
	});

	it("reports the HTTP status and the response body verbatim on failure", async () => {
		const { baseUrl } = await serve(() => ({
			status: 413,
			body: '{"error":{"message":"input too long for model"}}',
		}));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "openai",
		});
		await expect(embedder.embed(["a"])).rejects.toThrow(/413.*input too long for model/s);
	});

	it("rejects a vector whose length differs from the declared dimension", async () => {
		const { baseUrl } = await serve(({ body }) => ({
			body: { data: (body.input as string[]).map((_, index) => ({ index, embedding: [1, 2] })) },
		}));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "openai",
		});
		await expect(embedder.embed(["a"])).rejects.toThrow(/dimension/i);
	});

	it("learns the dimension from the first response when none is configured", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({
			body: { data: (body.input as string[]).map((text, index) => ({ index, embedding: vectorFor(text) })) },
		}));
		const embedder = createHttpEmbedder({ protocol: "openai", baseUrl, model: "m", provider: "openai" });
		expect(await embedder.identity()).toEqual({ provider: "openai", model: "m", dimension: 3 });
		expect(calls).toHaveLength(1);
	});
});

describe("createHttpEmbedder tei protocol", () => {
	it("posts {inputs} to /embed and reads a bare array of vectors", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({
			body: (body.inputs as string[]).map(vectorFor),
		}));
		const embedder = createHttpEmbedder({
			protocol: "tei",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "tei",
		});
		expect(await embedder.embed(["a", "bb"])).toEqual([vectorFor("a"), vectorFor("bb")]);
		expect(calls[0]?.url).toBe("/embed");
		expect(calls[0]?.body.inputs).toEqual(["a", "bb"]);
	});
});

describe("createHttpEmbedder tei truncation", () => {
	it("asks the server to truncate over-long inputs only when configured", async () => {
		const { baseUrl, calls } = await serve(({ body }) => ({ body: (body.inputs as string[]).map(vectorFor) }));
		const plain = createHttpEmbedder({ protocol: "tei", baseUrl, model: "m", dimension: 3, provider: "tei" });
		const truncating = createHttpEmbedder({
			protocol: "tei",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "tei",
			truncate: true,
		});
		await plain.embed(["a"]);
		await truncating.embed(["a"]);
		expect(calls[0]?.body.truncate).toBeUndefined();
		expect(calls[1]?.body.truncate).toBe(true);
	});
});

describe("createHttpEmbedder empty input", () => {
	it("returns no vectors without a request", async () => {
		const { baseUrl, calls } = await serve(() => ({ body: { data: [] } }));
		const embedder = createHttpEmbedder({
			protocol: "openai",
			baseUrl,
			model: "m",
			dimension: 3,
			provider: "openai",
		});
		expect(await embedder.embed([])).toEqual([]);
		expect(calls).toHaveLength(0);
	});
});
