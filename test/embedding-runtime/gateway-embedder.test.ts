import { describe, expect, it } from "vitest";
import { createGatewayEmbedder, EMBED_BATCH_SIZE } from "../../src/embedding-runtime/gateway-embedder.ts";

interface EmbedBody {
	readonly input: string[];
}

/** Shapes the JSON bodies this test's own fetch stub records. */
function parseEmbedBody(body: string): EmbedBody {
	return JSON.parse(body) as EmbedBody;
}

describe("createGatewayEmbedder", () => {
	it("batches large embed requests across multiple gateway calls", async () => {
		const bodies: string[] = [];
		const fetchImpl = (async (_input: unknown, init?: RequestInit) => {
			const body = String(init?.body ?? "{}");
			bodies.push(body);
			const parsed = parseEmbedBody(body);
			return new Response(
				JSON.stringify({ data: parsed.input.map((_, index) => ({ index, embedding: [1, 0, 0] })) }),
				{ status: 200 },
			);
		}) as unknown as typeof fetch;
		const embedder = createGatewayEmbedder({
			runtime: {
				ensureRuntime: async () => ({
					baseUrl: "http://127.0.0.1:9",
					profile: { queryPrefix: "", passagePrefix: "" },
					identity: { provider: "p", model: "m", dimension: 3 },
				}),
			},
			fetchImpl,
		});
		const texts = Array.from({ length: EMBED_BATCH_SIZE * 2 + 6 }, (_, index) => `text-${index}`);
		const vectors = await embedder.embed(texts);
		expect(vectors).toHaveLength(texts.length);
		expect(bodies.length).toBe(3);
		const batchSizes = bodies.map((body) => parseEmbedBody(body).input.length);
		expect(batchSizes).toEqual([EMBED_BATCH_SIZE, EMBED_BATCH_SIZE, 6]);
		// Order is preserved across batches: first text of batch 3 follows batch 2.
		const thirdBody = parseEmbedBody(bodies[2] ?? "{}");
		expect(thirdBody.input[0]).toBe(`text-${EMBED_BATCH_SIZE * 2}`);
	});
});

describe("createGatewayEmbedder roles", () => {
	function recordingEmbedder(prefixes: { queryPrefix: string; passagePrefix: string }) {
		const inputs: string[][] = [];
		const fetchImpl = (async (_input: unknown, init?: RequestInit) => {
			const parsed = parseEmbedBody(String(init?.body ?? "{}"));
			inputs.push(parsed.input);
			return new Response(
				JSON.stringify({ data: parsed.input.map((_, index) => ({ index, embedding: [1, 0, 0] })) }),
				{ status: 200 },
			);
		}) as unknown as typeof fetch;
		const embedder = createGatewayEmbedder({
			runtime: {
				ensureRuntime: async () => ({
					baseUrl: "http://127.0.0.1:9",
					profile: prefixes,
					identity: { provider: "p", model: "m", dimension: 3 },
				}),
			},
			fetchImpl,
		});
		return { embedder, inputs };
	}

	const qwen = { queryPrefix: "Instruct: find it\nQuery:", passagePrefix: "" };

	it('prepends the profile query prefix for role "query" and sends documents unchanged', async () => {
		const { embedder, inputs } = recordingEmbedder(qwen);
		await embedder.embed(["refund policy"], { role: "query" });
		await embedder.embed(["refund policy"], { role: "document" });
		expect(inputs).toEqual([["Instruct: find it\nQuery:refund policy"], ["refund policy"]]);
	});

	it("treats a call without a role as document embedding", async () => {
		const { embedder, inputs } = recordingEmbedder(qwen);
		await embedder.embed(["refund policy"]);
		expect(inputs).toEqual([["refund policy"]]);
	});

	it("applies the passage prefix to documents and the query prefix to every query in a batch", async () => {
		const { embedder, inputs } = recordingEmbedder({ queryPrefix: "q: ", passagePrefix: "p: " });
		await embedder.embed(["a", "b"], { role: "document" });
		await embedder.embed(["a", "b"], { role: "query" });
		expect(inputs).toEqual([
			["p: a", "p: b"],
			["q: a", "q: b"],
		]);
	});

	it("sends an empty query prefix verbatim", async () => {
		const { embedder, inputs } = recordingEmbedder({ queryPrefix: "", passagePrefix: "" });
		await embedder.embed(["refund"], { role: "query" });
		expect(inputs).toEqual([["refund"]]);
	});
});
