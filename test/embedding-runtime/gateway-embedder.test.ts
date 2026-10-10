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
