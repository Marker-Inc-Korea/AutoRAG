import { describe, expect, it } from "vitest";
import {
	LazykatokBm25Method,
	type LazykatokSearchClient,
	LazykatokSemanticMethod,
} from "../../../../src/datasource/skills/lazykatok/methods.ts";
import type {
	LazykatokFailureReason,
	LazykatokHit,
	LazykatokSearchMode,
	LazykatokSearchOptions,
	LazykatokSearchResult,
} from "../../../../src/datasource/skills/lazykatok/types.ts";

interface SearchCall {
	readonly mode: LazykatokSearchMode;
	readonly query: string;
	readonly options?: LazykatokSearchOptions;
}

class StubSearchClient implements LazykatokSearchClient {
	public readonly calls: SearchCall[] = [];
	public hits: readonly LazykatokHit[] = [];
	public failReason: LazykatokFailureReason | null = null;
	public throwError: Error | null = null;

	async search(
		mode: LazykatokSearchMode,
		query: string,
		options?: LazykatokSearchOptions,
	): Promise<LazykatokSearchResult> {
		this.calls.push({ mode, query, options });
		if (this.throwError !== null) throw this.throwError;
		if (this.failReason !== null) {
			return {
				ok: false,
				reason: this.failReason,
				hits: [],
				stdout: "",
				stderr: "lazykatok: unavailable",
				code: null,
			};
		}
		return { ok: true, hits: this.hits, data: { hits: this.hits }, stdout: "", stderr: "", code: 0 };
	}
}

const INSTANCE_ID = "default";

const BASE_HITS: readonly LazykatokHit[] = [
	{
		chunkId: "chunk-001",
		content: "refund policy approval workflow",
		score: 2.1,
		source: "/kakao/default/chunks/chunk-001",
		metadata: { room: "support-ops" },
	},
	{
		chunkId: "chunk-002",
		content: "chargeback dispute evidence packet",
		score: 1.7,
		source: "/kakao/default/chunks/chunk-002",
		metadata: { room: "finance" },
	},
	{
		chunkId: "chunk-003",
		content: "refund partial approval notes",
		score: 1.2,
		source: "/kakao/default/chunks/chunk-003",
		metadata: { room: "support-ops" },
	},
];

function makeClient(): StubSearchClient {
	const client = new StubSearchClient();
	client.hits = BASE_HITS;
	return client;
}

describe("LazykatokBm25Method descriptor", () => {
	it("exposes kakao-bm25 name, bm25 type, kakao datasource id, and pii tags", () => {
		const method = new LazykatokBm25Method({ client: makeClient(), instanceId: INSTANCE_ID });
		const descriptor = method.describe();

		expect(descriptor.name).toBe("kakao-bm25");
		expect(descriptor.type).toBe("bm25");
		expect(descriptor.datasourceId).toBe("kakao");
		expect(descriptor.status).toBe("active");
		expect(descriptor.tags).toEqual(expect.arrayContaining(["kakaotalk", "personal", "pii"]));
		expect(descriptor.capabilities.length).toBeGreaterThan(0);
	});

	it("forwards custom tags when provided", () => {
		const method = new LazykatokBm25Method({
			client: makeClient(),
			instanceId: INSTANCE_ID,
			tags: ["kakaotalk", "team"],
		});
		expect(method.describe().tags).toEqual(["kakaotalk", "team"]);
	});
});

describe("LazykatokSemanticMethod descriptor", () => {
	it("exposes kakao-semantic name and vector type", () => {
		const method = new LazykatokSemanticMethod({ client: makeClient(), instanceId: INSTANCE_ID });
		const descriptor = method.describe();

		expect(descriptor.name).toBe("kakao-semantic");
		expect(descriptor.type).toBe("vector");
		expect(descriptor.datasourceId).toBe("kakao");
		expect(descriptor.status).toBe("active");
	});
});

describe("LazykatokBm25Method retrieve", () => {
	it("maps hits to the canonical slash datasource source", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10 });

		expect(results.map((r) => r.source)).toEqual([
			"/kakao/default/chunks/chunk-001",
			"/kakao/default/chunks/chunk-002",
			"/kakao/default/chunks/chunk-003",
		]);
		for (const result of results) {
			expect(result.source.startsWith("/")).toBe(true);
			expect(result.source.startsWith("kakao:")).toBe(false);
			expect(result.id).toBe(`kakao:${INSTANCE_ID}:${result.metadata.chunkId}`);
		}
	});

	it("calls client.search in keyword mode with the trimmed query and topK", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		await method.retrieve("  refund  ", { topK: 5 });

		expect(client.calls).toEqual([{ mode: "keyword", query: "refund", options: { topK: 5, signal: undefined } }]);
	});

	it("attaches method, datasourceId, instanceId, mode, and chunkId metadata", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const [first] = await method.retrieve("refund", { topK: 1 });

		expect(first).toBeDefined();
		expect(first?.metadata).toMatchObject({
			method: "kakao-bm25",
			datasourceId: "kakao",
			instanceId: INSTANCE_ID,
			mode: "keyword",
			chunkId: "chunk-001",
			room: "support-ops",
		});
		expect(first?.score).toBe(2.1);
		expect(first?.content).toBe("refund policy approval workflow");
	});

	it("limits results to topK", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 2 });

		expect(results).toHaveLength(2);
	});

	it("returns [] for an empty query without calling the client", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("   ", {});

		expect(results).toEqual([]);
		expect(client.calls).toHaveLength(0);
	});

	it("keeps hits for the kakao instance scope", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10, scope: "/kakao/default" });

		expect(results).toHaveLength(3);
	});

	it("ignores source scope because lazykatok has no scope capability", async () => {
		const client = makeClient();
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10, scope: "/kakao/other" });

		expect(results).toHaveLength(3);
	});

	it("surfaces the lazykatok failure instead of answering with an empty result set", async () => {
		const client = makeClient();
		client.failReason = "binary-missing";
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		// The CLI's own words reach the retrieval pipeline, which reports kakao as unsearched.
		await expect(method.retrieve("refund", { topK: 5 })).rejects.toThrow("lazykatok: unavailable");
		expect(client.calls).toHaveLength(1);
	});

	it("propagates a thrown client error unchanged", async () => {
		const client = makeClient();
		client.throwError = new Error("spawn ENOENT");
		const method = new LazykatokBm25Method({ client, instanceId: INSTANCE_ID });

		await expect(method.retrieve("refund", { topK: 5 })).rejects.toThrow("spawn ENOENT");
	});
});

describe("LazykatokSemanticMethod retrieve", () => {
	it("calls client.search in semantic mode", async () => {
		const client = makeClient();
		const method = new LazykatokSemanticMethod({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("chargeback", { topK: 2 });

		expect(client.calls).toEqual([
			{ mode: "semantic", query: "chargeback", options: { topK: 2, signal: undefined } },
		]);
		expect(results[0]?.metadata).toMatchObject({
			method: "kakao-semantic",
			mode: "semantic",
			datasourceId: "kakao",
		});
	});

	it("surfaces a failed semantic search with the CLI reason", async () => {
		const client = makeClient();
		client.failReason = "nonzero-exit";
		const method = new LazykatokSemanticMethod({ client, instanceId: INSTANCE_ID });

		await expect(method.retrieve("chargeback", {})).rejects.toThrow(
			"kakao search --mode semantic failed (nonzero-exit",
		);
	});
});
