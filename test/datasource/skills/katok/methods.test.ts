import { describe, expect, it } from "vitest";
import {
	KatokBm25Method,
	type KatokSearchClient,
	KatokSemanticMethod,
} from "../../../../src/datasource/skills/katok/methods.ts";
import type {
	KatokFailureReason,
	KatokHit,
	KatokSearchMode,
	KatokSearchOptions,
	KatokSearchResult,
} from "../../../../src/datasource/skills/katok/types.ts";

interface SearchCall {
	readonly mode: KatokSearchMode;
	readonly query: string;
	readonly options?: KatokSearchOptions;
}

class StubSearchClient implements KatokSearchClient {
	public readonly calls: SearchCall[] = [];
	public hits: readonly KatokHit[] = [];
	public failReason: KatokFailureReason | null = null;
	public throwError: Error | null = null;

	async search(mode: KatokSearchMode, query: string, options?: KatokSearchOptions): Promise<KatokSearchResult> {
		this.calls.push({ mode, query, options });
		if (this.throwError !== null) throw this.throwError;
		if (this.failReason !== null) {
			return {
				ok: false,
				reason: this.failReason,
				hits: [],
				stdout: "",
				stderr: "katok: unavailable",
				code: null,
			};
		}
		return { ok: true, hits: this.hits, data: { hits: this.hits }, stdout: "", stderr: "", code: 0 };
	}
}

const INSTANCE_ID = "default";

const BASE_HITS: readonly KatokHit[] = [
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

describe("KatokBm25Method descriptor", () => {
	it("exposes kakao-bm25 name, bm25 type, kakao datasource id, and pii tags", () => {
		const method = new KatokBm25Method({ client: makeClient(), instanceId: INSTANCE_ID });
		const descriptor = method.describe();

		expect(descriptor.name).toBe("kakao-bm25");
		expect(descriptor.type).toBe("bm25");
		expect(descriptor.datasourceId).toBe("kakao");
		expect(descriptor.status).toBe("active");
		expect(descriptor.tags).toEqual(expect.arrayContaining(["kakaotalk", "personal", "pii"]));
		expect(descriptor.capabilities.length).toBeGreaterThan(0);
	});

	it("forwards custom tags when provided", () => {
		const method = new KatokBm25Method({
			client: makeClient(),
			instanceId: INSTANCE_ID,
			tags: ["kakaotalk", "team"],
		});
		expect(method.describe().tags).toEqual(["kakaotalk", "team"]);
	});
});

describe("KatokSemanticMethod descriptor", () => {
	it("exposes kakao-semantic name and vector type", () => {
		const method = new KatokSemanticMethod({ client: makeClient(), instanceId: INSTANCE_ID });
		const descriptor = method.describe();

		expect(descriptor.name).toBe("kakao-semantic");
		expect(descriptor.type).toBe("vector");
		expect(descriptor.datasourceId).toBe("kakao");
		expect(descriptor.status).toBe("active");
	});
});

describe("KatokBm25Method retrieve", () => {
	it("maps hits to slash-hierarchical /kakao/<instance>/chunks/<chunk-id> sources", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10 });

		expect(results.map((r) => r.source)).toEqual([
			"/kakao/default/chunks/chunk-001",
			"/kakao/default/chunks/chunk-002",
			"/kakao/default/chunks/chunk-003",
		]);
		for (const result of results) {
			expect(result.source).toMatch(/^\/kakao\/default\/chunks\/[^/]+$/u);
			expect(result.id).toBe(`kakao:${INSTANCE_ID}:${result.metadata.chunkId}`);
		}
	});

	it("calls client.search in keyword mode with the trimmed query and topK", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		await method.retrieve("  refund  ", { topK: 5 });

		expect(client.calls).toEqual([{ mode: "keyword", query: "refund", options: { topK: 5, signal: undefined } }]);
	});

	it("attaches method, datasourceId, instanceId, mode, and chunkId metadata", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

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
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 2 });

		expect(results).toHaveLength(2);
	});

	it("returns [] for an empty query without calling the client", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("   ", {});

		expect(results).toEqual([]);
		expect(client.calls).toHaveLength(0);
	});

	it("filters by folder scope", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10, scope: "/kakao/default" });

		expect(results).toHaveLength(3);
	});

	it("filters to a single chunk scope", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", {
			topK: 10,
			scope: "/kakao/default/chunks/chunk-002",
		});

		expect(results.map((r) => r.source)).toEqual(["/kakao/default/chunks/chunk-002"]);
	});

	it("returns [] when scope targets a different instance", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 10, scope: "/kakao/other" });

		expect(results).toEqual([]);
	});

	it("intersects allowedScopes against each source", async () => {
		const client = makeClient();
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", {
			topK: 10,
			allowedScopes: ["/kakao/default/chunks/chunk-001", "/kakao/default/chunks/chunk-003"],
		});

		expect(results.map((r) => r.source)).toEqual([
			"/kakao/default/chunks/chunk-001",
			"/kakao/default/chunks/chunk-003",
		]);
	});

	it("returns [] without throwing when search yields a failed result", async () => {
		const client = makeClient();
		client.failReason = "binary-missing";
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 5 });

		expect(results).toEqual([]);
		expect(client.calls).toHaveLength(1);
	});

	it("returns [] without throwing when search throws", async () => {
		const client = makeClient();
		client.throwError = new Error("spawn ENOENT");
		const method = new KatokBm25Method({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("refund", { topK: 5 });

		expect(results).toEqual([]);
	});
});

describe("KatokSemanticMethod retrieve", () => {
	it("calls client.search in semantic mode", async () => {
		const client = makeClient();
		const method = new KatokSemanticMethod({ client, instanceId: INSTANCE_ID });

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

	it("returns [] without throwing on a failed semantic search", async () => {
		const client = makeClient();
		client.failReason = "remote-embedding-rejected";
		const method = new KatokSemanticMethod({ client, instanceId: INSTANCE_ID });

		const results = await method.retrieve("chargeback", {});

		expect(results).toEqual([]);
	});
});
