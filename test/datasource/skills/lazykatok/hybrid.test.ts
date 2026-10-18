import { describe, expect, it } from "vitest";
import { LazykatokSkill, type LazykatokSkillClient } from "../../../../src/datasource/skills/lazykatok/skill.ts";
import type {
	LazykatokChunk,
	LazykatokHit,
	LazykatokParentResult,
	LazykatokSearchMode,
	LazykatokSearchResult,
} from "../../../../src/datasource/skills/lazykatok/types.ts";

/** lazykatok searches keywords over micro-chunks and semantics over their parent windows. */
class StubClient {
	public keywordHits: readonly LazykatokHit[] = [];
	public semanticHits: readonly LazykatokHit[] = [];
	public semanticFails = false;
	public windows = new Map<string, LazykatokChunk>();
	public failingParents = new Set<string>();
	public parentCalls: string[] = [];
	public inFlight = 0;
	public maxInFlight = 0;

	async search(mode: LazykatokSearchMode): Promise<LazykatokSearchResult> {
		if (mode === "semantic" && this.semanticFails) {
			return { ok: false, reason: "nonzero-exit", hits: [], stdout: "", stderr: "semantic_index_stale", code: 1 };
		}
		const hits = mode === "semantic" ? this.semanticHits : this.keywordHits;
		return { ok: true, hits, data: { hits }, stdout: "", stderr: "", code: 0 };
	}

	async parent(chunkId: string): Promise<LazykatokParentResult> {
		this.parentCalls.push(chunkId);
		this.inFlight += 1;
		this.maxInFlight = Math.max(this.maxInFlight, this.inFlight);
		await new Promise((resolve) => setTimeout(resolve, 1));
		this.inFlight -= 1;
		const window = this.windows.get(chunkId);
		if (this.failingParents.has(chunkId) || window === undefined) {
			return { ok: false, reason: "nonzero-exit", stdout: "", stderr: `no parent for ${chunkId}`, code: 1 };
		}
		return { ok: true, data: [window], stdout: "", stderr: "", code: 0 };
	}
}

function hit(chunkId: string, score: number, content = chunkId): LazykatokHit {
	return { chunkId, score, content, metadata: { chat_name: "room" } };
}

function window(chunkId: string): LazykatokChunk {
	return { chunkId, content: `text of ${chunkId}`, metadata: { chat_name: "room", message_count: 4 } };
}

function hybrid(stub: StubClient) {
	const skill = new LazykatokSkill({ client: stub as unknown as LazykatokSkillClient, instanceId: "default" });
	const method = skill.retrievalMethods().find((candidate) => candidate.describe().name === "kakao-hybrid");
	if (method === undefined) throw new Error("kakao-hybrid not registered");
	return method;
}

describe("kakao-hybrid (parent-window fusion)", () => {
	it("fuses a keyword chunk with the semantic window that contains it", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9), hit("chunk_b", 5)];
		stub.windows.set("chunk_a", window("window_1"));
		stub.windows.set("chunk_b", window("window_2"));
		stub.semanticHits = [hit("window_2", 0.9), hit("window_3", 0.8)];

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		// window_2 is the only window both rankings agree on; it outranks window_1 (keyword only, rank 1).
		expect(results.map((r) => r.id)).toEqual([
			"kakao:default:window_2",
			"kakao:default:window_1",
			"kakao:default:window_3",
		]);
		expect(results.map((r) => r.source)).toEqual([
			"/kakao/default/chunks/window_2",
			"/kakao/default/chunks/window_1",
			"/kakao/default/chunks/window_3",
		]);
		expect(results[0]?.score).toBeCloseTo(1 / 62 + 1 / 61, 10);
		expect(results[0]?.metadata).toMatchObject({ method: "kakao-hybrid", mode: "hybrid", instanceId: "default" });
	});

	it("returns the window text for a keyword-only window rather than the matching chunk", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9, "short matching message")];
		stub.windows.set("chunk_a", window("window_1"));
		stub.semanticHits = [hit("window_9", 0.5)];

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		const lifted = results.find((r) => r.id === "kakao:default:window_1");
		expect(lifted?.content).toBe("text of window_1");
		expect(lifted?.metadata).toMatchObject({ chunkId: "window_1", matchedChunkId: "chunk_a" });
	});

	it("counts several chunks of one window once, at the best chunk's rank", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9), hit("chunk_b", 8), hit("chunk_c", 7)];
		stub.windows.set("chunk_a", window("window_1"));
		stub.windows.set("chunk_b", window("window_1"));
		stub.windows.set("chunk_c", window("window_2"));
		stub.semanticHits = [hit("window_2", 0.9)];

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		// window_1 holds the two best chunks but is one vote at rank 1; window_2 is rank 2 lexically
		// (not 3) and also semantically rank 1, so it wins.
		expect(results.map((r) => r.id)).toEqual(["kakao:default:window_2", "kakao:default:window_1"]);
		expect(results[0]?.score).toBeCloseTo(1 / 62 + 1 / 61, 10);
		expect(results[1]?.score).toBeCloseTo(1 / 61, 10);
	});

	it("keeps the keyword ranking and score scale when semantic search is unavailable", async () => {
		const stub = new StubClient();
		stub.semanticFails = true;
		stub.keywordHits = [hit("chunk_a", 9), hit("chunk_b", 5)];
		stub.windows.set("chunk_a", window("window_1"));
		stub.windows.set("chunk_b", window("window_2"));

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		expect(results.map((r) => [r.id, r.score])).toEqual([
			["kakao:default:window_1", 9],
			["kakao:default:window_2", 5],
		]);
	});

	it("keeps a chunk under its own identity when only its parent lookup fails", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9), hit("chunk_b", 5)];
		stub.windows.set("chunk_a", window("window_1"));
		stub.failingParents.add("chunk_b");
		stub.semanticFails = true;

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		expect(results.map((r) => r.id)).toEqual(["kakao:default:window_1", "kakao:default:chunk_b"]);
	});

	it("falls back to semantic windows when no keyword chunk resolves to a parent", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9), hit("chunk_b", 5)];
		stub.failingParents = new Set(["chunk_a", "chunk_b"]);
		stub.semanticHits = [hit("window_7", 0.9)];

		const results = await hybrid(stub).retrieve("refund", { topK: 5 });

		expect(results.map((r) => r.id)).toEqual(["kakao:default:window_7"]);
	});

	it("reports the parent lookup failure when neither input can produce a ranking", async () => {
		const stub = new StubClient();
		stub.keywordHits = [hit("chunk_a", 9)];
		stub.failingParents.add("chunk_a");
		stub.semanticFails = true;

		await expect(hybrid(stub).retrieve("refund", { topK: 5 })).rejects.toThrow(
			/chunk parent[\s\S]*semantic_index_stale/,
		);
	});

	it("bounds concurrent parent lookups and resolves each chunk once", async () => {
		const stub = new StubClient();
		const chunks = Array.from({ length: 24 }, (_, index) => `chunk_${index}`);
		stub.keywordHits = chunks.map((chunkId, index) => hit(chunkId, 100 - index));
		for (const [index, chunkId] of chunks.entries()) stub.windows.set(chunkId, window(`window_${index}`));
		stub.semanticHits = [];

		const results = await hybrid(stub).retrieve("refund", { topK: 8 });

		expect(results).toHaveLength(8);
		expect(stub.parentCalls).toHaveLength(24);
		expect(new Set(stub.parentCalls).size).toBe(24);
		expect(stub.maxInFlight).toBeLessThanOrEqual(8);
		expect(stub.maxInFlight).toBeGreaterThan(1);
	});
});
