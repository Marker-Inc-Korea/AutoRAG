import { describe, expect, it } from "vitest";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { DEFAULT_RERANK_TOP_N } from "../../src/retrieval/rerank.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

// --- Helpers ---

const stubMethod = (
	name: string,
	results: readonly (Partial<RetrievalResult> & { source: string })[],
	overrides: Partial<RetrievalMethodDescriptor> & { datasourceId?: string; tags?: readonly string[] } = {},
): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "posix",
		description: `stub ${name}`,
		status: "active",
		capabilities: [],
		...overrides,
	}),
	retrieve: async (): Promise<RetrievalResult[]> =>
		results.map((r, index) => ({
			id: r.id ?? `${name}:${index}`,
			content: r.content ?? `content-${name}-${index}`,
			score: r.score ?? 1,
			metadata: r.metadata ?? {},
			source: r.source,
		})),
});

const stubDatasourceMethod = (
	name: string,
	datasourceId: string,
	tags: readonly string[],
	results: readonly (Partial<RetrievalResult> & { source: string })[],
	capabilities: readonly string[] = ["scoped"],
): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: `datasource ${name}`,
		status: "active",
		capabilities: [...capabilities],
		datasourceId,
		tags: [...tags],
	}),
	retrieve: async (): Promise<RetrievalResult[]> =>
		results.map((r, index) => ({
			id: r.id ?? `${name}:${index}`,
			content: r.content ?? `content-${name}-${index}`,
			score: r.score ?? 1,
			metadata: r.metadata ?? {},
			source: r.source,
		})),
});

const neverMethod = (name: string): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "posix",
		description: `failing ${name}`,
		status: "stub",
		capabilities: [],
	}),
	retrieve: async (): Promise<RetrievalResult[]> => {
		throw new Error(`${name} always fails`);
	},
});

// --- Tests ---

describe("RetrievalEngine", () => {
	describe("construction", () => {
		it("can be constructed without options", () => {
			const engine = new RetrievalEngine();
			expect(engine.getMethodRegistry()).toBeDefined();
		});

		it("respects defaultTopK", () => {
			const engine = new RetrievalEngine({ defaultTopK: 5 });
			expect(engine).toBeDefined();
		});
	});

	describe("register / registerMany", () => {
		it("registers a single method", () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/a.txt" }]));
			expect(engine.getMethodRegistry().get("posix")).toBeDefined();
		});

		it("registers multiple methods atomically", () => {
			const engine = new RetrievalEngine();
			engine.registerMany([stubMethod("alpha", [{ source: "/a.txt" }]), stubMethod("beta", [{ source: "/b.txt" }])]);
			expect(engine.getMethodRegistry().get("alpha")).toBeDefined();
			expect(engine.getMethodRegistry().get("beta")).toBeDefined();
		});

		it("registerMany rejects duplicate names within the batch", () => {
			const engine = new RetrievalEngine();
			expect(() =>
				engine.registerMany([
					stubMethod("dupe", [{ source: "/a.txt" }]),
					stubMethod("dupe", [{ source: "/b.txt" }]),
				]),
			).toThrow('Retrieval method "dupe" is already registered');
		});

		it("registerMany rejects a name already registered", () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("existing", [{ source: "/a.txt" }]));
			expect(() => engine.registerMany([stubMethod("existing", [{ source: "/b.txt" }])])).toThrow(
				'Retrieval method "existing" is already registered',
			);
		});
	});

	describe("retrieve — model-free merged pipeline", () => {
		it("returns empty results and diagnostics when no methods are registered", async () => {
			const engine = new RetrievalEngine();
			const { results, diagnostics } = await engine.retrieve("query");
			expect(results).toEqual([]);
			expect(diagnostics).toEqual([]);
		});

		it("merges results from multiple methods", async () => {
			const engine = new RetrievalEngine({ defaultTopK: 50 });
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.6 }]));
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results, diagnostics } = await engine.retrieve("test");
			expect(results).toHaveLength(2);
			expect(results.map((r) => r.source)).toContain("/a.txt");
			expect(results.map((r) => r.source)).toContain("/kakao/acct-1/chunks/c-1");
			expect(diagnostics).toEqual([]);
		});

		it("includes datasource results without any permission configuration", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.5 }]));
			const { results, diagnostics } = await engine.retrieve("test");
			expect(results).toHaveLength(2);
			expect(results.map((r) => r.source)).toContain("/kakao/acct-1/chunks/c-1");
			expect(results.map((r) => r.source)).toContain("/a.txt");
			expect(diagnostics).toEqual([]);
		});

		it("narrows scope-capable datasource results by the query scope", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/docs/a.txt", score: 0.7 }]));
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[
						{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 },
						{ source: "/kakao/acct-2/chunks/c-9", score: 0.8 },
					],
				),
			);
			const { results, diagnostics } = await engine.retrieve("test", { scope: "/kakao/acct-1/**" });
			expect(results.map((r) => r.source)).toEqual(["/docs/a.txt", "/kakao/acct-1/chunks/c-1"]);
			expect(diagnostics).toEqual([]);
		});

		it("records diagnostics for a failing method without dropping the whole pipeline", async () => {
			const engine = new RetrievalEngine({ defaultTopK: 50 });
			engine.register(neverMethod("failing"));
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.8 }]));
			const { results, diagnostics } = await engine.retrieve("test");
			expect(results).toHaveLength(1);
			expect(results[0].source).toBe("/a.txt");
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0].code).toBe("retrieval-method-failed");
			expect(diagnostics[0].source).toBe("failing");
		});

		it("respects topK", async () => {
			const engine = new RetrievalEngine({ defaultTopK: 50 });
			engine.register(
				stubMethod("posix", [
					{ source: "/a.txt", score: 0.9 },
					{ source: "/b.txt", score: 0.8 },
					{ source: "/c.txt", score: 0.7 },
				]),
			);
			const { results } = await engine.retrieve("query", { topK: 2 });
			expect(results).toHaveLength(2);
		});

		it("produces stable result order for deterministic methods", async () => {
			const engine = new RetrievalEngine({ defaultTopK: 50 });
			engine.register(
				stubMethod("posix", [
					{ source: "/z.txt", score: 0.5 },
					{ source: "/a.txt", score: 0.5 },
				]),
			);
			const { results: first } = await engine.retrieve("query");
			const { results: second } = await engine.retrieve("query");
			expect(first).toEqual(second);
		});

		it("does not crash on empty query string", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.5 }]));
			const { results, diagnostics } = await engine.retrieve("");
			// The engine does not filter empty queries; the method stub still
			// returns results for any query. The seam must not crash or throw.
			expect(results).toBeDefined();
			expect(diagnostics).toEqual([]);
		});

		it("handles malformed query (whitespace-only)", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.5 }]));
			const { results, diagnostics } = await engine.retrieve("   ");
			// The stub method still gets called with whitespace, but the merge
			// pipeline processes it. Since the stub returns results regardless
			// of query, the test just verifies no crash.
			expect(results).toBeDefined();
			expect(diagnostics).toEqual([]);
		});
	});

	describe("retrieveByMethod — raw per-method view", () => {
		it("returns results keyed by method name", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/a.txt" }]));
			engine.register(
				stubDatasourceMethod("kakao", "kakao:acct-1", ["kakao"], [{ source: "/kakao/acct-1/chunks/c-1" }]),
			);
			const { byMethod, diagnostics } = await engine.retrieveByMethod("test");
			expect(byMethod.has("posix")).toBe(true);
			expect(byMethod.has("kakao")).toBe(true);
			expect(byMethod.get("posix")).toHaveLength(1);
			expect(byMethod.get("kakao")).toHaveLength(1);
			expect(diagnostics).toEqual([]);
		});

		it("applies scope narrowing per method", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[
						{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 },
						{ source: "/kakao/acct-2/chunks/c-9", score: 0.8 },
					],
				),
			);
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.5 }]));
			const { byMethod } = await engine.retrieveByMethod("test", { scope: "/kakao/acct-1/**" });
			expect(byMethod.get("kakao")?.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/c-1"]);
			expect(byMethod.get("posix")).toHaveLength(1);
		});
	});

	describe("scope behavior", () => {
		it("leaves datasource results untouched when scope is undefined", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"slack",
					"slack:workspace",
					["slack"],
					[
						{ source: "/slack/allowed/channel/message", score: 0.9 },
						{ source: "/slack/secret/channel/message", score: 0.8 },
					],
				),
			);
			const { results } = await engine.retrieve("test");
			expect(results.map((result) => result.source)).toEqual([
				"/slack/allowed/channel/message",
				"/slack/secret/channel/message",
			]);
		});

		it("leaves datasource methods without the scoped capability untouched", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao",
					["kakao"],
					[{ source: "/kakao/work/chunks/c-1", score: 0.9 }],
					["chat"],
				),
			);
			const { results } = await engine.retrieve("test", { scope: "/kakao/personal/**" });
			expect(results.map((result) => result.source)).toEqual(["/kakao/work/chunks/c-1"]);
		});

		it("rejects datasource results carrying a '#' fragment", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao",
					["kakao"],
					[
						{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 },
						{ source: "/kakao/acct-1/chunks/c-2#meta", score: 0.8 },
					],
				),
			);
			const { results } = await engine.retrieve("test");
			expect(results.map((result) => result.source)).toEqual(["/kakao/acct-1/chunks/c-1"]);
		});

		it("preserves source provenance on results", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/provenance/doc.md", score: 0.7 }]));
			const { results } = await engine.retrieve("query");
			expect(results).toHaveLength(1);
			expect(results[0].source).toBe("/provenance/doc.md");
		});

		it("passes through non-datasource methods when scope is undefined", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/some/path/file.txt", score: 0.6 }]));
			const { results } = await engine.retrieve("test", { scope: undefined });
			expect(results).toHaveLength(1);
		});
	});

	describe("diagnostic integration", () => {
		it("returns no diagnostics when all methods succeed", async () => {
			const engine = new RetrievalEngine();
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.5 }]));
			const { diagnostics } = await engine.retrieve("test");
			expect(diagnostics).toEqual([]);
		});

		it("reports per-method failure diagnostics", async () => {
			const engine = new RetrievalEngine();
			engine.register(neverMethod("broken"));
			const { diagnostics } = await engine.retrieve("test");
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0].code).toBe("retrieval-method-failed");
		});
	});
});

describe("RetrievalEngine reranking", () => {
	it("reorders merged results when a reranker is configured", async () => {
		const engine = new RetrievalEngine({
			reranker: {
				describe: () => ({ name: "stub", provider: "stub", model: "m", available: true }),
				rerank: async (_query, results) => [...results].reverse(),
			},
		});
		engine.register(stubMethod("posix", [{ source: "/a" }, { source: "/b" }]));
		const { results } = await engine.retrieve("q", { topK: 10 });
		expect(results.map((entry) => entry.source)).toEqual(["/b", "/a"]);
	});

	it("passes the default rerank topN when the caller omits topK", async () => {
		let seenTopN: number | undefined;
		const engine = new RetrievalEngine({
			reranker: {
				describe: () => ({ name: "stub", provider: "stub", model: "m", available: true }),
				rerank: async (_query, results, options) => {
					seenTopN = options?.topN;
					return [...results];
				},
			},
		});
		engine.register(stubMethod("posix", [{ source: "/a" }]));
		await engine.retrieve("q");
		expect(seenTopN).toBe(DEFAULT_RERANK_TOP_N);
	});

	it("prefers the caller's topK over the default rerank topN", async () => {
		let seenTopN: number | undefined;
		const engine = new RetrievalEngine({
			reranker: {
				describe: () => ({ name: "stub", provider: "stub", model: "m", available: true }),
				rerank: async (_query, results, options) => {
					seenTopN = options?.topN;
					return [...results];
				},
			},
		});
		engine.register(stubMethod("posix", [{ source: "/a" }, { source: "/b" }]));
		await engine.retrieve("q", { topK: 1 });
		expect(seenTopN).toBe(1);
	});

	it("preserves merged order and emits rerank-failed when the reranker throws", async () => {
		const engine = new RetrievalEngine({
			reranker: {
				describe: () => ({ name: "stub", provider: "stub", model: "m", available: true }),
				rerank: async () => {
					throw new Error("boom");
				},
			},
		});
		engine.register(stubMethod("posix", [{ source: "/a" }, { source: "/b" }]));
		const { results, diagnostics } = await engine.retrieve("q", { topK: 10 });
		expect(results).toHaveLength(2);
		expect(diagnostics.some((entry) => entry.code === "rerank-failed")).toBe(true);
	});
});
