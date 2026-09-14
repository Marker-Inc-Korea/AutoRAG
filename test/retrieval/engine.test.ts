import { describe, expect, it } from "vitest";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
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
			expect(engine.getAccessContext().isDenyAll).toBe(true);
		});

		it("accepts datasource access options", () => {
			const engine = new RetrievalEngine({
				datasourceAccess: { allowedTags: ["kakao"] },
			});
			expect(engine.getAccessContext().isDenyAll).toBe(false);
			expect(engine.getAccessContext().allowedTags).toEqual(["kakao"]);
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
			const engine = new RetrievalEngine({ defaultTopK: 50, datasourceAccess: { allowedTags: ["kakao"] } });
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.6 }]));
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results, diagnostics } = await engine.retrieve("test", {
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			expect(results).toHaveLength(2);
			expect(results.map((r) => r.source)).toContain("/a.txt");
			expect(results.map((r) => r.source)).toContain("/kakao/acct-1/chunks/c-1");
			expect(diagnostics).toEqual([]);
		});

		it("drops denied datasource results (default-deny)", async () => {
			const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["slack"] } });
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
			// datasource method denied → 0 results from it; posix method still passes.
			expect(results).toHaveLength(1);
			expect(results[0].source).toBe("/a.txt");
			expect(diagnostics).toEqual([]);
		});

		it("passthrough non-datasource methods when deny-all", async () => {
			const engine = new RetrievalEngine(); // deny-all
			engine.register(stubMethod("posix", [{ source: "/docs/a.txt", score: 0.7 }]));
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results, diagnostics } = await engine.retrieve("test");
			expect(results).toHaveLength(1);
			expect(results[0].source).toBe("/docs/a.txt");
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
			const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao"] } });
			engine.register(stubMethod("posix", [{ source: "/a.txt" }]));
			engine.register(
				stubDatasourceMethod("kakao", "kakao:acct-1", ["kakao"], [{ source: "/kakao/acct-1/chunks/c-1" }]),
			);
			const { byMethod, diagnostics } = await engine.retrieveByMethod("test", {
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			expect(byMethod.has("posix")).toBe(true);
			expect(byMethod.has("kakao")).toBe(true);
			expect(byMethod.get("posix")).toHaveLength(1);
			expect(byMethod.get("kakao")).toHaveLength(1);
			expect(diagnostics).toEqual([]);
		});

		it("applies datasource access gating per method", async () => {
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
			// No allowedTags → default-deny, datasource results should be empty.
			const { byMethod } = await engine.retrieveByMethod("test");
			expect(byMethod.get("kakao")).toEqual([]);
			expect(byMethod.get("posix")).toHaveLength(1);
		});
	});

	describe("scope/access behavior", () => {
		it("denies datasource method results for unmatched tag", async () => {
			const engine = new RetrievalEngine({
				datasourceAccess: { allowedTags: ["slack"] },
			});
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results } = await engine.retrieve("test");
			expect(results).toHaveLength(0);
		});

		it("denies datasource method results when tags have no intersection with engine base context", async () => {
			const engine = new RetrievalEngine({
				datasourceAccess: { allowedTags: ["slack"] },
			});
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results } = await engine.retrieve("test", { allowedTags: ["kakao"] });
			// Engine base says "slack" only, user passes "kakao" — intersection is empty → deny.
			expect(results).toHaveLength(0);
		});

		it("does not let caller tags grant access to a deny-all trusted context", async () => {
			const engine = new RetrievalEngine();
			engine.register(
				stubDatasourceMethod(
					"kakao",
					"kakao:acct-1",
					["kakao"],
					[{ source: "/kakao/acct-1/chunks/c-1", score: 0.9 }],
				),
			);
			const { results } = await engine.retrieve("test", {
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/**"],
			});
			expect(results).toEqual([]);
		});

		it("keeps trusted scopes restrictive when caller supplies a broader scope", async () => {
			const engine = new RetrievalEngine({
				datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack/allowed/**"] },
			});
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
			const { results } = await engine.retrieve("test", { allowedScopes: ["/slack/**"] });
			expect(results.map((result) => result.source)).toEqual(["/slack/allowed/channel/message"]);
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

	describe("MinSync binary-missing diagnostic parity", () => {
		it("emits minsync-unavailable when minsync method returns empty and binary is missing", async () => {
			const engine = new RetrievalEngine({
				isMinSyncBinaryMissing: () => true,
			});
			engine.register(
				stubMethod("minsync", []), // empty results, no throw (simulates missing binary)
			);
			engine.register(stubMethod("posix", [{ source: "/a.txt", score: 0.7 }]));
			const { results, diagnostics } = await engine.retrieve("test");
			// Other methods work unaffected.
			expect(results).toHaveLength(1);
			expect(results[0].source).toBe("/a.txt");
			// Diagnostic emitted.
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0]).toEqual({
				code: "minsync-unavailable",
				severity: "warning",
				message: "MinSync semantic search is unavailable; results rely on other retrieval paths.",
				source: "minsync",
			});
		});

		it("does not emit minsync-unavailable when minsync method returns results", async () => {
			const engine = new RetrievalEngine({
				isMinSyncBinaryMissing: () => true,
			});
			engine.register(stubMethod("minsync", [{ source: "/synced/doc.txt", score: 0.9 }]));
			const { diagnostics } = await engine.retrieve("test");
			expect(diagnostics).toEqual([]);
		});

		it("does not emit minsync-unavailable when hook is not provided", async () => {
			// Regression: without the hook, no diagnostic appears even when
			// a minsync method returns empty — the engine is not MinSync-aware.
			const engine = new RetrievalEngine(); // no isMinSyncBinaryMissing
			engine.register(
				stubMethod("minsync", []), // empty results, no throw
			);
			const { diagnostics } = await engine.retrieve("test");
			expect(diagnostics).toEqual([]);
		});

		it("does not duplicate diagnostic when ParallelRetriever already emitted minsync-unavailable", async () => {
			const engine = new RetrievalEngine({
				isMinSyncBinaryMissing: () => true,
			});
			// A throwing method named "minsync" triggers retrieval-method-failed first.
			// The merger maps a failing minsync throw to minsync-unavailable.
			engine.register({
				describe: () => ({
					name: "minsync",
					type: "posix" as const,
					description: "failing minsync",
					status: "stub" as const,
					capabilities: [],
				}),
				retrieve: async () => {
					throw new Error("minsync binary missing");
				},
			});
			const { diagnostics } = await engine.retrieve("test");
			// ParallelRetriever maps throw → retrieval-method-failed with source minsync.
			// Merger's methodFailureCode maps "minsync" to minsync-unavailable.
			// The post-check sees code minsync-unavailable already present and skips.
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0].code).toBe("minsync-unavailable");
		});
	});

	describe("getAccessContext", () => {
		it("returns the configured access context", () => {
			const engine = new RetrievalEngine({
				datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack"] },
			});
			const ctx = engine.getAccessContext();
			expect(ctx.allowedTags).toEqual(["slack"]);
			expect(ctx.allowedScopes).toEqual(["/slack"]);
		});

		it("is deny-all when no options provided", () => {
			const engine = new RetrievalEngine();
			expect(engine.getAccessContext().isDenyAll).toBe(true);
		});
	});
});
