import { describe, expect, it } from "vitest";
import { DatasourceAccessContext } from "../../src/datasource/access-context.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { RetrievalSelectionError, resolveSelectedMethods } from "../../src/retrieval/selection.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

// --- Helpers ---

const localMethod = (
	name: string,
	results: readonly (Partial<RetrievalResult> & { source: string })[],
): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "posix",
		description: `local ${name}`,
		status: "active",
		capabilities: [],
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

const spyDatasourceMethod = (
	name: string,
	datasourceId: string,
	tags: readonly string[],
	results: readonly (Partial<RetrievalResult> & { source: string })[],
	calls: { count: number },
	capabilities: readonly string[] = ["scoped"],
): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: `datasource ${name}`,
		status: "active",
		capabilities: [...capabilities],
		tags: [...tags],
		datasourceId,
	}),
	retrieve: async (): Promise<RetrievalResult[]> => {
		calls.count += 1;
		return results.map((r, index) => ({
			id: r.id ?? `${name}:${index}`,
			content: r.content ?? `content-${name}-${index}`,
			score: r.score ?? 1,
			metadata: r.metadata ?? {},
			source: r.source,
		}));
	},
});

// --- Tests ---

describe("RetrievalEngine.retrieveSelected", () => {
	it("runs only the selected datasource and never invokes the others", async () => {
		const kakao = { count: 0 };
		const slack = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao", "slack"] } });
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], kakao));
		engine.register(spyDatasourceMethod("slack.keyword", "slack", ["slack"], [{ source: "/slack/s" }], slack));

		const { results } = await engine.retrieveSelected("q", { datasourceIds: ["kakao"] });

		expect(results.map((r) => r.source)).toEqual(["/kakao/a"]);
		expect(kakao.count).toBe(1);
		expect(slack.count).toBe(0);
	});

	it("never invokes an unauthorized datasource, even when nothing was selected", async () => {
		const denied = { count: 0 };
		const local = { count: 0 };
		const localSpy: RetrievalMethod = {
			describe: (): RetrievalMethodDescriptor => ({
				name: "posix",
				type: "posix",
				description: "local posix",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => {
				local.count += 1;
				return [{ id: "l1", source: "/repo/file", content: "c", score: 1, metadata: {} }];
			},
		};
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: [] } });
		engine.register(localSpy);
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], denied));

		const { results } = await engine.retrieveSelected("q", {});

		expect(results.map((r) => r.source)).toEqual(["/repo/file"]);
		expect(denied.count).toBe(0);
		expect(local.count).toBe(1);
	});

	it("excludes local methods for explicit datasourceIds unless local is true", async () => {
		const localRuns = { count: 0 };
		const kakao = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao"] } });
		const posix: RetrievalMethod = {
			describe: (): RetrievalMethodDescriptor => ({
				name: "posix",
				type: "posix",
				description: "local posix",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => {
				localRuns.count += 1;
				return [{ id: "l1", source: "/repo/file", content: "c", score: 1, metadata: {} }];
			},
		};
		engine.register(posix);
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], kakao));

		const withoutLocal = await engine.retrieveSelected("q", { datasourceIds: ["kakao"] });
		expect(withoutLocal.results.map((r) => r.source)).toEqual(["/kakao/a"]);
		expect(localRuns.count).toBe(0);

		const withLocal = await engine.retrieveSelected("q", { datasourceIds: ["kakao"], local: true });
		expect(new Set(withLocal.results.map((r) => r.source))).toEqual(new Set(["/kakao/a", "/repo/file"]));
		expect(localRuns.count).toBe(1);
	});

	it("excludes local methods when local is false", async () => {
		const localRuns = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao"] } });
		const posix: RetrievalMethod = {
			describe: (): RetrievalMethodDescriptor => ({
				name: "posix",
				type: "posix",
				description: "local posix",
				status: "active",
				capabilities: [],
			}),
			retrieve: async () => {
				localRuns.count += 1;
				return [{ id: "l1", source: "/repo/file", content: "c", score: 1, metadata: {} }];
			},
		};
		engine.register(posix);
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], { count: 0 }));

		const { results } = await engine.retrieveSelected("q", { local: false });

		expect(results.map((r) => r.source)).toEqual(["/kakao/a"]);
		expect(localRuns.count).toBe(0);
	});

	it("intersects an explicit methods selection with the eligible set", async () => {
		const kakao = { count: 0 };
		const slack = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao", "slack"] } });
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], kakao));
		engine.register(
			spyDatasourceMethod("kakao.semantic", "kakao", ["kakao"], [{ source: "/kakao/b" }], { count: 0 }),
		);
		engine.register(spyDatasourceMethod("slack.keyword", "slack", ["slack"], [{ source: "/slack/s" }], slack));

		const { results } = await engine.retrieveSelected("q", { methods: ["kakao.keyword"] });

		expect(results.map((r) => r.source)).toEqual(["/kakao/a"]);
		expect(kakao.count).toBe(1);
		expect(slack.count).toBe(0);
	});

	it("rejects an unknown method selection", async () => {
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: ["kakao"] } });
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], { count: 0 }));

		await expect(engine.retrieveSelected("q", { methods: ["nope"] })).rejects.toMatchObject({
			code: "unknown-method",
		});
	});

	it("rejects an unauthorized method selection without executing it", async () => {
		const denied = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: [] } });
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], denied));

		await expect(engine.retrieveSelected("q", { methods: ["kakao.keyword"] })).rejects.toMatchObject({
			code: "unauthorized-method",
		});
		expect(denied.count).toBe(0);
	});

	it("rejects an unauthorized and an unknown datasource selection", async () => {
		const denied = { count: 0 };
		const engine = new RetrievalEngine({ datasourceAccess: { allowedTags: [] } });
		engine.register(spyDatasourceMethod("kakao.keyword", "kakao", ["kakao"], [{ source: "/kakao/a" }], denied));

		await expect(engine.retrieveSelected("q", { datasourceIds: ["kakao"] })).rejects.toMatchObject({
			code: "unauthorized-datasource",
		});
		await expect(engine.retrieveSelected("q", { datasourceIds: ["ghost"] })).rejects.toMatchObject({
			code: "unknown-datasource",
		});
		expect(denied.count).toBe(0);
	});

	it("preserves trusted scope narrowing through the existing filter", async () => {
		const engine = new RetrievalEngine({
			datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack/allowed/**"] },
		});
		engine.register(
			spyDatasourceMethod(
				"slack.keyword",
				"slack",
				["slack"],
				[{ source: "/slack/allowed/channel/m1" }, { source: "/slack/secret/channel/m2" }],
				{ count: 0 },
			),
		);

		const { results } = await engine.retrieveSelected("q", { datasourceIds: ["slack"] });

		expect(results.map((r) => r.source)).toEqual(["/slack/allowed/channel/m1"]);
	});

	it("selects a method-less authorized datasource cleanly when the catalog supplies it", async () => {
		const engine = new RetrievalEngine({
			datasourceAccess: { allowedTags: ["kakao"] },
			authorizedDatasourceIds: () => ["kakao"],
		});
		engine.register(localMethod("posix", [{ source: "/repo/file" }]));

		const { results } = await engine.retrieveSelected("q", { datasourceIds: ["kakao"] });
		expect(results).toEqual([]);
	});
});

describe("resolveSelectedMethods", () => {
	const descriptor = (name: string, datasourceId?: string, tags?: readonly string[]): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: name,
		status: "active",
		capabilities: [],
		...(datasourceId !== undefined ? { datasourceId } : {}),
		...(tags !== undefined ? { tags } : {}),
	});
	const method = (d: RetrievalMethodDescriptor): RetrievalMethod => ({
		describe: () => d,
		retrieve: async () => [],
	});

	it("defaults to every authorized datasource plus local", () => {
		const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
		const methods = [method(descriptor("posix")), method(descriptor("kakao.keyword", "kakao", ["kakao"]))];

		const selected = resolveSelectedMethods(methods, ctx, {}, ["kakao"]);
		expect(selected.map((m) => m.describe().name)).toEqual(["posix", "kakao.keyword"]);
	});

	it("throws a typed error for an unknown method", () => {
		const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
		expect(() => resolveSelectedMethods([], ctx, { methods: ["nope"] }, [])).toThrow(RetrievalSelectionError);
	});
});
