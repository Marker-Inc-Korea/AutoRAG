import { describe, expect, it } from "vitest";
import { DatasourceAccessContext } from "../../src/datasource/access-context.ts";
import { DatasourceResultFilter } from "../../src/datasource/result-filter.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

const result = (source: string, score = 1): RetrievalResult => ({
	id: source,
	source,
	content: `content@${source}`,
	score,
	metadata: {},
});

const dsMethod = (name: string, datasourceId: string, tags: readonly string[]): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: "datasource method",
		status: "active",
		capabilities: ["chat"],
		datasourceId,
		tags,
	}),
	retrieve: async () => [],
});

const plainMethod = (name: string): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "posix",
		description: "plain retrieval method",
		status: "active",
		capabilities: [],
	}),
	retrieve: async () => [],
});

const filter = new DatasourceResultFilter();

describe("DatasourceResultFilter", () => {
	describe("single-scope source filtering", () => {
		it("keeps only sources under the trusted scope for an allowed datasource method", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([
				[
					"kakao",
					[
						result("/kakao/acct-1/chunks/c-1"),
						result("/kakao/acct-1/chunks/c-2"),
						result("/kakao/acct-3/chunks/c-9"),
					],
				],
			]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.get("kakao")?.map((r) => r.source)).toEqual([
				"/kakao/acct-1/chunks/c-1",
				"/kakao/acct-1/chunks/c-2",
			]);
		});
	});

	describe("multi-scope source filtering", () => {
		it("keeps sources under any of the trusted scopes", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([
				[
					"kakao",
					[
						result("/kakao/acct-1/chunks/c-1"),
						result("/kakao/acct-2/chunks/c-7"),
						result("/kakao/acct-3/chunks/c-x"),
					],
				],
			]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.get("kakao")?.map((r) => r.source)).toEqual([
				"/kakao/acct-1/chunks/c-1",
				"/kakao/acct-2/chunks/c-7",
			]);
		});
	});

	describe("userScope intersection", () => {
		it("narrows trusted-scope results to the user scope", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([
				["kakao", [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-2/chunks/c-7")]],
			]);
			const out = filter.filter(byMethod, [method], ctx, "/kakao/acct-1");
			expect(out.get("kakao")?.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/c-1"]);
		});

		it("keeps all trusted-scope results when userScope is undefined", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([
				["kakao", [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-2/chunks/c-7")]],
			]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.get("kakao")?.map((r) => r.source)).toEqual([
				"/kakao/acct-1/chunks/c-1",
				"/kakao/acct-2/chunks/c-7",
			]);
		});
	});

	describe("deny-all context", () => {
		it("drops all datasource results to an explicit empty array (never undefined)", () => {
			const ctx = new DatasourceAccessContext();
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([["kakao", [result("/kakao/acct-1/chunks/c-1")]]]);
			const out = filter.filter(byMethod, [method], ctx);
			const entry = out.get("kakao");
			expect(Array.isArray(entry)).toBe(true);
			expect(entry).toEqual([]);
		});

		it("drops all datasource results when tags do not intersect allowedTags", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["slack"] });
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([["kakao", [result("/kakao/acct-1/chunks/c-1")]]]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.get("kakao")).toEqual([]);
		});
	});

	describe("empty allowed scopes (tags set, no scopes)", () => {
		it("denies every source explicitly while keeping the method entry present", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>([["kakao", [result("/kakao/acct-1/chunks/c-1")]]]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.has("kakao")).toBe(true);
			expect(out.get("kakao")).toEqual([]);
		});
	});

	describe("non-datasource passthrough", () => {
		it("leaves plain (non-datasource) method results untouched", () => {
			const ctx = new DatasourceAccessContext(); // deny-all for datasources
			const method = plainMethod("posix");
			const original = [result("/docs/a.txt"), result("/docs/b.md")];
			const byMethod = new Map<string, RetrievalResult[]>([["posix", original]]);
			const out = filter.filter(byMethod, [method], ctx, "/docs");
			expect(out.get("posix")).toBe(original);
		});

		it("does not pass undefined to matchesVirtualPathScope as deny for non-datasource results", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			const method = plainMethod("posix");
			const original = [result("/docs/a.txt")];
			const byMethod = new Map<string, RetrievalResult[]>([["posix", original]]);
			// userScope undefined must not be treated as a deny for passthrough.
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.get("posix")).toBe(original);
		});
	});

	describe("input immutability", () => {
		it("does not mutate the input map or its result arrays", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const originalResults = [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-3/chunks/c-9")];
			const byMethod = new Map<string, RetrievalResult[]>([["kakao", originalResults]]);
			const out = filter.filter(byMethod, [method], ctx);
			expect(out).not.toBe(byMethod);
			expect(out.get("kakao")).not.toBe(originalResults);
			// Input array unchanged.
			expect(originalResults).toHaveLength(2);
		});
	});

	describe("methods without a matching result entry", () => {
		it("produces no entry for a datasource method absent from byMethod", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1"],
			});
			const method = dsMethod("kakao", "kakao:acct-1", ["kakao"]);
			const byMethod = new Map<string, RetrievalResult[]>();
			const out = filter.filter(byMethod, [method], ctx);
			expect(out.has("kakao")).toBe(false);
		});
	});
});
