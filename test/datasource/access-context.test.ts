import { describe, expect, it } from "vitest";
import { DatasourceAccessContext } from "../../src/datasource/access-context.ts";
import type { DatasourceAccessible } from "../../src/datasource/types.ts";

const datasourceDescriptor = (overrides: Partial<DatasourceAccessible> = {}): DatasourceAccessible => ({
	datasourceId: "kakao:acct-1",
	tags: ["kakao"],
	...overrides,
});

const nonDatasourceDescriptor = (): DatasourceAccessible => ({
	// No datasourceId → non-datasource, pass-through.
	tags: [],
});

describe("DatasourceAccessContext", () => {
	describe("default-deny", () => {
		it("is deny-all when allowedTags is undefined", () => {
			const ctx = new DatasourceAccessContext();
			expect(ctx.isDenyAll).toBe(true);
			expect(ctx.allowedTags).toEqual([]);
		});

		it("is deny-all when allowedTags is an empty array", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: [] });
			expect(ctx.isDenyAll).toBe(true);
		});

		it("denies a datasource descriptor when deny-all", () => {
			const ctx = new DatasourceAccessContext();
			expect(ctx.isAccessible(datasourceDescriptor())).toBe(false);
		});

		it("predicate returns explicit false for every source when deny-all", () => {
			const ctx = new DatasourceAccessContext();
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(false);
			expect(predicate("/anything")).toBe(false);
			// Explicit false, never undefined-as-deny.
			expect(predicate("/kakao/acct-1")).toBe(false);
		});
	});

	describe("isAccessible", () => {
		it("allows a datasource descriptor whose tags intersect allowedTags", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isAccessible(datasourceDescriptor({ tags: ["kakao"] }))).toBe(true);
		});

		it("denies a datasource descriptor with no intersecting tag", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isAccessible(datasourceDescriptor({ tags: ["slack"] }))).toBe(false);
		});

		it("denies a datasource descriptor with no tags at all", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isAccessible(datasourceDescriptor({ tags: [] }))).toBe(false);
		});

		it("allows when multiple tags and only one intersects", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao", "chat"] });
			expect(ctx.isAccessible(datasourceDescriptor({ tags: ["chat", "experimental"] }))).toBe(true);
		});

		it("passes through non-datasource descriptors (no datasourceId) even when deny-all", () => {
			const ctx = new DatasourceAccessContext();
			expect(ctx.isAccessible(nonDatasourceDescriptor())).toBe(true);
		});

		it("passes through non-datasource descriptors when allowedTags are set", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isAccessible(nonDatasourceDescriptor())).toBe(true);
		});
	});

	describe("allowedSourcesPredicate — multi-scope source matching", () => {
		const ctx = new DatasourceAccessContext({
			allowedTags: ["kakao"],
			allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
		});

		it("matches a source under the first trusted scope", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(true);
		});

		it("matches a source under the second trusted scope", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-2/chunks/c-9")).toBe(true);
		});

		it("matches the instance root itself", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1")).toBe(true);
		});

		it("denies a source under neither trusted scope", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-3/chunks/c-1")).toBe(false);
		});

		it("denies a source under a different skill entirely", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/slack/workspace-1/channels/general")).toBe(false);
		});

		it("denies sources containing a '#' fragment", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1/chunks/c-1#meta")).toBe(false);
		});
	});

	describe("allowedSourcesPredicate — userScope intersection", () => {
		const ctx = new DatasourceAccessContext({
			allowedTags: ["kakao"],
			allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
		});

		it("allows when source is in trusted scope AND userScope", () => {
			const predicate = ctx.allowedSourcesPredicate("/kakao/acct-1");
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(true);
		});

		it("denies when source is in trusted scope but outside userScope", () => {
			const predicate = ctx.allowedSourcesPredicate("/kakao/acct-1");
			expect(predicate("/kakao/acct-2/chunks/c-1")).toBe(false);
		});

		it("denies when source is outside trusted scope but inside userScope", () => {
			const predicate = ctx.allowedSourcesPredicate("/kakao/acct-3");
			expect(predicate("/kakao/acct-3/chunks/c-1")).toBe(false);
		});

		it("treats undefined userScope as no extra restriction", () => {
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-2/chunks/c-7")).toBe(true);
		});

		it("supports glob userScope intersecting a trusted scope", () => {
			const predicate = ctx.allowedSourcesPredicate("/kakao/acct-1/chunks/*");
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(true);
			expect(predicate("/kakao/acct-1/chunks/c-2")).toBe(true);
			expect(predicate("/kakao/acct-1/index")).toBe(false);
		});
	});

	describe("allowedSourcesPredicate — empty trusted scopes with allow-tags set", () => {
		it("denies every source explicitly (not deny-all, but no trusted scope)", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isDenyAll).toBe(false);
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(false);
		});
	});
});
