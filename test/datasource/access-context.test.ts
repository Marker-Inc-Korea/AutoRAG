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

	describe("allowedSourcesPredicate — tag-level gating only", () => {
		// Datasource result sources are datasource identities, not
		// slash-hierarchical virtual paths, so no per-source path gate exists.
		// Access is decided per skill via allow-tags (see isAccessible).

		it("allows every source when tags are granted", () => {
			const ctx = new DatasourceAccessContext({
				allowedTags: ["kakao"],
				allowedScopes: ["/kakao/acct-1", "/kakao/acct-2"],
			});
			const predicate = ctx.allowedSourcesPredicate();
			expect(predicate("/kakao/acct-1/chunks/c-1")).toBe(true);
			expect(predicate("/kakao/acct-2/chunks/c-9")).toBe(true);
			expect(predicate("kakao:오픈소스 개발과제/chunk_58b3")).toBe(true);
			expect(predicate("/slack/workspace-1/channels/general")).toBe(true);
		});

		it("userScope never restricts sources (no virtual-path gate)", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			const predicate = ctx.allowedSourcesPredicate("/kakao/acct-1");
			expect(predicate("/kakao/acct-2/chunks/c-1")).toBe(true);
			expect(predicate("kakao:other-room/chunk-9")).toBe(true);
		});

		it("empty trusted scopes with allow-tags set still allow sources", () => {
			const ctx = new DatasourceAccessContext({ allowedTags: ["kakao"] });
			expect(ctx.isDenyAll).toBe(false);
			expect(ctx.allowedSourcesPredicate()("/kakao/acct-1/chunks/c-1")).toBe(true);
		});
	});
});
