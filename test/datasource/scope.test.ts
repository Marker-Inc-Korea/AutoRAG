import { describe, expect, it } from "vitest";
import {
	buildDatasourceChunkSource,
	buildDatasourceInstanceSource,
	DATASOURCE_CHUNKS_SEGMENT,
	datasourceSourceHasFragment,
	filterDatasourceScope,
	isDatasourceSource,
	matchesDatasourceScope,
} from "../../src/datasource/scope.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalResult } from "../../src/retrieval/types.ts";

describe("datasource scope helpers", () => {
	describe("buildDatasourceInstanceSource", () => {
		it("builds a slash-hierarchical instance root", () => {
			expect(buildDatasourceInstanceSource("kakao", "acct-1")).toBe("/kakao/acct-1");
		});

		it("normalizes redundant slashes and trailing slashes", () => {
			expect(buildDatasourceInstanceSource("kakao", "acct-1/")).toBe("/kakao/acct-1");
			expect(buildDatasourceInstanceSource("kakao/", "/acct-1")).toBe("/kakao/acct-1");
		});

		it("does not produce a '#' fragment", () => {
			const source = buildDatasourceInstanceSource("kakao", "acct-1");
			expect(source.includes("#")).toBe(false);
		});
	});

	describe("buildDatasourceChunkSource", () => {
		it("builds a slash-hierarchical chunk source", () => {
			expect(buildDatasourceChunkSource("kakao", "acct-1", "c-42")).toBe(
				`/kakao/acct-1/${DATASOURCE_CHUNKS_SEGMENT}/c-42`,
			);
		});

		it("places chunks under the instance root with the chunks segment", () => {
			const source = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(source.startsWith("/kakao/acct-1/")).toBe(true);
			expect(source.includes(`/${DATASOURCE_CHUNKS_SEGMENT}/`)).toBe(true);
		});

		it("never emits a '#' fragment separator (slash-hierarchical only)", () => {
			const source = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(source).toBe("/kakao/acct-1/chunks/c-42");
			expect(source.includes("#")).toBe(false);
		});
	});

	describe("datasourceSourceHasFragment / isDatasourceSource", () => {
		it("detects a '#' fragment", () => {
			expect(datasourceSourceHasFragment("/kakao/acct-1/chunks/c-1#meta")).toBe(true);
			expect(datasourceSourceHasFragment("/kakao/acct-1")).toBe(false);
		});

		it("isDatasourceSource accepts clean slash paths", () => {
			expect(isDatasourceSource("/kakao/acct-1")).toBe(true);
			expect(isDatasourceSource("/kakao/acct-1/chunks/c-1")).toBe(true);
		});

		it("isDatasourceSource rejects root, empty, and fragment paths", () => {
			expect(isDatasourceSource("/")).toBe(false);
			expect(isDatasourceSource("")).toBe(false);
			expect(isDatasourceSource("/kakao/acct-1#frag")).toBe(false);
		});
	});

	describe("matchesDatasourceScope", () => {
		it("matches a chunk source against its instance scope", () => {
			const chunk = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(matchesDatasourceScope(chunk, "/kakao/acct-1")).toBe(true);
		});

		it("matches a chunk source against a chunks glob scope", () => {
			const chunk = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(matchesDatasourceScope(chunk, "/kakao/acct-1/chunks/*")).toBe(true);
		});

		it("does not match a chunk source against a different instance scope", () => {
			const chunk = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(matchesDatasourceScope(chunk, "/kakao/acct-2")).toBe(false);
		});

		it("treats undefined scope as a wildcard (match everything valid)", () => {
			const chunk = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(matchesDatasourceScope(chunk, undefined)).toBe(true);
		});

		it("rejects sources containing a '#' fragment even when scope would match", () => {
			expect(matchesDatasourceScope("/kakao/acct-1/chunks/c-1#meta", "/kakao/acct-1")).toBe(false);
		});

		it("chunk source is slash-hierarchical: no '#' separator between segments", () => {
			const chunk = buildDatasourceChunkSource("kakao", "acct-1", "c-42");
			expect(chunk).toBe("/kakao/acct-1/chunks/c-42");
			expect(chunk.includes("#")).toBe(false);
		});
	});
});

const result = (source: string): RetrievalResult => ({
	id: source,
	source,
	content: `content@${source}`,
	score: 1,
	metadata: {},
});

const scopedMethod = (name: string, datasourceId: string): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: "scoped datasource method",
		status: "active",
		capabilities: ["scoped"],
		datasourceId,
		tags: ["ds"],
	}),
	retrieve: async () => [],
});

const unscopedMethod = (name: string, datasourceId: string): RetrievalMethod => ({
	describe: (): RetrievalMethodDescriptor => ({
		name,
		type: "vector",
		description: "datasource method without source scopes",
		status: "active",
		capabilities: [],
		datasourceId,
		tags: ["ds"],
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

describe("filterDatasourceScope", () => {
	it("narrows scoped datasource results by the query scope", () => {
		const method = scopedMethod("kakao", "kakao");
		const byMethod = new Map<string, RetrievalResult[]>([
			["kakao", [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-2/chunks/c-7")]],
		]);
		const out = filterDatasourceScope(byMethod, [method], "/kakao/acct-1");
		expect(out.get("kakao")?.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/c-1"]);
	});

	it("keeps every valid source when scope is undefined", () => {
		const method = scopedMethod("kakao", "kakao");
		const byMethod = new Map<string, RetrievalResult[]>([
			["kakao", [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-2/chunks/c-7")]],
		]);
		const out = filterDatasourceScope(byMethod, [method]);
		expect(out.get("kakao")?.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/c-1", "/kakao/acct-2/chunks/c-7"]);
	});

	it("rejects sources containing a '#' fragment regardless of scope", () => {
		const method = scopedMethod("kakao", "kakao");
		const byMethod = new Map<string, RetrievalResult[]>([
			["kakao", [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-1/chunks/c-2#meta")]],
		]);
		const out = filterDatasourceScope(byMethod, [method]);
		expect(out.get("kakao")?.map((r) => r.source)).toEqual(["/kakao/acct-1/chunks/c-1"]);
	});

	it("leaves plain (non-datasource) method results untouched", () => {
		const method = plainMethod("posix");
		const original = [result("/docs/a.txt"), result("/docs/b.md")];
		const byMethod = new Map<string, RetrievalResult[]>([["posix", original]]);
		const out = filterDatasourceScope(byMethod, [method], "/docs/other");
		expect(out.get("posix")).toBe(original);
	});

	it("leaves datasource methods without the scoped capability untouched", () => {
		const method = unscopedMethod("kakao", "kakao");
		const original = [result("/kakao/personal/chunks/c-1")];
		const byMethod = new Map<string, RetrievalResult[]>([["kakao", original]]);
		const out = filterDatasourceScope(byMethod, [method], "/kakao/other/**");
		expect(out.get("kakao")).toBe(original);
	});

	it("passes through entries with no matching method descriptor", () => {
		const byMethod = new Map<string, RetrievalResult[]>([["mystery", [result("/x/y")]]]);
		const out = filterDatasourceScope(byMethod, [], "/scope");
		expect(out.get("mystery")).toBe(byMethod.get("mystery"));
	});

	it("does not mutate the input map or its result arrays", () => {
		const method = scopedMethod("kakao", "kakao");
		const originalResults = [result("/kakao/acct-1/chunks/c-1"), result("/kakao/acct-9/chunks/c-9")];
		const byMethod = new Map<string, RetrievalResult[]>([["kakao", originalResults]]);
		const out = filterDatasourceScope(byMethod, [method], "/kakao/acct-1");
		expect(out).not.toBe(byMethod);
		expect(out.get("kakao")).not.toBe(originalResults);
		expect(originalResults).toHaveLength(2);
	});
});
