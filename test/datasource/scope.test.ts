import { describe, expect, it } from "vitest";
import {
	buildDatasourceChunkSource,
	buildDatasourceInstanceSource,
	DATASOURCE_CHUNKS_SEGMENT,
	datasourceSourceHasFragment,
	isDatasourceSource,
	matchesDatasourceScope,
} from "../../src/datasource/scope.ts";

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
