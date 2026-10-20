import { describe, expect, it } from "vitest";
import { normalizeVirtualPath } from "../../src/filesystem/source-paths.ts";

describe("normalizeVirtualPath", () => {
	it("collapses duplicate separators in virtual and datasource ids", () => {
		expect(normalizeVirtualPath("/kakao//default//chunks/chunk-1")).toBe("/kakao/default/chunks/chunk-1");
	});

	it("keeps file names that contain '#' or '?' as canonical virtual ids", () => {
		// Real file names are data, not URL syntax: a receipt named
		// `... #2832-1476.txt` must keep its canonical id so parsed mirrors stay
		// indexable end to end (issue #1617).
		expect(normalizeVirtualPath("/docs/receipt #2832-1476.txt")).toBe("/docs/receipt #2832-1476.txt");
		expect(normalizeVirtualPath("/docs/what ? query.txt")).toBe("/docs/what ? query.txt");
		// URL-shaped sources are still rejected: the ban targeted syntax, not names.
		expect(normalizeVirtualPath("file:///docs/a.md")).toBeUndefined();
	});

	it("fails closed for traversal, scheme, backslash, and empty sources", () => {
		expect(normalizeVirtualPath("/docs/../secret.md")).toBeUndefined();
		expect(normalizeVirtualPath("kakao:chat/sender/chunk-1")).toBeUndefined();
		expect(normalizeVirtualPath("file:///docs/a.md")).toBeUndefined();
		expect(normalizeVirtualPath("docs\\a.md")).toBeUndefined();
		expect(normalizeVirtualPath("")).toBeUndefined();
	});
});
