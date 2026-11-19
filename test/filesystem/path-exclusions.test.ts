import { describe, expect, it } from "vitest";
import { createPathExclusionMatcher, isGlobExclusion } from "../../src/filesystem/path-exclusions.ts";

describe("isGlobExclusion", () => {
	it.each([
		["**/node_modules/**", true],
		["/Users/me/*.log", true],
		["/Users/me/file?.txt", true],
		["/Users/me/[ab].txt", true],
		["/Users/me/Documents/private", false],
		["relative/dir", false],
	])("%s -> %s", (entry, expected) => {
		expect(isGlobExclusion(entry)).toBe(expected);
	});
});

describe("createPathExclusionMatcher", () => {
	it("matches an excluded directory and everything under it, not a sibling that shares its prefix", () => {
		const excluded = createPathExclusionMatcher(["/data/private"], "linux");
		expect(excluded("/data/private")).toBe(true);
		expect(excluded("/data/private/a/b.txt")).toBe(true);
		expect(excluded("/data/private-notes/a.txt")).toBe(false);
	});

	it("matches a directory glob at any depth, including directly under the root", () => {
		const excluded = createPathExclusionMatcher(["**/node_modules/**"], "linux");
		expect(excluded("/home/me/app/node_modules/lodash/index.js")).toBe(true);
		expect(excluded("/node_modules/x.js")).toBe(true);
		expect(excluded("/home/me/app/src/index.js")).toBe(false);
	});

	it("does not match a directory whose name only contains the glob's name", () => {
		const excluded = createPathExclusionMatcher(["**/node_modules/**"], "linux");
		expect(excluded("/home/me/my_node_modules_notes/n.txt")).toBe(false);
		expect(excluded("/home/me/node_modules_backup/n.txt")).toBe(false);
	});

	it("matches a file glob by extension", () => {
		const excluded = createPathExclusionMatcher(["**/*.log"], "linux");
		expect(excluded("/var/app/a.log")).toBe(true);
		expect(excluded("/a.log")).toBe(true);
		expect(excluded("/var/app/a.logs")).toBe(false);
		expect(excluded("/var/app/a.txt")).toBe(false);
	});

	it("matches a glob anchored to a folder, such as everything under Downloads", () => {
		const excluded = createPathExclusionMatcher(["/Users/me/Downloads/**"], "linux");
		expect(excluded("/Users/me/Downloads/IMG_1.png")).toBe(true);
		expect(excluded("/Users/me/Downloads/sub/IMG_1.png")).toBe(true);
		expect(excluded("/Users/me/Documents/IMG_1.png")).toBe(false);
	});

	it("handles path entries and glob entries together", () => {
		const excluded = createPathExclusionMatcher(["/data/private", "**/*.tmp"], "linux");
		expect(excluded("/data/private/a.txt")).toBe(true);
		expect(excluded("/data/x/y.tmp")).toBe(true);
		expect(excluded("/data/x/y.txt")).toBe(false);
	});

	it("compares case-insensitively on Windows for paths and globs", () => {
		const excluded = createPathExclusionMatcher(["C:\\Data\\Private", "**/Node_Modules/**"], "win32");
		expect(excluded("c:\\data\\private\\a.txt")).toBe(true);
		expect(excluded("C:\\Proj\\NODE_MODULES\\x\\i.js")).toBe(true);
	});

	it("excludes nothing when given no entries", () => {
		expect(createPathExclusionMatcher([], "linux")("/anything")).toBe(false);
	});
});
