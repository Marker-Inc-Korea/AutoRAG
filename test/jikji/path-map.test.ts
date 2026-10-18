import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { outermostJikjiRoots } from "../../src/jikji/index.ts";

const at = (...parts: string[]): string => resolve("/data", ...parts);

describe("outermostJikjiRoots", () => {
	it("keeps the first occurrence of a duplicated root", () => {
		expect(outermostJikjiRoots([at("docs"), at("notes"), at("docs")])).toEqual([at("docs"), at("notes")]);
	});

	it("drops a root nested inside another root, whichever order they are listed in", () => {
		expect(outermostJikjiRoots([at("docs"), at("docs", "work")])).toEqual([at("docs")]);
		expect(outermostJikjiRoots([at("docs", "work", "deep"), at("docs")])).toEqual([at("docs")]);
	});

	it("keeps sibling roots that merely share a name prefix", () => {
		expect(outermostJikjiRoots([at("docs"), at("docs-2")])).toEqual([at("docs"), at("docs-2")]);
	});

	it("treats a '..'-prefixed directory name as nested content, not as an escape", () => {
		expect(outermostJikjiRoots([at("docs"), at("docs", "..cache")])).toEqual([at("docs")]);
	});

	it("keeps the caller's order for the surviving roots", () => {
		expect(outermostJikjiRoots([at("zeta"), at("alpha", "x"), at("alpha"), at("mid")])).toEqual([
			at("zeta"),
			at("alpha"),
			at("mid"),
		]);
	});

	it("returns nothing for no roots", () => {
		expect(outermostJikjiRoots([])).toEqual([]);
	});
});
