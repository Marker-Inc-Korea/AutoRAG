import { describe, expect, it } from "vitest";
import { isHiddenName, visibleEntries } from "../src/renderer/src/state/visibility";

describe("isHiddenName", () => {
	it("treats dot-prefixed names as hidden", () => {
		expect(isHiddenName(".git")).toBe(true);
		expect(isHiddenName(".env.local")).toBe(true);
	});

	it("treats regular names as visible", () => {
		expect(isHiddenName("docs")).toBe(false);
		expect(isHiddenName("회의록_0922.md")).toBe(false);
	});
});

describe("visibleEntries", () => {
	const rows = [
		{ name: "Finance" },
		{ name: ".hidden" },
		{ name: "예산.xlsx" },
		{ name: ".config" },
	] as const;

	it("drops dot-prefixed entries when hidden files are off (default)", () => {
		expect(visibleEntries(rows, false).map((entry) => entry.name)).toEqual(["Finance", "예산.xlsx"]);
	});

	it("keeps every entry in order when hidden files are on", () => {
		expect(visibleEntries(rows, true).map((entry) => entry.name)).toEqual([
			"Finance",
			".hidden",
			"예산.xlsx",
			".config",
		]);
	});
});
