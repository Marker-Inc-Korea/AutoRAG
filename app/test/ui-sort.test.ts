import { describe, expect, it } from "vitest";
import { cycleSort, sortEntries } from "../src/renderer/src/state/sort";
import type { SortableEntry } from "../src/renderer/src/state/sort";

function e(
	name: string,
	kind: "folder" | "file",
	modifiedValue: number,
	sizeValue: number,
	kindLabel: string,
	location = "Documents",
): SortableEntry {
	return { name, kind, modifiedValue, sizeValue, kindLabel, location };
}

describe("cycleSort", () => {
	it("starts Name ascending", () => {
		expect(cycleSort(null, "name")).toEqual({ key: "name", dir: 1, flipped: false });
	});

	it("starts Date and Size descending", () => {
		expect(cycleSort(null, "date")).toEqual({ key: "date", dir: -1, flipped: false });
		expect(cycleSort(null, "size")).toEqual({ key: "size", dir: -1, flipped: false });
	});

	it("flips direction on the second click", () => {
		const first = cycleSort(null, "name");
		expect(cycleSort(first, "name")).toEqual({ key: "name", dir: -1, flipped: true });
	});

	it("clears sorting on the third click", () => {
		const second = cycleSort(cycleSort(null, "date"), "date");
		expect(cycleSort(second, "date")).toBeNull();
	});

	it("restarts the cycle when another column is clicked", () => {
		const flipped = cycleSort(cycleSort(null, "name"), "name");
		expect(cycleSort(flipped, "size")).toEqual({ key: "size", dir: -1, flipped: false });
	});
});

describe("sortEntries", () => {
	const rows = [
		e("나무.pdf", "file", 20, 200, "PDF Document"),
		e("Zebra", "folder", 10, 0, "Folder"),
		e("가방.xlsx", "file", 30, 100, "Excel Spreadsheet"),
		e("apple", "folder", 40, 0, "Folder"),
	];

	it("keeps the listing order when sorting is cleared", () => {
		expect(sortEntries(rows, null).map((r) => r.name)).toEqual(["나무.pdf", "Zebra", "가방.xlsx", "apple"]);
	});

	it("puts folders before files in both directions", () => {
		expect(sortEntries(rows, { key: "name", dir: 1, flipped: false }).map((r) => r.kind)).toEqual([
			"folder",
			"folder",
			"file",
			"file",
		]);
		expect(sortEntries(rows, { key: "name", dir: -1, flipped: true }).map((r) => r.kind)).toEqual([
			"folder",
			"folder",
			"file",
			"file",
		]);
	});

	it("compares names with the Korean locale", () => {
		expect(sortEntries(rows, { key: "name", dir: 1, flipped: false }).map((r) => r.name)).toEqual([
			"apple",
			"Zebra",
			"가방.xlsx",
			"나무.pdf",
		]);
	});

	it("sorts dates and sizes numerically", () => {
		expect(sortEntries(rows, { key: "date", dir: -1, flipped: false }).map((r) => r.name)).toEqual([
			"apple",
			"Zebra",
			"가방.xlsx",
			"나무.pdf",
		]);
		expect(sortEntries(rows, { key: "size", dir: 1, flipped: false }).map((r) => r.name)).toEqual([
			"Zebra",
			"apple",
			"가방.xlsx",
			"나무.pdf",
		]);
	});

	it("breaks Kind ties with the name", () => {
		const same = [
			e("b.pdf", "file", 1, 1, "PDF Document"),
			e("a.pdf", "file", 2, 2, "PDF Document"),
			e("c.xlsx", "file", 3, 3, "Excel Spreadsheet"),
		];
		expect(sortEntries(same, { key: "kind", dir: 1, flipped: false }).map((r) => r.name)).toEqual([
			"c.xlsx",
			"a.pdf",
			"b.pdf",
		]);
	});

	it("sorts search results by their location for the Where column", () => {
		const hits = [
			e("a.pdf", "file", 1, 1, "PDF Document", "Downloads"),
			e("b.pdf", "file", 2, 2, "PDF Document", "Desktop"),
		];
		expect(sortEntries(hits, { key: "where", dir: 1, flipped: false }).map((r) => r.location)).toEqual([
			"Desktop",
			"Downloads",
		]);
	});

	it("does not mutate the input", () => {
		const input = [...rows];
		sortEntries(input, { key: "name", dir: 1, flipped: false });
		expect(input.map((r) => r.name)).toEqual(["나무.pdf", "Zebra", "가방.xlsx", "apple"]);
	});
});
