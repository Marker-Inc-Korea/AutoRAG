import { describe, expect, it } from "vitest";
import type { FinderEntry } from "../src/renderer/src/data/entries";
import {
	applyVersionStacks,
	buildVersionFamilies,
	defaultIndexIncluded,
	EMPTY_VERSION_FAMILIES,
	relationLabel,
	type VersionFamilyData,
} from "../src/renderer/src/state/version-family";

function entry(path: string, name = path.split("/").pop() ?? ""): FinderEntry {
	return {
		name,
		path,
		kind: "file",
		fileKind: "xlsx",
		dateLabel: "Sep 13, 16:48",
		sizeLabel: "84 KB",
		kindLabel: "Excel Spreadsheet",
		location: path.slice(0, path.lastIndexOf("/")) || path,
		modifiedValue: 0,
		sizeValue: 0,
	};
}

const HEAD = "Documents/Finance/2026 Q3/Q3_마케팅예산_v3.xlsx";
const SAME_FOLDER_MEMBER = "Documents/Finance/2026 Q3/Q3_마케팅예산_v2.xlsx";
const CROSS_FOLDER_MEMBER = "Downloads/Q3_마케팅예산_v3 (1).xlsx";

function families(): VersionFamilyData[] {
	return [
		{
			head: HEAD,
			members: [
				{ path: SAME_FOLDER_MEMBER, relation: "near" },
				{ path: CROSS_FOLDER_MEMBER, relation: "exact" },
			],
			entriesByPath: {
				[HEAD]: entry(HEAD),
				[SAME_FOLDER_MEMBER]: entry(SAME_FOLDER_MEMBER),
				[CROSS_FOLDER_MEMBER]: entry(CROSS_FOLDER_MEMBER),
			},
		},
	];
}

describe("applyVersionStacks", () => {
	it("passes rows through when there are no families", () => {
		const rows = [entry(HEAD)];
		const out = applyVersionStacks(rows, {
			families: EMPTY_VERSION_FAMILIES,
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out).toEqual([{ entry: rows[0], stackCount: 0, stackOpen: false, child: false, relation: null }]);
	});

	it("hides same-folder members from the flat list", () => {
		const out = applyVersionStacks([entry(HEAD), entry(SAME_FOLDER_MEMBER)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out.map((row) => row.entry.path)).toEqual([HEAD]);
	});

	it("keeps cross-folder members as their own flat rows", () => {
		const out = applyVersionStacks([entry(CROSS_FOLDER_MEMBER)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out.map((row) => row.entry.path)).toEqual([CROSS_FOLDER_MEMBER]);
		expect(out[0]?.child).toBe(false);
	});

	it("counts members on the head and expands on selection", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [HEAD],
		});
		expect(out[0]?.stackCount).toBe(2);
		expect(out[0]?.stackOpen).toBe(true);
		expect(out.map((row) => row.entry.path)).toEqual([HEAD, SAME_FOLDER_MEMBER, CROSS_FOLDER_MEMBER]);
		expect(out[1]?.child).toBe(true);
		expect(out[1]?.relation).toBe("near");
		expect(out[2]?.relation).toBe("exact");
	});

	it("auto-expands when a member is selected", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [CROSS_FOLDER_MEMBER],
		});
		expect(out[0]?.stackOpen).toBe(true);
	});

	it("expands through the manual pin", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set([HEAD]),
			selectedKeys: [],
		});
		expect(out[0]?.stackOpen).toBe(true);
	});

	it("stays closed when nothing selects or pins it", () => {
		const out = applyVersionStacks([entry(HEAD)], {
			families: buildVersionFamilies(families()),
			manualOpen: new Set(),
			selectedKeys: [],
		});
		expect(out[0]?.stackOpen).toBe(false);
		expect(out.map((row) => row.entry.path)).toEqual([HEAD]);
	});
});

describe("relationLabel", () => {
	it("labels the relation alone in the same folder", () => {
		expect(relationLabel("exact", "Documents/Finance/2026 Q3", "Documents/Finance/2026 Q3")).toBe("동일본");
	});

	it("appends the folder when the member lives elsewhere", () => {
		expect(relationLabel("near", "Downloads", "Documents/Finance/2026 Q3")).toBe("유사본 · Downloads");
	});
});

describe("defaultIndexIncluded", () => {
	it("excludes stack members by default", () => {
		const built = buildVersionFamilies(families());
		expect(defaultIndexIncluded(SAME_FOLDER_MEMBER, built)).toBe(false);
		expect(defaultIndexIncluded(CROSS_FOLDER_MEMBER, built)).toBe(false);
		expect(defaultIndexIncluded(HEAD, built)).toBe(true);
		expect(defaultIndexIncluded("Documents/other.txt", built)).toBe(true);
	});
});
