import { describe, expect, it } from "vitest";
import {
	formatModified,
	formatSize,
	indexToast,
	parseModifiedLabel,
	searchSummaryText,
	statusBarText,
	trashToast,
} from "../src/renderer/src/state/format";

describe("formatModified", () => {
	it("renders the reference date format", () => {
		expect(formatModified("2026-09-13T16:48:00")).toBe("Sep 13, 16:48");
		expect(formatModified("2026-01-02T09:05:00")).toBe("Jan 2, 09:05");
	});

	it("falls back to an em dash for an unreadable timestamp", () => {
		expect(formatModified("not-a-date")).toBe("—");
	});

	it("round-trips into a sortable value", () => {
		expect(parseModifiedLabel("Sep 13, 16:48")).toBeGreaterThan(parseModifiedLabel("Sep 13, 09:05"));
		expect(parseModifiedLabel("Oct 1, 00:00")).toBeGreaterThan(parseModifiedLabel("Sep 30, 23:59"));
		expect(parseModifiedLabel("—")).toBe(0);
	});
});

describe("formatSize", () => {
	it("uses the reference units", () => {
		expect(formatSize(86016)).toBe("84 KB");
		expect(formatSize(5033164)).toBe("4.8 MB");
		expect(formatSize(1048576)).toBe("1.0 MB");
		expect(formatSize(6144)).toBe("6 KB");
		expect(formatSize(512)).toBe("512 B");
	});

	it("renders folders as an em dash", () => {
		expect(formatSize(null)).toBe("—");
	});
});

describe("status strings", () => {
	it("counts items and the selection", () => {
		expect(statusBarText(12, 0)).toBe("12 items");
		expect(statusBarText(12, 3)).toBe("12 items · 3 selected");
	});

	it("summarises a search", () => {
		expect(searchSummaryText(7)).toBe("7 results across all locations");
	});
});

describe("toasts", () => {
	it("names a single trashed item and counts many", () => {
		expect(trashToast(["Documents/Finance/a.pdf"])).toBe("a.pdf — 휴지통으로 이동했습니다");
		expect(trashToast(["a.pdf", "b.pdf", "c.pdf"])).toBe("3개 항목을 휴지통으로 이동했습니다");
	});

	it("reports index include and exclude", () => {
		expect(indexToast("a.pdf", false)).toBe("a.pdf — 인덱싱에서 제외했습니다");
		expect(indexToast("a.pdf", true)).toBe("a.pdf — 인덱싱에 포함했습니다");
	});
});
