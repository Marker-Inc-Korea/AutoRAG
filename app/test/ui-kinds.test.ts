import { describe, expect, it } from "vitest";
import { KIND_ORDER, kindFromExtension, kindMeta } from "../src/renderer/src/state/kinds";

describe("kindMeta", () => {
	it("carries the reference letter, tile token, and kind label", () => {
		expect(kindMeta("xlsx")).toEqual({
			letter: "X",
			tileToken: "--tile-xlsx",
			foregroundToken: "--text-on-tile",
			label: "Excel Spreadsheet",
		});
		expect(kindMeta("pdf").letter).toBe("P");
		expect(kindMeta("notion")).toEqual({
			letter: "N",
			tileToken: "--tile-notion",
			foregroundToken: "--tile-notion-fg",
			label: "Notion Page",
		});
	});

	it("labels folders without a tile letter", () => {
		expect(kindMeta("folder").letter).toBe("");
		expect(kindMeta("folder").label).toBe("Folder");
	});

	it("falls back to the md entry for an unknown kind", () => {
		expect(kindMeta("wat")).toEqual(kindMeta("md"));
	});

	it("covers every kind in the reference KD map", () => {
		expect(KIND_ORDER).toEqual([
			"folder",
			"xlsx",
			"csv",
			"pdf",
			"docx",
			"pptx",
			"md",
			"png",
			"zip",
			"slack",
			"mail",
			"notion",
			"discord",
			"telegram",
			"contact",
		]);
	});
});

describe("kindFromExtension", () => {
	it("maps the office and archive families", () => {
		expect(kindFromExtension("xlsx")).toBe("xlsx");
		expect(kindFromExtension("XLS")).toBe("xlsx");
		expect(kindFromExtension("docx")).toBe("docx");
		expect(kindFromExtension("pptx")).toBe("pptx");
		expect(kindFromExtension("tar")).toBe("zip");
	});

	it("maps images to the png tile", () => {
		expect(kindFromExtension("jpeg")).toBe("png");
		expect(kindFromExtension("heic")).toBe("png");
	});

	it("falls back to md for anything else", () => {
		expect(kindFromExtension("")).toBe("md");
		expect(kindFromExtension("weird")).toBe("md");
	});
});
