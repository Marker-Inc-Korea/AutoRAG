import { describe, expect, it } from "vitest";
import {
	activeNavPath,
	basename,
	breadcrumbTrail,
	dirname,
	joinPath,
	pathSegments,
	whereSegments,
} from "../src/renderer/src/state/paths";

describe("path helpers", () => {
	it("splits relative and absolute paths", () => {
		expect(pathSegments("Documents/Finance/2026 Q3")).toEqual(["Documents", "Finance", "2026 Q3"]);
		expect(pathSegments("/Users/jeffrey/Documents")).toEqual(["Users", "jeffrey", "Documents"]);
		expect(pathSegments("")).toEqual([]);
	});

	it("reads the last segment", () => {
		expect(basename("Documents/Finance")).toBe("Finance");
		expect(basename("/Users/jeffrey/a.pdf")).toBe("a.pdf");
		expect(basename("Desktop")).toBe("Desktop");
	});

	it("reads the parent directory", () => {
		expect(dirname("Documents/Finance/a.pdf")).toBe("Documents/Finance");
		expect(dirname("/Users/jeffrey/a.pdf")).toBe("/Users/jeffrey");
		expect(dirname("Desktop")).toBe("");
		expect(dirname("/a.pdf")).toBe("/");
	});

	it("joins a directory and a name", () => {
		expect(joinPath("Documents", "a.pdf")).toBe("Documents/a.pdf");
		expect(joinPath("/Users/jeffrey", "a.pdf")).toBe("/Users/jeffrey/a.pdf");
		expect(joinPath("/", "a.pdf")).toBe("/a.pdf");
		expect(joinPath("", "Desktop")).toBe("Desktop");
	});

	it("renders the Where column with the reference separator", () => {
		expect(whereSegments("Documents/Finance/2026 Q3")).toBe("Documents › Finance › 2026 Q3");
	});
});

describe("breadcrumbTrail", () => {
	it("shows a single segment with no separator", () => {
		expect(breadcrumbTrail("Desktop")).toEqual([
			{ label: "Desktop", path: "Desktop", separator: false, isLast: true },
		]);
	});

	it("shows two segments with one separator", () => {
		expect(breadcrumbTrail("Documents/Finance")).toEqual([
			{ label: "Documents", path: "Documents", separator: false, isLast: false },
			{ label: "Finance", path: "Documents/Finance", separator: true, isLast: true },
		]);
	});

	it("keeps the last two segments and prefixes an ellipsis crumb", () => {
		expect(breadcrumbTrail("Documents/Finance/2026 Q3/벤더 견적")).toEqual([
			{ label: "…", path: "Documents/Finance", separator: false, isLast: false },
			{ label: "2026 Q3", path: "Documents/Finance/2026 Q3", separator: true, isLast: false },
			{ label: "벤더 견적", path: "Documents/Finance/2026 Q3/벤더 견적", separator: true, isLast: true },
		]);
	});

	it("keeps absolute crumb paths navigable", () => {
		const trail = breadcrumbTrail("/Users/jeffrey/Documents/Finance");
		expect(trail.map((c) => c.path)).toEqual([
			"/Users/jeffrey",
			"/Users/jeffrey/Documents",
			"/Users/jeffrey/Documents/Finance",
		]);
		expect(trail[0]?.label).toBe("…");
	});

	it("returns nothing for an empty path", () => {
		expect(breadcrumbTrail("")).toEqual([]);
	});
});

describe("activeNavPath", () => {
	const nav = [
		{ name: "Recents", path: "Recents" },
		{ name: "Documents", path: "/Users/jeffrey/Documents" },
		{ name: "Desktop", path: "/Users/jeffrey/Desktop" },
	];

	it("matches the nav entry that owns the current path", () => {
		expect(activeNavPath("/Users/jeffrey/Documents/Finance", nav)).toBe("/Users/jeffrey/Documents");
		expect(activeNavPath("Recents", nav)).toBe("Recents");
	});

	it("prefers the longest matching entry", () => {
		const nested = [
			{ name: "Home", path: "/Users/jeffrey" },
			{ name: "Documents", path: "/Users/jeffrey/Documents" },
		];
		expect(activeNavPath("/Users/jeffrey/Documents/Finance", nested)).toBe("/Users/jeffrey/Documents");
	});

	it("does not match a partial segment", () => {
		expect(activeNavPath("/Users/jeffrey/Documents-old", nav)).toBeNull();
	});

	it("returns null when nothing matches", () => {
		expect(activeNavPath("Slack/#general", nav)).toBeNull();
	});
});
