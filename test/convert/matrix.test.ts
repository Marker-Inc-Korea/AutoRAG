import { describe, expect, it } from "vitest";
import {
	ROUTES,
	reachableTargets,
	renderConversionGuide,
	SOURCE_FAMILIES,
	TARGET_GROUPS,
} from "../../src/convert/index.ts";

describe("conversion matrix", () => {
	it("routes only name known source families and target groups", () => {
		for (const [family, route] of Object.entries(ROUTES)) {
			expect(SOURCE_FAMILIES[family], `family ${family}`).toBeDefined();
			for (const group of route.groups) expect(TARGET_GROUPS[group], `group ${group} in ${family}`).toBeDefined();
		}
		for (const family of Object.keys(SOURCE_FAMILIES)) expect(ROUTES[family], `route for ${family}`).toBeDefined();
	});

	it("never lists a format twice across the source families", () => {
		const seen = new Map<string, string>();
		for (const [family, members] of Object.entries(SOURCE_FAMILIES)) {
			for (const format of members) {
				expect(seen.get(format), `${format} in ${family} and ${seen.get(format)}`).toBeUndefined();
				seen.set(format, family);
			}
		}
	});

	it("answers the planning questions a user actually asks", () => {
		expect(reachableTargets("word")).toEqual(expect.arrayContaining(["pdf", "hwpx", "md", "png", "epub"]));
		expect(reachableTargets("hwp")).toEqual(expect.arrayContaining(["docx", "pdf", "md", "mp4"]));
		expect(reachableTargets("sheet")).toEqual(expect.arrayContaining(["csv", "json", "geojson", "pptx"]));
		expect(reachableTargets("video")).toEqual(expect.arrayContaining(["mp3", "srt", "png", "pdf"]));
		// Only inside their own family:
		expect(reachableTargets("audio")).not.toContain("pdf");
		expect(reachableTargets("font")).not.toContain("png");
		expect(reachableTargets("archive")).not.toContain("pdf");
		expect(reachableTargets("nonsense")).toEqual([]);
	});

	it("renders a guide with the families, groups, routes and the chaining rule", () => {
		const guide = renderConversionGuide();
		expect(guide).toContain("hwp: hwp hwpx");
		expect(guide).toContain("- hwp -> ");
		expect(guide).toContain("chain two conversions");
		expect(guide).toContain("outputDir");
		expect(guide.length).toBeLessThan(12_000);
	});
});
