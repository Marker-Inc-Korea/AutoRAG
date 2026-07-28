import { describe, expect, it } from "vitest";
import { planSourceRoots } from "../../src/filesystem/source-paths.js";
import { normalizeVirtualPath } from "../../src/retrieval/scope.js";

function medianMs(run: () => void, samples = 5): number {
	const durations: number[] = [];
	for (let index = 0; index < samples; index += 1) {
		const started = performance.now();
		run();
		durations.push(performance.now() - started);
	}
	return durations.sort((a, b) => a - b)[Math.floor(samples / 2)];
}

describe("planSourceRoots trailing-separator handling (CodeQL js/polynomial-redos #19)", () => {
	it("derives the prefix from the final path segment", () => {
		expect(planSourceRoots(["/tmp/ulw-docs"])[0].prefix).toBe("/ulw-docs");
		expect(planSourceRoots(["/tmp/ulw-docs/"])[0].prefix).toBe("/ulw-docs");
		expect(planSourceRoots(["/tmp/ulw-docs///"])[0].prefix).toBe("/ulw-docs");
	});

	it("disambiguates duplicate basenames", () => {
		const roots = planSourceRoots(["/a/docs", "/b/docs"]);
		expect(roots.map((root) => root.prefix).sort()).toEqual(["/docs", "/docs-2"]);
	});

	it("does not degrade quadratically on a long trailing backslash run", () => {
		// node:path.resolve() collapses '/' runs but leaves '\' untouched on POSIX,
		// so backslashes reach the trailing-separator regex at full length.
		const build = (count: number) => `/tmp/ulw-docs${"\\".repeat(count)}x`;
		const small = medianMs(() => planSourceRoots([build(16_000)]), 3);
		const large = medianMs(() => planSourceRoots([build(64_000)]), 3);

		// 4x the input must not cost ~16x the time (the quadratic signature).
		expect(large).toBeLessThan(Math.max(small, 0.05) * 8);
		expect(large).toBeLessThan(50);
	});
});

describe("normalizeVirtualPath trailing-separator handling (CodeQL js/polynomial-redos #20)", () => {
	it("collapses separators and strips trailing ones", () => {
		expect(normalizeVirtualPath("/docs/reports///")).toBe("/docs/reports");
		expect(normalizeVirtualPath("/docs//reports")).toBe("/docs/reports");
		expect(normalizeVirtualPath("docs/reports/")).toBe("/docs/reports");
		expect(normalizeVirtualPath("///")).toBe("/");
		expect(normalizeVirtualPath("")).toBe("/");
	});

	it("normalizes backslash separators", () => {
		expect(normalizeVirtualPath("\\docs\\reports\\")).toBe("/docs/reports");
	});

	it("stays fast on adversarial separator runs", () => {
		const slashes = `/docs${"/".repeat(64_000)}x`;
		const backslashes = `/docs${"\\".repeat(64_000)}x`;
		expect(medianMs(() => normalizeVirtualPath(slashes), 3)).toBeLessThan(50);
		expect(medianMs(() => normalizeVirtualPath(backslashes), 3)).toBeLessThan(50);
	});
});
