import { describe, expect, it } from "vitest";
import { normalizeBaseUrl, resolveBaseUrl } from "../../src/cloud/config.ts";

describe("normalizeBaseUrl", () => {
	it("trims whitespace and strips trailing slashes", () => {
		expect(normalizeBaseUrl("  https://api.dazziapp.com///  ")).toBe("https://api.dazziapp.com");
		expect(normalizeBaseUrl("https://api.dazziapp.com/")).toBe("https://api.dazziapp.com");
		expect(normalizeBaseUrl("https://api.dazziapp.com")).toBe("https://api.dazziapp.com");
		expect(normalizeBaseUrl("  https://api.dazziapp.com/v1  ")).toBe("https://api.dazziapp.com/v1");
	});

	it("handles empty and slash-only input", () => {
		expect(normalizeBaseUrl("")).toBe("");
		expect(normalizeBaseUrl("   ")).toBe("");
		expect(normalizeBaseUrl("/")).toBe("");
		expect(normalizeBaseUrl("////")).toBe("");
	});

	it("scans a long run of slashes followed by a non-slash in linear time", () => {
		const pathological = `${"/".repeat(200_000)}x`;
		const started = performance.now();
		expect(normalizeBaseUrl(pathological)).toBe(pathological);
		expect(performance.now() - started).toBeLessThan(1_000);
	});

	it("strips a long run of trailing slashes quickly", () => {
		const started = performance.now();
		expect(normalizeBaseUrl(`https://api.dazziapp.com${"/".repeat(200_000)}`)).toBe("https://api.dazziapp.com");
		expect(performance.now() - started).toBeLessThan(1_000);
	});
});

describe("resolveBaseUrl", () => {
	it("uses AUTORAG_BASE_URL when set and falls back to the default otherwise", () => {
		expect(resolveBaseUrl({ AUTORAG_BASE_URL: "https://staging.example.com//" })).toBe("https://staging.example.com");
		expect(resolveBaseUrl({})).toBe("https://api.dazziapp.com");
		expect(resolveBaseUrl({ AUTORAG_BASE_URL: "   " })).toBe("https://api.dazziapp.com");
	});
});
