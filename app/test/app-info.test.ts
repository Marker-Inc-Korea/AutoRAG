import { describe, expect, it } from "vitest";
import { APP_NAME, APP_VERSION, formatWindowTitle } from "../src/shared/app-info";

describe("formatWindowTitle", () => {
	it("prefixes a bare semver with v", () => {
		expect(formatWindowTitle(APP_NAME, "1.2.3")).toBe("AutoRAG Finder v1.2.3");
	});

	it("does not double the v prefix", () => {
		expect(formatWindowTitle(APP_NAME, "v1.2.3")).toBe("AutoRAG Finder v1.2.3");
	});

	it("titles the shipped app version", () => {
		expect(formatWindowTitle(APP_NAME, APP_VERSION)).toBe("AutoRAG Finder v0.1.0");
	});
});
