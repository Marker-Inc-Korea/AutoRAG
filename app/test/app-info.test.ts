import { describe, expect, it } from "vitest";
import { APP_NAME, APP_VERSION, formatDevLabel, formatWindowTitle } from "../src/shared/app-info";

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

	it("appends the clone label for dev runs", () => {
		expect(
			formatWindowTitle(APP_NAME, APP_VERSION, {
				clonePath: "/clones/autorag",
				branch: "feat/x",
				commit: "612ae5f",
			}),
		).toBe("AutoRAG Finder v0.1.0 — /clones/autorag (feat/x@612ae5f)");
	});

	it("treats a null label as a shipped title", () => {
		expect(formatWindowTitle(APP_NAME, APP_VERSION, null)).toBe("AutoRAG Finder v0.1.0");
	});
});

describe("formatDevLabel", () => {
	it("prints path, branch, and short commit", () => {
		expect(formatDevLabel({ clonePath: "/clones/autorag", branch: "main", commit: "0bfb85a" })).toBe(
			"/clones/autorag (main@0bfb85a)",
		);
	});

	it("marks a detached HEAD", () => {
		expect(formatDevLabel({ clonePath: "/clones/autorag", branch: null, commit: "612ae5f" })).toBe(
			"/clones/autorag (detached@612ae5f)",
		);
	});

	it("prints only the path without git metadata", () => {
		expect(formatDevLabel({ clonePath: "/clones/autorag", branch: null, commit: null })).toBe("/clones/autorag");
	});
});
