import { describe, expect, it } from "vitest";
import { homeCloudRoots, pathLooksLikeCloudRoot } from "../../src/filesystem/cloud-placeholder.ts";

describe("cloud placeholder detection", () => {
	it("recognizes macOS File Provider roots including Google Drive", () => {
		expect(pathLooksLikeCloudRoot("/Users/x/Library/CloudStorage/GoogleDrive-user/My Drive")).toBe(true);
		expect(pathLooksLikeCloudRoot("/Users/x/Library/Mobile Documents/com~apple~CloudDocs")).toBe(true);
	});

	it("recognizes Windows OneDrive and Google Drive mounts", () => {
		expect(pathLooksLikeCloudRoot(String.raw`C:\Users\x\OneDrive - Company\Documents`)).toBe(true);
		expect(pathLooksLikeCloudRoot(String.raw`G:\Google Drive\My Drive\Documents`)).toBe(true);
		expect(pathLooksLikeCloudRoot(String.raw`G:\My Drive\Documents`)).toBe(true);
		expect(pathLooksLikeCloudRoot(String.raw`C:\Users\x\Documents`)).toBe(false);
	});

	it("lists macOS home cloud roots", () => {
		expect(homeCloudRoots("/Users/demo")).toEqual([
			"/Users/demo/Library/CloudStorage",
			"/Users/demo/Library/Mobile Documents",
		]);
	});
});
