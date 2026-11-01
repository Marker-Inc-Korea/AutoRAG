import { describe, expect, it } from "vitest";
import {
	datalessFromDarwinFlags,
	homeCloudRoots,
	pathLooksLikeCloudRoot,
} from "../../src/filesystem/cloud-placeholder.ts";

describe("darwin dataless flag parsing", () => {
	// Values measured with `/usr/bin/stat -f %f` on macOS 26: a File Provider placeholder reads
	// 1073741920 (0x40000060, `compressed,dataless`); a plain hidden file reads 32768 (0x8000).
	it("detects a File Provider placeholder from the decimal st_flags stat prints", () => {
		expect(datalessFromDarwinFlags(1073741920)).toBe(true);
	});

	it("does not mistake the UF_HIDDEN bit for a placeholder", () => {
		expect(datalessFromDarwinFlags(32768)).toBe(false);
	});

	it("treats an ordinary materialized file as local", () => {
		expect(datalessFromDarwinFlags(0)).toBe(false);
		expect(datalessFromDarwinFlags(0x20)).toBe(false);
	});
});

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
