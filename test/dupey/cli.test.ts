import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { DupeyCliError, fingerprintWithDupey, scanWithDupey } from "../../src/dupey/index.ts";

describe("dupey CLI adapter", () => {
	it("runs JSON scan and normalizes the directory", async () => {
		const result = await scanWithDupey("docs", {
			run: async (args) => {
				expect(args).toEqual(["scan", resolve("docs"), "--json"]);
				return JSON.stringify({ dir: "/docs", files: [], families: [], errors: [] });
			},
		});
		expect(result.files).toEqual([]);
	});

	it("passes one --exclude-dir per nested cloud folder name", async () => {
		await scanWithDupey("docs", {
			excludeDirs: ["GoogleDrive-x", "OneDrive - Work"],
			run: async (args) => {
				expect(args).toEqual([
					"scan",
					resolve("docs"),
					"--json",
					"--exclude-dir",
					"GoogleDrive-x",
					"--exclude-dir",
					"OneDrive - Work",
				]);
				return JSON.stringify({ dir: "/docs", files: [], families: [], errors: [] });
			},
		});
	});

	it("rejects malformed output", async () => {
		await expect(scanWithDupey(".", { run: async () => "not-json" })).rejects.toBeInstanceOf(DupeyCliError);
	});
});

describe("dupey fingerprint adapter", () => {
	// Measured on dupey 0.1.2: `fingerprint` prints tab-separated path/format/chars/exact/minhash_width/modified/revision
	// lines, and `exact` equals `content_hash` from `scan --json` for the same file.
	const sample = [
		"path\t/tmp/a.txt",
		"format\tTxt",
		"chars\t18",
		"exact\t993e53379ba505badfc13f1287a4e784499f666e59724b2314fbd3d9138ab5d5",
		"minhash_width\t128",
		"modified\t-",
		"revision\t-",
		"",
	].join("\n");

	it("parses the tab-separated fingerprint output", async () => {
		const fingerprint = await fingerprintWithDupey("/tmp/a.txt", {
			run: async (args) => {
				expect(args).toEqual(["fingerprint", resolve("/tmp/a.txt")]);
				return sample;
			},
		});

		expect(fingerprint).toEqual({
			path: "/tmp/a.txt",
			format: "Txt",
			chars: 18,
			exact: "993e53379ba505badfc13f1287a4e784499f666e59724b2314fbd3d9138ab5d5",
			minhashWidth: 128,
			modified: null,
			revision: null,
		});
	});

	it("keeps a document's own modified time when dupey records one", async () => {
		const fingerprint = await fingerprintWithDupey("/tmp/a.docx", {
			run: async () => "path\t/tmp/a.docx\nexact\tabc\nmodified\t2024-05-01T10:00:00Z\nrevision\tr2\n",
		});

		expect(fingerprint.modified).toBe("2024-05-01T10:00:00Z");
		expect(fingerprint.revision).toBe("r2");
	});

	it("rejects output without an exact hash", async () => {
		await expect(fingerprintWithDupey("/tmp/a.png", { run: async () => "format\tPng\n" })).rejects.toBeInstanceOf(
			DupeyCliError,
		);
	});
});
