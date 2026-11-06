import { chmodSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
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

describe.skipIf(process.platform === "win32")("dupey version floor", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
	});

	/** A fake `dupey` that reports `version` and logs every other command it is asked to run. */
	function fakeDupey(version: string): { readonly executable: string; readonly log: string } {
		const dir = mkdtempSync(join(tmpdir(), "dupey-version-"));
		dirs.push(dir);
		const executable = join(dir, "dupey");
		const log = join(dir, "calls.log");
		writeFileSync(log, "");
		writeFileSync(
			executable,
			`#!/bin/sh\nif [ "$1" = "--version" ]; then echo "dupey ${version}"; exit 0; fi\necho "$1" >> "${log}"\necho '{"dir":"/d","files":[],"families":[],"errors":[]}'\n`,
		);
		chmodSync(executable, 0o755);
		return { executable, log };
	}

	it("refuses a dupey older than 0.2.0 without running the scan", async () => {
		const { executable, log } = fakeDupey("0.1.2");

		const scan = scanWithDupey(tmpdir(), { executable });

		await expect(scan).rejects.toBeInstanceOf(DupeyCliError);
		await expect(scan).rejects.toThrow(/dupey 0\.1\.2 .* older than the 0\.2\.0/);
		expect(readFileSync(log, "utf8")).toBe("");
	});

	it("runs a dupey at the floor", async () => {
		const { executable, log } = fakeDupey("0.2.0");

		await scanWithDupey(tmpdir(), { executable });
		await scanWithDupey(tmpdir(), { executable });

		expect(readFileSync(log, "utf8")).toBe("scan\nscan\n");
	});
});

describe("dupey fingerprint adapter", () => {
	// Measured on dupey 0.2.0: `fingerprint` prints tab-separated path/format/chars/exact/minhash_width/modified/revision
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
