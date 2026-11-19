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

	it("refuses a dupey older than 0.3.1 without running the scan", async () => {
		const { executable, log } = fakeDupey("0.1.2");

		const scan = scanWithDupey(tmpdir(), { executable });

		await expect(scan).rejects.toBeInstanceOf(DupeyCliError);
		await expect(scan).rejects.toThrow(/dupey 0\.1\.2 .* older than the 0\.3\.1/);
		expect(readFileSync(log, "utf8")).toBe("");
	});

	it.each(["0.2.0", "0.3.0"])(
		"refuses dupey %s: it drops unreadable folders from a scan and hashes UTF-16 or BOM text differently",
		async (version) => {
			const { executable, log } = fakeDupey(version);

			await expect(scanWithDupey(tmpdir(), { executable })).rejects.toThrow(
				new RegExp(`dupey ${version.replaceAll(".", "\\.")} .* older than the 0\\.3\\.1`),
			);
			expect(readFileSync(log, "utf8")).toBe("");
		},
	);

	it("runs a dupey at the floor, and newer ones", async () => {
		const { executable, log } = fakeDupey("0.3.1");

		await scanWithDupey(tmpdir(), { executable });
		await scanWithDupey(tmpdir(), { executable });

		expect(readFileSync(log, "utf8")).toBe("scan\nscan\n");
	});
});

describe.skipIf(process.platform === "win32")("dupey process output", () => {
	const dirs: string[] = [];
	afterEach(() => {
		for (const dir of dirs.splice(0)) rmSync(dir, { recursive: true, force: true });
	});

	/** A fake `dupey` that writes `body` to stdout in `chunkBytes`-sized writes, flushed one at a time. */
	function chunkedDupey(body: string, chunkBytes: number): string {
		const dir = mkdtempSync(join(tmpdir(), "dupey-chunks-"));
		dirs.push(dir);
		const bodyFile = join(dir, "body.json");
		writeFileSync(bodyFile, body);
		const executable = join(dir, "dupey");
		writeFileSync(
			executable,
			`#!/usr/bin/env node
if (process.argv[2] === "--version") { console.log("dupey 0.3.1"); process.exit(0); }
const fs = require("node:fs");
const bytes = fs.readFileSync(${JSON.stringify(bodyFile)});
let offset = 0;
const writeNext = () => {
  if (offset >= bytes.length) return;
  const end = Math.min(offset + ${chunkBytes}, bytes.length);
  const more = process.stdout.write(bytes.subarray(offset, end));
  offset = end;
  if (more) setImmediate(writeNext); else process.stdout.once("drain", writeNext);
};
writeNext();
`,
		);
		chmodSync(executable, 0o755);
		return executable;
	}

	it("keeps multi-byte characters intact when they straddle a pipe chunk boundary", async () => {
		// Every path is Korean (3 bytes per syllable) and the stream is cut into 7-byte writes, so almost every
		// chunk boundary lands inside a character. Decoding each chunk on its own turns those into U+FFFD.
		const paths = Array.from({ length: 40 }, (_, index) => `/문서/한국어 파일 이름 ${index}.txt`);
		const body = JSON.stringify({
			dir: "/문서",
			files: paths.map((path) => ({ path, content_hash: "h" })),
			families: [],
			errors: [],
		});
		const executable = chunkedDupey(body, 7);

		const scan = await scanWithDupey("/문서", { executable });

		expect(scan.files.map((file) => file.path)).toEqual(paths);
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
