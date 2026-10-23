import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join, resolve } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import {
	convertWithFilee,
	DEFAULT_FILEE_TIMEOUT_MS,
	FileeCliError,
	type FileeCliRun,
	listFileeFormats,
	resolveFileeExecutable,
} from "../../src/convert/index.ts";

const roots: string[] = [];
afterEach(() => {
	for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});
function tempRoot(): string {
	const root = mkdtempSync(join(tmpdir(), "filee-cli-"));
	roots.push(root);
	return root;
}

const OK_JSON = JSON.stringify({
	preset: "PDF",
	target: "pdf",
	converted: 1,
	skipped: 0,
	failed: 0,
	files: [{ source: "/in/a.docx", state: "done", outputs: ["/out/a.pdf"] }],
});

function fakeRun(run: FileeCliRun, seen: string[][] = []) {
	return async (args: readonly string[]): Promise<FileeCliRun> => {
		seen.push([...args]);
		return run;
	};
}

describe("convertWithFilee", () => {
	it("builds an absolute, --json invocation and parses the result", async () => {
		const seen: string[][] = [];
		const result = await convertWithFilee(
			{ inputs: ["docs/a.docx", "/abs/b.png"], to: "pdf" },
			{ cwd: "/work", run: fakeRun({ stdout: OK_JSON, stderr: "", exitCode: 0 }, seen) },
		);
		expect(seen).toEqual([
			["convert", resolve("/work", "docs/a.docx"), resolve("/abs/b.png"), "--to", "pdf", "--json"],
		]);
		expect(result.converted).toBe(1);
		expect(result.files[0]?.outputs).toEqual(["/out/a.pdf"]);
	});

	it("passes every option through, including --overwrite", async () => {
		const seen: string[][] = [];
		await convertWithFilee(
			{
				inputs: ["/in"],
				to: "webp",
				outputDir: "/out",
				recursive: true,
				quality: 80,
				overwrite: true,
				keepDates: true,
			},
			{ run: fakeRun({ stdout: OK_JSON, stderr: "", exitCode: 0 }, seen) },
		);
		expect(seen[0]).toEqual([
			"convert",
			resolve("/in"),
			"--to",
			"webp",
			"--json",
			"-o",
			resolve("/out"),
			"--recursive",
			"--quality",
			"80",
			"--overwrite",
			"--keep-dates",
		]);
	});

	it("maps skipExisting to --skip", async () => {
		const seen: string[][] = [];
		await convertWithFilee(
			{ inputs: ["/in/a.png"], to: "jpg", skipExisting: true },
			{ run: fakeRun({ stdout: OK_JSON, stderr: "", exitCode: 0 }, seen) },
		);
		expect(seen[0]).toContain("--skip");
		expect(seen[0]).not.toContain("--overwrite");
	});

	it("returns per-file failures (exit 1) instead of throwing", async () => {
		const json = JSON.stringify({
			preset: "WEBP",
			target: "webp",
			converted: 1,
			skipped: 0,
			failed: 1,
			files: [
				{ source: "/in/a.png", state: "done", outputs: ["/out/a.webp"] },
				{ source: "/in/b.png", state: "failed", outputs: [], error: "bad header" },
			],
		});
		const result = await convertWithFilee(
			{ inputs: ["/in"], to: "webp" },
			{ run: fakeRun({ stdout: json, stderr: "", exitCode: 1 }) },
		);
		expect(result.failed).toBe(1);
		expect(result.files[1]).toMatchObject({ state: "failed", error: "bad header" });
	});

	it("throws the usage message for exit 2 and 3", async () => {
		await expect(
			convertWithFilee(
				{ inputs: ["/in"], to: "nope" },
				{ run: fakeRun({ stdout: "", stderr: "filee: unknown format 'nope'.\nRun 'filee help'.\n", exitCode: 2 }) },
			),
		).rejects.toThrow("unknown format 'nope'");
		await expect(
			convertWithFilee(
				{ inputs: ["/empty"], to: "pdf" },
				{ run: fakeRun({ stdout: "", stderr: "filee: no files to convert.\n", exitCode: 3 }) },
			),
		).rejects.toThrow("no files to convert");
	});

	it("rejects malformed JSON and unexpected exit codes", async () => {
		await expect(
			convertWithFilee(
				{ inputs: ["/in"], to: "pdf" },
				{ run: fakeRun({ stdout: "oops", stderr: "", exitCode: 0 }) },
			),
		).rejects.toBeInstanceOf(FileeCliError);
		await expect(
			convertWithFilee(
				{ inputs: ["/in"], to: "pdf" },
				{ run: fakeRun({ stdout: "", stderr: "boom", exitCode: 139 }) },
			),
		).rejects.toThrow("exit code 139");
	});

	it("validates the request before spawning anything", async () => {
		const never = async (): Promise<FileeCliRun> => {
			throw new Error("must not spawn");
		};
		await expect(convertWithFilee({ inputs: [], to: "pdf" }, { run: never })).rejects.toThrow("at least one");
		await expect(convertWithFilee({ inputs: ["/a"], to: " " }, { run: never })).rejects.toThrow("target format");
		await expect(convertWithFilee({ inputs: ["/a"], to: "jpg", quality: 0 }, { run: never })).rejects.toThrow(
			"quality",
		);
		await expect(convertWithFilee({ inputs: ["/a"], to: "jpg", quality: 101 }, { run: never })).rejects.toThrow(
			"quality",
		);
		await expect(
			convertWithFilee({ inputs: ["/a"], to: "jpg", overwrite: true, skipExisting: true }, { run: never }),
		).rejects.toThrow("overwrite");
	});

	it("waits as long as 30 minutes by default", () => {
		expect(DEFAULT_FILEE_TIMEOUT_MS).toBe(30 * 60 * 1000);
	});

	it("runs a real executable, decodes multibyte output intact and honours FILEE_DATA_DIR", async () => {
		const root = tempRoot();
		const script = join(root, "fake-filee.mjs");
		writeFileSync(
			script,
			`#!/usr/bin/env node
const out = { preset: "X", target: "pdf", converted: 1, skipped: 0, failed: 0,
  files: [{ source: process.argv[3], state: "done", outputs: [process.env.FILEE_DATA_DIR + "/한글🙂.pdf"] }] };
process.stdout.write(JSON.stringify(out));
`,
		);
		chmodSync(script, 0o755);
		const result = await convertWithFilee(
			{ inputs: ["/in/한글.docx"], to: "pdf" },
			{ executable: script, dataDir: join(root, "data") },
		);
		expect(result.files[0]?.source).toBe(resolve("/in/한글.docx"));
		expect(result.files[0]?.outputs[0]).toBe(`${join(root, "data")}/한글🙂.pdf`);
	});

	it("kills a run that exceeds its timeout", async () => {
		const root = tempRoot();
		const script = join(root, "slow.mjs");
		writeFileSync(script, "#!/usr/bin/env node\nsetTimeout(() => {}, 60000);\n");
		chmodSync(script, 0o755);
		await expect(
			convertWithFilee({ inputs: ["/a.png"], to: "jpg" }, { executable: script, timeoutMs: 200 }),
		).rejects.toThrow("timed out");
	});

	it("stops when the caller aborts", async () => {
		const root = tempRoot();
		const script = join(root, "slow.mjs");
		writeFileSync(script, "#!/usr/bin/env node\nsetTimeout(() => {}, 60000);\n");
		chmodSync(script, 0o755);
		const controller = new AbortController();
		const pending = convertWithFilee(
			{ inputs: ["/a.png"], to: "jpg" },
			{ executable: script, signal: controller.signal },
		);
		setTimeout(() => controller.abort(), 100);
		await expect(pending).rejects.toThrow("aborted");
	});

	it("reports a missing executable without leaking a stack", async () => {
		await expect(
			convertWithFilee({ inputs: ["/a.png"], to: "jpg" }, { executable: join(tempRoot(), "no-such-filee") }),
		).rejects.toThrow("Unable to run filee");
	});
});

describe("listFileeFormats", () => {
	it("lists everything, or one format's targets", async () => {
		const seen: string[][] = [];
		const text = await listFileeFormats(undefined, {
			run: fakeRun({ stdout: "Images:\n  png\n", stderr: "", exitCode: 0 }, seen),
		});
		await listFileeFormats("heic", {
			run: fakeRun({ stdout: "HEIC converts to:\n  png\n", stderr: "", exitCode: 0 }, seen),
		});
		expect(text).toBe("Images:\n  png");
		expect(seen).toEqual([["formats"], ["formats", "heic"]]);
	});

	it("surfaces an unknown format", async () => {
		await expect(
			listFileeFormats("zzz", {
				run: fakeRun({ stdout: "", stderr: "filee: unknown format 'zzz'.\n", exitCode: 2 }),
			}),
		).rejects.toThrow("unknown format 'zzz'");
	});
});

describe("resolveFileeExecutable", () => {
	function bin(dir: string, name: string): string {
		mkdirSync(dir, { recursive: true });
		const path = join(dir, name);
		writeFileSync(path, "x");
		chmodSync(path, 0o755);
		return path;
	}

	it("finds `filee` on a POSIX PATH, preferring earlier directories", () => {
		const root = tempRoot();
		const first = bin(join(root, "a"), "filee");
		bin(join(root, "b"), "filee");
		const env = { PATH: [join(root, "a"), join(root, "b")].join(delimiter) };
		expect(resolveFileeExecutable(env, "darwin")).toBe(first);
	});

	it("falls back to filee-cli on POSIX", () => {
		const root = tempRoot();
		const cli = bin(join(root, "a"), "filee-cli");
		expect(resolveFileeExecutable({ PATH: join(root, "a") }, "linux")).toBe(cli);
	});

	it("returns undefined when nothing is installed", () => {
		expect(resolveFileeExecutable({ PATH: tempRoot() }, "darwin")).toBeUndefined();
		expect(resolveFileeExecutable({}, "darwin")).toBeUndefined();
	});

	it("on Windows maps the filee.cmd shim to filee-cli.exe next to its parent folder", () => {
		const root = tempRoot();
		const exe = bin(root, "filee-cli.exe");
		bin(join(root, "cli"), "filee.cmd");
		expect(resolveFileeExecutable({ Path: join(root, "cli") }, "win32")).toBe(exe);
	});

	it("on Windows never picks filee.exe, which is the GUI app", () => {
		const root = tempRoot();
		bin(join(root, "gui"), "filee.exe");
		expect(resolveFileeExecutable({ PATH: join(root, "gui") }, "win32")).toBeUndefined();
	});
});
