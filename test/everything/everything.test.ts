import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import JSZip from "jszip";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	buildEverythingIni,
	buildEverythingSearchArgs,
	type EverythingBundleManifest,
	EverythingClient,
	ensureEverythingBinaries,
	everythingInstanceName,
	parseEverythingJson,
} from "../../src/everything/index.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-everything-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function sha256(bytes: Buffer): string {
	return createHash("sha256").update(bytes).digest("hex");
}

async function zipWith(member: string, bytes: Buffer): Promise<Buffer> {
	const zip = new JSZip();
	zip.file(member, bytes);
	return Buffer.from(await zip.generateAsync({ type: "uint8array" }));
}

async function writeBundle(options: { tamperEs?: boolean } = {}): Promise<{
	bundleDir: string;
	manifest: EverythingBundleManifest;
}> {
	const bundleDir = join(root, "bundle");
	mkdirSync(bundleDir, { recursive: true });
	const everythingBytes = Buffer.from("everything-binary");
	const esBytes = Buffer.from("es-binary");
	const everythingZip = await zipWith("everything.exe", everythingBytes);
	const esZip = await zipWith("es.exe", options.tamperEs ? Buffer.from("tampered") : esBytes);
	writeFileSync(join(bundleDir, "Everything.zip"), everythingZip);
	writeFileSync(join(bundleDir, "ES.zip"), esZip);
	const manifest: EverythingBundleManifest = {
		everythingVersion: "1.4.1.1032",
		esVersion: "1.1.0.38",
		licenseFiles: [],
		architectures: {
			x64: {
				everythingArchive: "Everything.zip",
				everythingArchiveSha256: sha256(everythingZip),
				everythingMember: "everything.exe",
				everythingSha256: sha256(everythingBytes),
				esArchive: "ES.zip",
				esArchiveSha256: sha256(esZip),
				esMember: "es.exe",
				esSha256: sha256(esBytes),
			},
		},
	};
	return { bundleDir, manifest };
}

describe("ensureEverythingBinaries", () => {
	it("extracts the bundled portable Everything and ES into the workspace cache on Windows", async () => {
		const { bundleDir, manifest } = await writeBundle();
		const result = await ensureEverythingBinaries({
			root,
			platform: "win32",
			arch: "x64",
			bundleDir,
			manifest,
		});
		expect(result).toMatchObject({ ok: true, source: "installed" });
		if (!result.ok) throw new Error("expected install");
		expect(readFileSync(result.everythingPath, "utf8")).toBe("everything-binary");
		expect(readFileSync(result.esPath, "utf8")).toBe("es-binary");
		expect(result.everythingPath.startsWith(join(root, ".autorag", "everything", "1.4.1.1032"))).toBe(true);

		const again = await ensureEverythingBinaries({ root, platform: "win32", arch: "x64", bundleDir, manifest });
		expect(again).toMatchObject({ ok: true, source: "cached" });
	});

	it("refuses a bundle whose extracted binary does not match the pinned digest", async () => {
		const { bundleDir, manifest } = await writeBundle({ tamperEs: true });
		const tampered = {
			...manifest,
			architectures: {
				x64: {
					...manifest.architectures.x64!,
					esArchiveSha256: sha256(readFileSync(join(bundleDir, "ES.zip"))),
				},
			},
		};
		const result = await ensureEverythingBinaries({
			root,
			platform: "win32",
			arch: "x64",
			bundleDir,
			manifest: tampered,
		});
		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected failure");
		expect(result.message).toContain("es.exe");
		expect(result.message).toContain("SHA-256");
		expect(existsSync(join(root, ".autorag", "everything", "1.4.1.1032", "x64", "es.exe"))).toBe(false);
	});

	it("is unavailable outside Windows", async () => {
		const { bundleDir, manifest } = await writeBundle();
		const result = await ensureEverythingBinaries({ root, platform: "darwin", arch: "arm64", bundleDir, manifest });
		expect(result).toMatchObject({ ok: false, reason: "unsupported-platform" });
	});

	it("reports an unsupported Windows architecture verbatim", async () => {
		const { bundleDir, manifest } = await writeBundle();
		const result = await ensureEverythingBinaries({ root, platform: "win32", arch: "ia32", bundleDir, manifest });
		expect(result).toMatchObject({ ok: false, reason: "unsupported-platform" });
		if (result.ok) throw new Error("expected failure");
		expect(result.message).toContain("ia32");
	});
});

describe("buildEverythingIni", () => {
	it("indexes only configured folders and never requests elevation, volume scans, or network servers", () => {
		const ini = buildEverythingIni({
			folders: ["C:\\Users\\me\\docs dir", "D:\\a,folder"],
			excludeFolders: ["C:\\Users\\me\\docs dir\\.autorag"],
		});
		const lines = ini.split("\r\n");
		expect(lines[0]).toBe("[Everything]");
		expect(lines).toContain("run_as_admin=0");
		expect(lines).toContain("app_data=0");
		expect(lines).toContain("auto_include_fixed_volumes=0");
		expect(lines).toContain("auto_include_removable_volumes=0");
		expect(lines).toContain("auto_include_fixed_refs_volumes=0");
		expect(lines).toContain("auto_include_removable_refs_volumes=0");
		expect(lines).toContain("allow_http_server=0");
		expect(lines).toContain("allow_etp_server=0");
		expect(lines).toContain("check_for_updates_on_startup=0");
		expect(lines).toContain("show_tray_icon=0");
		expect(lines).toContain('folders="C:\\\\Users\\\\me\\\\docs dir","D:\\\\a,folder"');
		expect(lines).toContain("folder_monitor_changes=1,1");
		expect(lines).toContain('exclude_folders="C:\\\\Users\\\\me\\\\docs dir\\\\.autorag"');
		expect(ini.endsWith("\r\n")).toBe(true);
	});
});

describe("everythingInstanceName", () => {
	it("is stable per workspace and distinct across workspaces", () => {
		const a = everythingInstanceName("C:\\work\\a");
		expect(a).toMatch(/^autorag-[0-9a-f]{12}$/);
		expect(everythingInstanceName("C:\\work\\a")).toBe(a);
		expect(everythingInstanceName("C:\\work\\b")).not.toBe(a);
	});
});

describe("buildEverythingSearchArgs", () => {
	it("uses CommandLineToArgvW parsing and passes the query after -- so leading dashes are never ES switches", () => {
		const args = buildEverythingSearchArgs("autorag-abc", { query: "-weird name", maxResults: 50 });
		expect(args.slice(0, 7)).toEqual([
			"-argv",
			"-instance",
			"autorag-abc",
			"-cp",
			"65001",
			"-json",
			"-full-path-and-name",
		]);
		expect(args).toContain("-size");
		expect(args).toContain("-date-modified");
		expect(args.slice(-4)).toEqual(["-n", "50", "--", "-weird name"]);
	});

	it("maps typed filters to ES switches", () => {
		const args = buildEverythingSearchArgs("i", {
			query: "^refund.*\\.pdf$",
			regex: true,
			matchCase: true,
			matchPath: true,
			wholeWord: true,
			kind: "folders",
			path: "C:\\docs",
			sort: "date-modified-descending",
			offset: 10,
			maxResults: 5,
		});
		expect(args).toEqual(
			expect.arrayContaining([
				"-regex",
				"-case",
				"-match-path",
				"-whole-word",
				"/ad",
				"-path",
				"C:\\docs",
				"-sort",
				"date-modified-descending",
				"-offset",
				"10",
			]),
		);
		expect(args.indexOf("-regex")).toBe(args.length - 2);
		expect(args.at(-1)).toBe("^refund.*\\.pdf$");
	});
});

describe("parseEverythingJson", () => {
	it("parses ES JSON with UTF-8 names, folder suffixes, and FILETIME dates", () => {
		const stdout =
			'[{"filename":"C:\\\\docs\\\\하위\\\\","size":null,"date_modified":"2026-10-01T17:09:33"},\r\n' +
			'{"filename":"C:\\\\docs\\\\하위\\\\환불 정책.txt","size":3,"date_modified":"2026-10-01T17:09:33"}]\r\n';
		expect(parseEverythingJson(stdout)).toEqual([
			{ path: "C:\\docs\\하위", type: "folder", size: undefined, dateModified: "2026-10-01T17:09:33" },
			{ path: "C:\\docs\\하위\\환불 정책.txt", type: "file", size: 3, dateModified: "2026-10-01T17:09:33" },
		]);
	});

	it("treats empty ES output as zero results", () => {
		expect(parseEverythingJson("")).toEqual([]);
		expect(parseEverythingJson("\r\n")).toEqual([]);
	});

	it("throws the raw output when ES does not return a JSON array", () => {
		expect(() => parseEverythingJson("Error 8: Everything IPC not found.")).toThrow(/Error 8: Everything IPC/);
	});
});

describe("EverythingClient", () => {
	function fakeClient(responses: Array<{ code: number; stdout?: string; stderr?: string }>) {
		const calls: Array<{ command: string; args: readonly string[] }> = [];
		const launched: Array<{ command: string; args: readonly string[] }> = [];
		const client = new EverythingClient({
			root,
			folders: [join(root, "docs")],
			platform: "win32",
			resolveBinaries: async () => ({
				ok: true,
				everythingPath: "C:\\cache\\everything.exe",
				esPath: "C:\\cache\\es.exe",
				source: "cached",
			}),
			run: async (command, args) => {
				calls.push({ command, args });
				const next = responses.shift() ?? { code: 0, stdout: "" };
				return { code: next.code, stdout: next.stdout ?? "", stderr: next.stderr ?? "" };
			},
			launch: (command, args) => {
				launched.push({ command, args });
			},
			startupTimeoutMs: 200,
			pollIntervalMs: 1,
		});
		return { client, calls, launched };
	}

	it("starts a private user-level instance with its own config and db, then searches", async () => {
		const { client, calls, launched } = fakeClient([
			{ code: 8, stderr: "Error 8: Everything IPC not found." },
			{ code: 0, stdout: "1.4.1.1032\r\n" },
			{ code: 0, stdout: '[{"filename":"C:\\\\docs\\\\a.txt","size":1,"date_modified":"2026-10-01T00:00:00"}]' },
		]);
		const result = await client.search({ query: "a" });
		expect(result).toEqual({
			ok: true,
			results: [{ path: "C:\\docs\\a.txt", type: "file", size: 1, dateModified: "2026-10-01T00:00:00" }],
		});
		expect(launched).toHaveLength(1);
		const launchArgs = launched[0]!.args;
		expect(launched[0]!.command).toBe("C:\\cache\\everything.exe");
		expect(launchArgs).toEqual(
			expect.arrayContaining(["-instance", client.instanceName, "-startup", "-config", "-db"]),
		);
		expect(launchArgs).not.toContain("-admin");
		expect(launchArgs).not.toContain("-install-service");
		const ini = readFileSync(join(root, ".autorag", "everything", "Everything.ini"), "utf8");
		expect(ini).toContain("run_as_admin=0");
		expect(calls.at(-1)!.args).toEqual(expect.arrayContaining(["-instance", client.instanceName, "-json"]));
	});

	it("surfaces ES exit status and stderr verbatim when a search fails", async () => {
		const { client } = fakeClient([
			{ code: 0, stdout: "1.4.1.1032\r\n" },
			{ code: 4, stderr: "Error 4: Expected switch parameter." },
		]);
		const result = await client.search({ query: "x" });
		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected failure");
		expect(result.message).toContain("exit 4");
		expect(result.message).toContain("Error 4: Expected switch parameter.");
	});

	it("reports a startup failure with the last ES error when the instance never answers", async () => {
		const { client } = fakeClient(
			Array.from({ length: 500 }, () => ({ code: 8, stderr: "Error 8: Everything IPC not found." })),
		);
		const result = await client.index();
		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected failure");
		expect(result.message).toContain("Error 8: Everything IPC not found.");
	});

	it("restarts a running instance with the rewritten folder config and waits for the indexed count", async () => {
		const { client, calls, launched } = fakeClient([
			{ code: 0, stdout: "1.4.1.1032\r\n" },
			{ code: 0 },
			{ code: 0, stdout: "1.4.1.1032\r\n" },
			{ code: 0 },
			{ code: 0, stdout: "3\r\n" },
		]);
		const result = await client.index();
		expect(result).toEqual({ ok: true, indexedFolders: [join(root, "docs")], indexedItems: 3 });
		expect(calls.map((call) => call.args[2])).toEqual([
			"-get-everything-version",
			"-exit",
			"-get-everything-version",
			"-save-db",
			"-get-result-count",
		]);
		expect(launched).toHaveLength(1);
	});

	it("does nothing on non-Windows platforms", async () => {
		const client = new EverythingClient({ root, folders: [root], platform: "linux" });
		expect(client.isSupported()).toBe(false);
		expect(await client.search({ query: "a" })).toMatchObject({ ok: false, reason: "unsupported-platform" });
	});
});
