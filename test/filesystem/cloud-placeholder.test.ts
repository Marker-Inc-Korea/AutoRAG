import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	buildWindowsAttributeScript,
	buildWindowsDehydrateScript,
	buildWindowsDiskScript,
	type CloudDirEntry,
	type CloudExec,
	type CloudRoot,
	canonicalPath,
	canonicalPathMapper,
	cloudProviderFromDomain,
	cloudProviderFromPath,
	cloudScope,
	DarwinCloudProbe,
	datalessFromDarwinFlags,
	datalessFromWindowsAttributes,
	defaultCloudProbe,
	homeCloudRoots,
	isPathInside,
	networkMountPoints,
	type PowerShellResult,
	parseDarwinMounts,
	parseWindowsAttributeOutput,
	parseWindowsDehydrateOutput,
	parseWindowsLogicalDisks,
	parseWindowsSyncRoots,
	parseXattrValues,
	pathLooksLikeCloudRoot,
	providerFromWindowsSyncRoot,
	uncShareRoot,
	WINDOWS_ATTRIBUTE_BATCH,
	WINDOWS_DRIVE_TYPE_NETWORK,
	WindowsCloudProbe,
	windowsRootsFromDisks,
	XATTR_BATCH,
} from "../../src/filesystem/cloud-placeholder.ts";

/** The comparable form of a path: `join` uses `\` on Windows while every root comparison works in `/`. */
const slash = (path: string): string => path.replaceAll("\\", "/");

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

	it("names the provider behind a File Provider domain and a mount path", () => {
		expect(cloudProviderFromDomain("com.apple.CloudDocs.iCloudDriveFileProvider/1A2B")).toBe("icloud");
		expect(cloudProviderFromDomain("com.google.drivefs.files/1A2B")).toBe("google-drive");
		expect(cloudProviderFromDomain("com.microsoft.OneDrive-mac/1A2B")).toBe("onedrive");
		expect(cloudProviderFromDomain("com.getdropbox.dropbox.fileprovider/1A2B")).toBe("dropbox");
		expect(cloudProviderFromDomain("com.example.unknown/1A2B")).toBe("file-provider");
		expect(cloudProviderFromPath("/Users/x/Library/CloudStorage/GoogleDrive-user/My Drive")).toBe("google-drive");
		expect(cloudProviderFromPath("/Users/x/Library/CloudStorage/OneDrive-Company")).toBe("onedrive");
		expect(cloudProviderFromPath("/Users/x/Library/CloudStorage/Dropbox")).toBe("dropbox");
		expect(cloudProviderFromPath("/Users/x/Library/Mobile Documents/com~apple~CloudDocs")).toBe("icloud");
	});
});

describe("darwin probe release path", () => {
	it("releases through osascript -l JavaScript and never dereferences the NSError out-param", async () => {
		const calls: { readonly file: string; readonly args: readonly string[] }[] = [];
		const exec: CloudExec = async (file, args) => {
			calls.push({ file, args });
			return { stdout: '{"ok":true}\n', stderr: "" };
		};

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/file.txt");

		expect(result).toEqual({ ok: true });
		expect(calls).toHaveLength(1);
		expect(calls[0]?.file).toBe("/usr/bin/osascript");
		expect(calls[0]?.args.slice(0, 4)).toEqual(["-l", "JavaScript", "-e", expect.any(String)]);
		// The release helper always receives the target path as its last argument.
		expect(calls[0]?.args.at(-1)).toBe("/tmp/cloud/file.txt");
		const program = calls[0]!.args[3] ?? "";
		expect(program).toContain("evictUbiquitousItemAtURLError");
		// Reading the failed eviction's NSError out-param segfaults osascript (measured exit 139), so the program
		// must pass the Ref and never read it; the refusal reason comes from resource keys instead.
		expect(program).not.toMatch(/errorRef\s*\[/u);
		expect(program).toContain("NSURLUbiquitousItemIsUploadingKey");
	});

	it("reports the upload state when a not-yet-uploaded file is refused", async () => {
		const exec: CloudExec = async () => ({
			stdout:
				'{"ok":false,"isUploaded":false,"isUploading":true,"downloadingStatus":"NSURLUbiquitousItemDownloadingStatusCurrent"}\n',
			stderr: "",
		});

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/new.txt");

		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected a refusal");
		expect(result.message).toMatch(/still uploading/u);
	});

	it("reports a plain refusal with the download status", async () => {
		const exec: CloudExec = async () => ({
			stdout:
				'{"ok":false,"isUploaded":true,"isUploading":false,"downloadingStatus":"NSURLUbiquitousItemDownloadingStatusNotDownloaded"}\n',
			stderr: "",
		});

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/file.txt");

		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected a refusal");
		expect(result.message).toContain("NSURLUbiquitousItemDownloadingStatusNotDownloaded");
	});

	it("treats a segfaulting helper as a failed release, never a throw", async () => {
		const exec: CloudExec = async () => {
			const error = new Error("Command failed: osascript") as Error & { signal?: string };
			error.signal = "SIGSEGV";
			throw error;
		};

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/file.txt");

		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected a refusal");
		expect(result.message).toContain("SIGSEGV");
	});

	it("treats a timed-out helper as a failed release", async () => {
		const exec: CloudExec = async () => {
			const error = new Error("Command failed: osascript") as Error & { signal?: string; killed?: boolean };
			error.signal = "SIGTERM";
			error.killed = true;
			throw error;
		};

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/file.txt");

		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected a refusal");
		expect(result.message).toMatch(/killed/u);
	});

	it("fails safely when the helper prints something that is not a release result", async () => {
		const exec: CloudExec = async () => ({ stdout: "not a release result\n", stderr: "" });

		const result = await new DarwinCloudProbe({ exec }).evict("/tmp/cloud/file.txt");

		expect(result.ok).toBe(false);
		if (result.ok) throw new Error("expected a refusal");
		expect(result.message).toContain("not a release result");
	});

	it("invokes release for exactly the providers measured to work", () => {
		const probe = new DarwinCloudProbe({ exec: async () => ({ stdout: '{"ok":true}\n', stderr: "" }) });

		expect(probe.canRelease("icloud")).toBe(true);
		expect(probe.canRelease("google-drive")).toBe(true);
		expect(probe.canRelease("onedrive")).toBe(true);
		expect(probe.canRelease("dropbox")).toBe(true);
		expect(probe.canRelease("file-provider")).toBe(false);
		expect(probe.canRelease(undefined)).toBe(false);
	});
});

describe("windows dataless attribute parsing", () => {
	// Measured on Windows 11 (26200) with OneDrive: an online-only placeholder carries
	// FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS (0x400000); FILE_ATTRIBUTE_OFFLINE (0x1000) is NOT set on it.
	it("detects a Cloud Files placeholder from the decimal FileAttributes value", () => {
		expect(datalessFromWindowsAttributes(0x0040_0000)).toBe(true);
		expect(datalessFromWindowsAttributes(0x0040_0000 | 0x20)).toBe(true);
		expect(datalessFromWindowsAttributes(0x0040_0000 | 0x0008_0000)).toBe(true);
	});

	it("does not require FILE_ATTRIBUTE_OFFLINE, which real placeholders do not carry", () => {
		expect(datalessFromWindowsAttributes(0x0000_1000)).toBe(false);
	});

	it("treats an ordinary downloaded file as local", () => {
		expect(datalessFromWindowsAttributes(0x20)).toBe(false);
		expect(datalessFromWindowsAttributes(0)).toBe(false);
	});

	it("maps one attribute word per line back to the batch paths, dropping unreadable entries", () => {
		const paths = ["a.txt", "b.txt", "c.txt", "d.txt"];
		const parsed = parseWindowsAttributeOutput("32\r\n4194336\nERR\n", paths);

		expect([...parsed.entries()]).toEqual([
			["a.txt", 32],
			["b.txt", 4194336],
		]);
	});

	it("ignores lines past the end of the batch", () => {
		const parsed = parseWindowsAttributeOutput("32\n64\nextra\n", ["only.txt"]);
		expect([...parsed.entries()]).toEqual([["only.txt", 32]]);
	});
});

describe("windows sync root provider mapping", () => {
	it("parses the SyncRootManager dump into local roots with their display names", () => {
		const stdout = [
			`C:\\Users\\x\\OneDrive - Company\tOneDrive - Company`,
			`C:\\Users\\x\\Dropbox\tDropbox`,
			"",
			`no-separator-line`,
		].join("\r\n");

		expect(parseWindowsSyncRoots(stdout)).toEqual([
			{ path: String.raw`C:\Users\x\OneDrive - Company`, label: "OneDrive - Company" },
			{ path: String.raw`C:\Users\x\Dropbox`, label: "Dropbox" },
		]);
	});

	it("maps a sync root display name to a provider", () => {
		expect(providerFromWindowsSyncRoot("OneDrive - 경희대학교")).toBe("onedrive");
		expect(providerFromWindowsSyncRoot("Microsoft OneDrive")).toBe("onedrive");
		expect(providerFromWindowsSyncRoot("Google Drive")).toBe("google-drive");
		expect(providerFromWindowsSyncRoot("Dropbox")).toBe("file-provider");
		expect(providerFromWindowsSyncRoot("Box")).toBe("file-provider");
	});
});

describe("windows evict result mapping", () => {
	const ok = (stdout: string): PowerShellResult => ({ ok: true, stdout, stderr: "" });

	it("reports success when the placeholder was dehydrated", () => {
		expect(parseWindowsDehydrateOutput(ok("OPEN=0x00000000\nDEHYDRATE=0x00000000\n"))).toEqual({ ok: true });
	});

	it("surfaces the OS message when dehydration is refused", () => {
		const result = parseWindowsDehydrateOutput(
			ok("OPEN=0x00000000\nDEHYDRATE=0x80070178\nMSG=The file is not a cloud file\n"),
		);
		expect(result.ok).toBe(false);
		expect(result).toHaveProperty("message", "The file is not a cloud file");
	});

	it("fails when the file could not be opened for release", () => {
		const result = parseWindowsDehydrateOutput(ok("OPEN=0x80070005\nMSG=Access is denied\n"));
		expect(result.ok).toBe(false);
		expect(result).toHaveProperty("message", "Access is denied");
	});

	it("fails with the process stderr when the command produced no result", () => {
		const result = parseWindowsDehydrateOutput({ ok: false, stdout: "", stderr: "powershell.exe not found" });
		expect(result.ok).toBe(false);
		expect(result).toHaveProperty("message", "powershell.exe not found");
	});
});

/**
 * PowerShell accepts the ASCII apostrophe and the four typographic single quotes U+2018–U+201B as
 * single-quote delimiters. A path embedded in one of the scripts is a single-quoted literal, so every
 * one of those characters must be doubled or the literal ends early and the whole script fails to parse
 * (NomaDamas/AutoRAG-app#463). The same rule PowerShell's own
 * `CodeGeneration.EscapeSingleQuotedStringContent` applies.
 */
describe("windows PowerShell path quoting", () => {
	const SINGLE_QUOTES = ["'", "\u2018", "\u2019", "\u201A", "\u201B"] as const;
	const DOUBLE_QUOTES = ["\u201C", "\u201D"] as const;
	const SINGLE_QUOTE_CLASS = /['\u2018\u2019\u201A\u201B]/gu;
	const DOUBLED_SINGLE_QUOTE_CLASS = /('|\u2018|\u2019|\u201A|\u201B)\1/gu;

	/** The path literals of an attribute script: the `@(…)` array between `@(` and the `)){` that opens the loop. */
	function attributePathLiterals(script: string): string[] {
		const start = script.indexOf("@(");
		const end = script.indexOf(")){", start);
		expect(start).toBeGreaterThanOrEqual(0);
		expect(end).toBeGreaterThan(start);
		return script.slice(start + 2, end).split(",");
	}

	/**
	 * Asserts `chunk` is one PowerShell single-quoted literal for `value`: it opens and closes with `'`, every
	 * delimiter inside is doubled (so none can terminate it early), and it carries exactly `value`'s characters.
	 */
	function expectSingleQuotedLiteral(chunk: string | undefined, value: string): void {
		expect(chunk).toBeDefined();
		const literal = chunk as string;
		expect(literal.startsWith("'")).toBe(true);
		expect(literal.endsWith("'")).toBe(true);
		const inner = literal.slice(1, -1);
		expect(inner.replace(DOUBLED_SINGLE_QUOTE_CLASS, "")).not.toMatch(SINGLE_QUOTE_CLASS);
		expect(inner).toBe(value.replace(SINGLE_QUOTE_CLASS, (quote) => quote + quote));
	}

	it("doubles every single-quote delimiter an embedded path contains", () => {
		for (const quote of SINGLE_QUOTES) {
			const path = `C:\\Users\\x\\That${quote}s right.txt`;
			const [literal] = attributePathLiterals(buildWindowsAttributeScript([path]));
			expectSingleQuotedLiteral(literal, path);
		}
	});

	it("leaves double typographic quotes inside the literal unchanged", () => {
		for (const quote of DOUBLE_QUOTES) {
			const path = `C:\\Users\\x\\dq${quote}x.txt`;
			const [literal] = attributePathLiterals(buildWindowsAttributeScript([path]));
			expect(literal).toBe(`'${path}'`);
		}
	});

	it("produces one valid literal per path in a mixed batch", () => {
		const paths = [
			"C:\\Users\\x\\plain.txt",
			"C:\\Users\\x\\it's ascii.txt",
			"C:\\Users\\x\\That\u2019s right.txt",
			"C:\\Users\\x\\\u2018left\u2019.txt",
			"C:\\Users\\x\\low\u201Aquote.txt",
			"C:\\Users\\x\\rev\u201Bquote.txt",
			"C:\\Users\\x\\dq\u201Cx\u201D.txt",
		];
		const literals = attributePathLiterals(buildWindowsAttributeScript(paths));
		expect(literals).toHaveLength(paths.length);
		paths.forEach((path, index) => {
			expectSingleQuotedLiteral(literals[index], path);
		});
	});

	it("doubles the same delimiters in the dehydrate script's path literal", () => {
		for (const quote of [...SINGLE_QUOTES, ...DOUBLE_QUOTES]) {
			const path = `C:\\Users\\x\\That${quote}s right.txt`;
			const script = buildWindowsDehydrateScript(path);
			const marker = "[Autorag.CfApi]::CfOpenFileWithOplock(";
			const start = script.indexOf(marker) + marker.length;
			const literal = script.slice(start, script.indexOf(",0x1,[ref]$h)", start));
			expectSingleQuotedLiteral(literal, path);
		}
	});
});

/** Records every script it is handed and answers with a caller-supplied body. */
class FakePowerShell {
	readonly scripts: string[] = [];
	handler: (script: string, callIndex: number) => PowerShellResult = () => ({ ok: true, stdout: "", stderr: "" });
	readonly run = async (script: string): Promise<PowerShellResult> => {
		const index = this.scripts.length;
		this.scripts.push(script);
		return this.handler(script, index);
	};
}

describe("WindowsCloudProbe", () => {
	it("reads attributes in one process per batch", async () => {
		const powerShell = new FakePowerShell();
		const paths = Array.from({ length: WINDOWS_ATTRIBUTE_BATCH * 2 + 3 }, (_, index) => `file-${index}.txt`);
		powerShell.handler = (_script, callIndex) => {
			const batch = paths.slice(callIndex * WINDOWS_ATTRIBUTE_BATCH, (callIndex + 1) * WINDOWS_ATTRIBUTE_BATCH);
			return {
				ok: true,
				stdout: batch.map((path) => (path.endsWith("file-1.txt") ? "4194336" : "32")).join("\n"),
				stderr: "",
			};
		};

		const dataless = await new WindowsCloudProbe({ run: powerShell.run }).datalessAmong(paths);

		expect(powerShell.scripts).toHaveLength(3);
		expect([...dataless]).toEqual([paths[1]]);
	});

	it("resolves the sync root provider once and reuses it for later paths", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = (script) =>
			script.includes("SyncRootManager")
				? {
						ok: true,
						stdout: `${String.raw`C:\Users\x\Documents`}\tGoogle Drive\n${String.raw`C:\Users\x\Documents\OneDrive`}\tOneDrive\n`,
						stderr: "",
					}
				: { ok: true, stdout: "", stderr: "" };
		const probe = new WindowsCloudProbe({ run: powerShell.run });

		expect(await probe.providerOf(String.raw`C:\Users\x\Documents\OneDrive\a.txt`)).toBe("onedrive");
		expect(await probe.providerOf(String.raw`C:\Users\x\Documents\OneDrive\b.txt`)).toBe("onedrive");
		expect(await probe.providerOf(String.raw`C:\Users\x\Documents\report.txt`)).toBe("google-drive");
		expect(powerShell.scripts.filter((script) => script.includes("SyncRootManager"))).toHaveLength(1);
	});

	it("falls back to path markers when no sync root is registered", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: true, stdout: "", stderr: "" });
		const probe = new WindowsCloudProbe({ run: powerShell.run });

		expect(await probe.providerOf(String.raw`C:\Users\x\OneDrive - Company\a.txt`)).toBe("onedrive");
		expect(await probe.providerOf(String.raw`C:\Users\x\Documents\a.txt`)).toBe("file-provider");
	});

	it("releases every identified Cloud Files provider, because the platform enforces the refusals", () => {
		const probe = new WindowsCloudProbe({ run: new FakePowerShell().run });

		for (const provider of ["onedrive", "google-drive", "icloud", "dropbox", "file-provider"] as const)
			expect(probe.canRelease(provider)).toBe(true);
		expect(probe.canRelease(undefined)).toBe(false);
	});

	it("evicts through one PowerShell process and maps its result", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: true, stdout: "OPEN=0x00000000\nDEHYDRATE=0x00000000\n", stderr: "" });

		await expect(
			new WindowsCloudProbe({ run: powerShell.run }).evict(String.raw`C:\Users\x\OneDrive\a.txt`),
		).resolves.toEqual({
			ok: true,
		});
		expect(powerShell.scripts).toHaveLength(1);
		expect(powerShell.scripts[0]).toContain("CfDehydratePlaceholder");
	});

	it("parses the unprefixed hex the dehydrate script actually prints", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: true, stdout: "OPEN=00000000\nDEHYDRATE=00000000\n", stderr: "" });

		await expect(
			new WindowsCloudProbe({ run: powerShell.run }).evict(String.raw`C:\Users\x\OneDrive\a.txt`),
		).resolves.toEqual({ ok: true });
	});

	it("reports the refusal HRESULT when the provider will not dehydrate", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: true, stdout: "OPEN=00000000\nDEHYDRATE=8007018A\n", stderr: "" });

		await expect(
			new WindowsCloudProbe({ run: powerShell.run }).evict(String.raw`C:\Users\x\OneDrive\a.txt`),
		).resolves.toEqual({
			ok: false,
			message: "the provider refused to dehydrate the file (0x8007018A)",
		});
	});

	it("names the platform refusals instead of the generic OS text the script prints", async () => {
		const results = [];
		for (const hresult of ["80070179", "80070188"]) {
			const powerShell = new FakePowerShell();
			powerShell.handler = () => ({
				ok: true,
				stdout: `OPEN=00000000\nDEHYDRATE=${hresult}\nMSG=The cloud operation failed.\n`,
				stderr: "",
			});
			results.push(
				await new WindowsCloudProbe({ run: powerShell.run }).evict(String.raw`C:\Users\x\OneDrive\a.txt`),
			);
		}

		const [unsynced, pinned] = results;
		expect(unsynced).toMatchObject({ ok: false, message: expect.stringMatching(/not synced/) });
		expect(pinned).toMatchObject({ ok: false, message: expect.stringMatching(/pinned/) });
	});

	it("keeps the OS text for a refusal it does not know", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({
			ok: true,
			stdout: "OPEN=00000000\nDEHYDRATE=8007018A\nMSG=The cloud operation was canceled.\n",
			stderr: "",
		});

		await expect(
			new WindowsCloudProbe({ run: powerShell.run }).evict(String.raw`C:\Users\x\OneDrive\a.txt`),
		).resolves.toEqual({ ok: false, message: "The cloud operation was canceled." });
	});

	it("propagates a failed attribute read instead of silently treating placeholders as local", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: false, stdout: "", stderr: "powershell.exe is unavailable" });

		await expect(new WindowsCloudProbe({ run: powerShell.run }).datalessAmong(["a.txt"])).rejects.toThrow(
			/powershell\.exe is unavailable/u,
		);
	});
});

describe("defaultCloudProbe platform selection", () => {
	function withPlatform(platform: NodeJS.Platform, run: () => void): void {
		const original = Object.getOwnPropertyDescriptor(process, "platform");
		Object.defineProperty(process, "platform", { value: platform, configurable: true });
		try {
			run();
		} finally {
			if (original !== undefined) Object.defineProperty(process, "platform", original);
			else Reflect.deleteProperty(process, "platform");
		}
	}

	it("returns the Windows probe on win32 and the Darwin probe on darwin", () => {
		withPlatform("win32", () => expect(defaultCloudProbe()).toBeInstanceOf(WindowsCloudProbe));
		withPlatform("darwin", () => expect(defaultCloudProbe()).toBeInstanceOf(DarwinCloudProbe));
	});

	it("returns no probe on platforms AutoRAG does not handle yet", () => {
		withPlatform("linux", () => expect(defaultCloudProbe()).toBeUndefined());
	});
});

describe("cloud scope (pure)", () => {
	const roots: CloudRoot[] = [
		{ path: "/Users/x/Library/CloudStorage/GoogleDrive-user", kind: "file-provider" },
		{ path: "/Users/x/Library/CloudStorage/OneDrive-Work", kind: "file-provider" },
		{ path: "/Users/x/Library/Mobile Documents", kind: "file-provider" },
		{ path: "/Volumes/nas", kind: "network" },
	];

	it("matches equality and descendants, not sibling prefixes", () => {
		expect(isPathInside("/a/b", "/a/b")).toBe(true);
		expect(isPathInside("/a/b/c", "/a/b")).toBe(true);
		expect(isPathInside("/a/bc", "/a/b")).toBe(false);
		expect(isPathInside("/a", "/a/b")).toBe(false);
	});

	it("marks a search root inside a root of any kind, including the root itself", () => {
		expect(cloudScope("/Users/x/Library/CloudStorage/GoogleDrive-user/My Drive", roots).insideCloudRoot).toBe(true);
		expect(cloudScope("/Users/x/Library/Mobile Documents", roots).insideCloudRoot).toBe(true);
		expect(cloudScope("/Volumes/nas/shared", roots).insideCloudRoot).toBe(true);
		expect(cloudScope("/Users/x/Documents", roots).insideCloudRoot).toBe(false);
	});

	it("reports the roots nested under a search root, with their kind, and not the root itself", () => {
		const scope = cloudScope("/Users/x", roots);
		expect(scope.insideCloudRoot).toBe(false);
		expect(scope.nestedCloudRoots).toEqual([
			{ path: "/Users/x/Library/CloudStorage/GoogleDrive-user", kind: "file-provider" },
			{ path: "/Users/x/Library/CloudStorage/OneDrive-Work", kind: "file-provider" },
			{ path: "/Users/x/Library/Mobile Documents", kind: "file-provider" },
		]);
		expect(cloudScope("/Users/x/Library/CloudStorage/GoogleDrive-user", roots).nestedCloudRoots).toEqual([]);
		// A root on another volume is never nested under this search root.
		expect(cloudScope("/Users/x", roots).nestedCloudRoots.some((root) => root.kind === "network")).toBe(false);
	});

	it("normalizes Windows separators and trailing slashes", () => {
		expect(isPathInside(String.raw`C:\Users\x\OneDrive\a`, "C:/Users/x/OneDrive/")).toBe(true);
	});

	it("resolves a search root's own symlinks so a root reported at its real path still matches", () => {
		const base = realpathSync(mkdtempSync(join(tmpdir(), "autorag-cloudscope-")));
		const real = join(base, "real");
		const link = join(base, "link");
		mkdirSync(real);
		// A junction on Windows (no privilege needed) and an ordinary symlink elsewhere; the type is ignored on POSIX.
		symlinkSync(real, link, "junction");
		try {
			// A mount at the real path (macOS reports `/private/var/…` for a mount made at `/var/…`).
			expect(cloudScope(link, [{ path: real, kind: "network" }]).insideCloudRoot).toBe(true);
			expect(cloudScope(link, [{ path: link, kind: "network" }]).nestedCloudRoots).toEqual([]);
		} finally {
			rmSync(base, { recursive: true, force: true });
		}
	});
});

describe("canonical path mapping", () => {
	it("returns the path unchanged when no search root needs rewriting", () => {
		const map = canonicalPathMapper(["/nonexistent-autorag-root"]);
		expect(map("/nonexistent-autorag-root/a.txt")).toBe("/nonexistent-autorag-root/a.txt");
		expect(map("/elsewhere/a.txt")).toBe("/elsewhere/a.txt");
	});

	it("rewrites a walked path below the canonical search root", () => {
		const base = realpathSync(mkdtempSync(join(tmpdir(), "autorag-canonical-")));
		const real = join(base, "real");
		const link = join(base, "link");
		mkdirSync(real);
		// A junction on Windows (no privilege needed) and an ordinary symlink elsewhere; the type is ignored on POSIX.
		symlinkSync(real, link, "junction");
		try {
			expect(canonicalPath(link)).toBe(real);
			const map = canonicalPathMapper([link, join(base, "other")]);
			// A search root that is already canonical is not rewritten, so its path keeps the platform's separators;
			// a rewritten path comes back in the comparable (`/`-separated) form.
			expect(map(join(link, "a", "b.txt"))).toBe(slash(join(real, "a", "b.txt")));
			expect(map(join(base, "other", "c.txt"))).toBe(join(base, "other", "c.txt"));
			// A sibling path that merely shares a prefix is not rewritten.
			expect(map(`${link}-sibling/c.txt`)).toBe(`${link}-sibling/c.txt`);
		} finally {
			rmSync(base, { recursive: true, force: true });
		}
	});
});

/** Measured `/sbin/mount` shapes: local volumes carry `local`, `autofs` does not but is not a share. */
const DARWIN_MOUNT_OUTPUT = [
	"/dev/disk3s1s1 on / (apfs, sealed, local, read-only, journaled)",
	"devfs on /dev (devfs, local, nobrowse)",
	"map auto_home on /System/Volumes/Data/home (autofs, automounted, nobrowse)",
	"/dev/disk5s1 on /Volumes/SSD1 (apfs, local, nodev, nosuid, journaled, noowners)",
	"http://127.0.0.1:18765/ on /private/var/folders/tmp.X (webdav, nodev, noexec, nosuid, mounted by jeffrey)",
	"//server/share on /Volumes/My Share (smbfs, nodev, nosuid, mounted by jeffrey)",
	"nas:/export on /Volumes/NFS NAS (nfs, nodev, nosuid, mounted by jeffrey)",
].join("\n");

describe("darwin mount parsing", () => {
	it("parses device, mount point, filesystem, and options, keeping spaces in the mount point", () => {
		const mounts = parseDarwinMounts(DARWIN_MOUNT_OUTPUT);

		expect(mounts[0]).toEqual({
			device: "/dev/disk3s1s1",
			mountPoint: "/",
			fileSystem: "apfs",
			options: ["sealed", "local", "read-only", "journaled"],
		});
		expect(mounts.find((mount) => mount.fileSystem === "smbfs")).toEqual({
			device: "//server/share",
			mountPoint: "/Volumes/My Share",
			fileSystem: "smbfs",
			options: ["nodev", "nosuid", "mounted by jeffrey"],
		});
		expect(mounts.some((mount) => mount.mountPoint === "/Volumes/NFS NAS")).toBe(true);
	});

	it("ignores blank and malformed lines", () => {
		expect(parseDarwinMounts("\nnot a mount line\n(on /x (apfs, local))\n")).toEqual([]);
	});

	it("returns only the volumes that are not local storage", () => {
		expect(networkMountPoints(parseDarwinMounts(DARWIN_MOUNT_OUTPUT))).toEqual([
			"/private/var/folders/tmp.X",
			"/Volumes/My Share",
			"/Volumes/NFS NAS",
		]);
	});

	it("never treats autofs as a network share", () => {
		expect(
			networkMountPoints(parseDarwinMounts("map auto_home on /System/Volumes/Data/home (autofs, automounted)")),
		).toEqual([]);
	});
});

describe("xattr batch parsing", () => {
	it("maps each printed line to its path, even when the path itself contains a colon", () => {
		const candidates = ["/a/Documents", "/a/Documents: weird", "/a/plain", "/a/Desktop"];
		const stdout = [
			"/a/Documents: com.apple.CloudDocs.iCloudDriveFileProvider/ABC",
			"/a/Documents: weird: com.apple.CloudDocs.iCloudDriveFileProvider/ABC",
			"/a/Desktop: com.apple.CloudDocs.iCloudDriveFileProvider/ABC",
		].join("\n");

		expect([...parseXattrValues(stdout, candidates).entries()]).toEqual([
			["/a/Documents", "com.apple.CloudDocs.iCloudDriveFileProvider/ABC"],
			["/a/Documents: weird", "com.apple.CloudDocs.iCloudDriveFileProvider/ABC"],
			["/a/Desktop", "com.apple.CloudDocs.iCloudDriveFileProvider/ABC"],
		]);
	});

	it("returns nothing for empty output or no candidates", () => {
		expect(parseXattrValues("", ["/a"]).size).toBe(0);
		expect(parseXattrValues("/a: x", []).size).toBe(0);
	});
});

describe("windows disk list parsing", () => {
	const DISKS = [
		"C:\t3\tNTFS\tWindows",
		"Z:\t4\t\t",
		"M:\t3\tFUSE\t",
		"G:\t3\tGoogleDriveFS\tGoogle Drive",
		"E:\t3\tNTFS\tData",
	].join("\r\n");

	it("parses the tab-separated disk list and keeps volume names with spaces", () => {
		expect(parseWindowsLogicalDisks(DISKS)).toEqual([
			{ deviceId: "C:", driveType: 3, fileSystem: "NTFS", volumeName: "Windows" },
			{ deviceId: "Z:", driveType: 4, fileSystem: "", volumeName: "" },
			{ deviceId: "M:", driveType: 3, fileSystem: "FUSE", volumeName: "" },
			{ deviceId: "G:", driveType: 3, fileSystem: "GoogleDriveFS", volumeName: "Google Drive" },
			{ deviceId: "E:", driveType: 3, fileSystem: "NTFS", volumeName: "Data" },
		]);
		expect(parseWindowsLogicalDisks("C:\t3\tNTFS\tMy Data Disk")[0]).toMatchObject({
			volumeName: "My Data Disk",
		});
		expect(parseWindowsLogicalDisks("garbage\n")).toEqual([]);
	});

	it("names mapped network drives, FUSE volumes, and the unverified Google Drive virtual drive", () => {
		expect(windowsRootsFromDisks(parseWindowsLogicalDisks(DISKS))).toEqual([
			{ path: "Z:\\", kind: "network" },
			{ path: "M:\\", kind: "network" },
			{ path: "G:\\", kind: "cloud-files" },
		]);
		expect(WINDOWS_DRIVE_TYPE_NETWORK).toBe(4);
		expect(buildWindowsDiskScript()).toContain("Win32_LogicalDisk");
	});
});

describe("UNC share roots", () => {
	it("returns the share root of a UNC path", () => {
		expect(uncShareRoot(String.raw`\\server\share`)).toBe(String.raw`\\server\share`);
		expect(uncShareRoot(String.raw`\\server\share\docs\a.txt`)).toBe(String.raw`\\server\share`);
		expect(uncShareRoot("//localhost/D/docs")).toBe(String.raw`\\localhost\D`);
	});

	it("rejects local paths, bare servers, and device paths", () => {
		expect(uncShareRoot(String.raw`C:\Users\x`)).toBeUndefined();
		expect(uncShareRoot(String.raw`\\server`)).toBeUndefined();
		expect(uncShareRoot(String.raw`\\?\C:\Users\x`)).toBeUndefined();
		expect(uncShareRoot(String.raw`\\.\PhysicalDrive0`)).toBeUndefined();
		expect(uncShareRoot("/Volumes/nas/docs")).toBeUndefined();
	});
});

describe("cloud root listing", () => {
	/** A fake directory tree keyed by absolute path; every value is one entry. */
	function fakeTree(
		tree: Record<string, readonly CloudDirEntry[]>,
	): (path: string) => Promise<readonly CloudDirEntry[]> {
		return async (path: string) => {
			const entries = tree[path];
			if (entries === undefined) {
				const error: NodeJS.ErrnoException = new Error(`ENOENT: ${path}`);
				error.code = "ENOENT";
				throw error;
			}
			return entries;
		};
	}

	const dir = (name: string): CloudDirEntry => ({ name, isDirectory: () => true });
	const file = (name: string): CloudDirEntry => ({ name, isDirectory: () => false });

	/**
	 * A fake helper runner: `/sbin/mount` prints `mount`, `xattr -v -p <attr> <path>…` marks only the `managed`
	 * paths as carrying the attribute, and a helper named in `fail` rejects without partial output.
	 */
	function fakeExec(options: {
		readonly mount?: string;
		readonly managed?: readonly string[];
		readonly fail?: readonly string[];
	}): { readonly exec: CloudExec; readonly calls: readonly string[] } {
		const calls: string[] = [];
		const exec: CloudExec = async (helper, args) => {
			calls.push(helper);
			if (options.fail?.includes(helper) === true) throw new Error(`${helper} is unavailable`);
			if (helper === "/sbin/mount") return { stdout: options.mount ?? "", stderr: "" };
			if (helper === "/usr/bin/xattr") {
				const stdout = args
					.slice(3)
					.filter((path) => (options.managed ?? []).includes(path))
					.map((path) => `${path}: com.apple.CloudDocs.iCloudDriveFileProvider/ABC`)
					.join("\n");
				return { stdout: stdout === "" ? "" : `${stdout}\n`, stderr: "" };
			}
			throw new Error(`unexpected helper: ${helper}`);
		};
		return { exec, calls };
	}

	/** The probe's roots in the comparable form (`join` separates with `\` on Windows). */
	const machineRoot = (root: CloudRoot): string => `${root.kind}:${slash(root.path)}`;

	it("lists the File Provider mounts, iCloud Drive, and non-local mounts without reading file contents", async () => {
		const readDir = fakeTree({
			"/home/u/Library/CloudStorage": [dir("GoogleDrive-user"), dir("OneDrive-Work"), file("stray.txt")],
			"/home/u/Library/Mobile Documents": [dir("com~apple~CloudDocs")],
		});
		const { exec } = fakeExec({ mount: DARWIN_MOUNT_OUTPUT });

		const roots = await new DarwinCloudProbe({ readDir, exec, home: "/home/u" }).cloudRoots();

		expect(roots.map(machineRoot)).toEqual([
			"file-provider:/home/u/Library/CloudStorage/GoogleDrive-user",
			"file-provider:/home/u/Library/CloudStorage/OneDrive-Work",
			"file-provider:/home/u/Library/Mobile Documents",
			"network:/private/var/folders/tmp.X",
			"network:/Volumes/My Share",
			"network:/Volumes/NFS NAS",
		]);
	});

	it("finds a File Provider–managed search root from the xattr on itself or an ancestor", async () => {
		const readDir = fakeTree({ "/Users/testuser/Documents": [dir("sub"), file("note.txt")] });
		const { exec } = fakeExec({ managed: ["/Users/testuser/Documents"] });

		const direct = await new DarwinCloudProbe({ readDir, exec, home: "/Users/testuser" }).cloudRoots([
			"/Users/testuser/Documents",
		]);
		expect(direct).toEqual([{ path: "/Users/testuser/Documents", kind: "file-provider" }]);

		const nested = await new DarwinCloudProbe({ readDir, exec, home: "/Users/testuser" }).cloudRoots([
			"/Users/testuser/Documents/project",
		]);
		expect(nested).toEqual([{ path: "/Users/testuser/Documents", kind: "file-provider" }]);
	});

	it("finds a File Provider–managed directory among the search root's direct children", async () => {
		// The probe builds a child path with `join`, so the fixture and the expectation build it the same way.
		const searchRoot = join("/Users/testuser");
		const child = join(searchRoot, "Documents");
		const readDir = fakeTree({ [searchRoot]: [dir("Documents"), dir("Library")] });
		const { exec } = fakeExec({ managed: [child] });

		const roots = await new DarwinCloudProbe({ readDir, exec, home: searchRoot }).cloudRoots([searchRoot]);

		expect(roots).toEqual([{ path: child, kind: "file-provider" }]);
	});

	it("reads the mount table and the xattr in one process each", async () => {
		const readDir = fakeTree({ "/Users/testuser/Documents": [] });
		const { exec, calls } = fakeExec({});

		await new DarwinCloudProbe({ readDir, exec, home: "/Users/testuser" }).cloudRoots(["/Users/testuser/Documents"]);

		expect(calls).toEqual(["/sbin/mount", "/usr/bin/xattr"]);
	});

	it("batches the xattr query over a search root with many entries", async () => {
		const children = Array.from({ length: XATTR_BATCH + 50 }, (_, index) => dir(`child-${index}`));
		const readDir = fakeTree({ "/Users/testuser/Documents": children });
		const { exec, calls } = fakeExec({ managed: ["/Users/testuser/Documents"] });

		const roots = await new DarwinCloudProbe({ readDir, exec, home: "/Users/testuser" }).cloudRoots([
			"/Users/testuser/Documents",
		]);

		// The root, its ancestors and its direct children: two batches for more than XATTR_BATCH candidates.
		expect(calls.filter((helper) => helper === "/usr/bin/xattr")).toHaveLength(2);
		expect(roots).toEqual([{ path: "/Users/testuser/Documents", kind: "file-provider" }]);
	});

	it("returns no extra roots when a helper fails, never failing the refresh", async () => {
		const readDir = fakeTree({});
		const { exec } = fakeExec({ fail: ["/sbin/mount", "/usr/bin/xattr"] });

		const roots = await new DarwinCloudProbe({ readDir, exec, home: "/Users/testuser" }).cloudRoots([
			"/Users/testuser/Documents",
		]);

		expect(roots).toEqual([]);
	});

	it("returns an empty list when neither cloud folder exists", async () => {
		const { exec } = fakeExec({});
		const roots = await new DarwinCloudProbe({ readDir: fakeTree({}), exec, home: "/home/u" }).cloudRoots([]);
		expect(roots).toEqual([]);
	});

	it("lists Windows sync roots, network drives, FUSE volumes, and the UNC search root", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = (script) => {
			if (script.includes("SyncRootManager")) {
				return {
					ok: true,
					stdout: `${String.raw`C:\Users\x\OneDrive - Work`}\tOneDrive - Work\n${String.raw`C:\Users\x\Dropbox`}\tDropbox\n`,
					stderr: "",
				};
			}
			if (script.includes("Win32_LogicalDisk")) {
				return { ok: true, stdout: "C:\t3\tNTFS\tWindows\nZ:\t4\t\t\nM:\t3\tFUSE\t\n", stderr: "" };
			}
			return { ok: true, stdout: "", stderr: "" };
		};

		const roots = await new WindowsCloudProbe({ run: powerShell.run }).cloudRoots([String.raw`\\server\share\docs`]);

		expect(roots).toEqual([
			{ path: String.raw`C:\Users\x\OneDrive - Work`, kind: "cloud-files" },
			{ path: String.raw`C:\Users\x\Dropbox`, kind: "cloud-files" },
			{ path: "Z:\\", kind: "network" },
			{ path: "M:\\", kind: "network" },
			{ path: String.raw`\\server\share`, kind: "network" },
		]);
	});

	it("returns no Windows roots when the disk list cannot be read", async () => {
		const powerShell = new FakePowerShell();
		powerShell.handler = () => ({ ok: false, stdout: "", stderr: "powershell.exe is unavailable" });

		await expect(
			new WindowsCloudProbe({ run: powerShell.run }).cloudRoots([String.raw`\\server\share\docs`]),
		).resolves.toEqual([{ path: String.raw`\\server\share`, kind: "network" }]);
	});
});
