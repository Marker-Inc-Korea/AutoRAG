import { describe, expect, it } from "vitest";
import {
	DarwinCloudProbe,
	datalessFromDarwinFlags,
	datalessFromWindowsAttributes,
	defaultCloudProbe,
	homeCloudRoots,
	type PowerShellResult,
	parseWindowsAttributeOutput,
	parseWindowsDehydrateOutput,
	parseWindowsSyncRoots,
	pathLooksLikeCloudRoot,
	providerFromWindowsSyncRoot,
	WINDOWS_ATTRIBUTE_BATCH,
	WindowsCloudProbe,
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

	it("releases only the providers verified live on Windows", () => {
		const probe = new WindowsCloudProbe({ run: new FakePowerShell().run });

		expect(probe.canRelease("onedrive")).toBe(true);
		expect(probe.canRelease("google-drive")).toBe(false);
		expect(probe.canRelease("icloud")).toBe(false);
		expect(probe.canRelease("file-provider")).toBe(false);
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
