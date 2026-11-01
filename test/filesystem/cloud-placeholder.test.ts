import { describe, expect, it } from "vitest";
import {
	type CloudExec,
	cloudProviderFromDomain,
	cloudProviderFromPath,
	DarwinCloudProbe,
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
