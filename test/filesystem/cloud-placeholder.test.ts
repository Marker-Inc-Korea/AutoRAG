import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	classifyFilesystemRoot,
	homeCloudRoots,
	pathLooksLikeCloudRoot,
} from "../../src/filesystem/cloud-placeholder.ts";

describe("cloud placeholder detection", () => {
	it("treats CloudStorage and Mobile Documents paths as file-provider roots", () => {
		expect(pathLooksLikeCloudRoot("/Users/x/Library/CloudStorage/OneDrive-Personal/docs")).toBe(true);
		expect(pathLooksLikeCloudRoot("/Users/x/Library/Mobile Documents/com~apple~CloudDocs")).toBe(true);
		expect(pathLooksLikeCloudRoot("/Users/x/Downloads")).toBe(false);
		expect(pathLooksLikeCloudRoot("/Users/x/PycharmProjects/banana/paper")).toBe(false);
	});

	it("lists well-known home cloud roots", () => {
		const roots = homeCloudRoots("/Users/demo");
		expect(roots).toContain("/Users/demo/Library/CloudStorage");
		expect(roots).toContain("/Users/demo/Library/Mobile Documents");
	});

	it("classifies a local temp directory as local", async () => {
		const dir = mkdtempSync(join(tmpdir(), "autorag-cloud-"));
		mkdirSync(join(dir, "docs"));
		writeFileSync(join(dir, "docs", "a.txt"), "hello\n");
		const classified = await classifyFilesystemRoot(dir);
		expect(classified.kind).toBe("local");
		expect(classified.reason).toBe("local");
	});

	it("classifies CloudStorage paths without touching the filesystem", async () => {
		const classified = await classifyFilesystemRoot("/Users/demo/Library/CloudStorage/GoogleDrive-x/My Drive");
		expect(classified.kind).toBe("file-provider");
		expect(classified.reason).toBe("path-marker");
	});

	it("classifies this machine's Documents as file-provider when the xattr is present", async () => {
		if (process.platform !== "darwin") return;
		const classified = await classifyFilesystemRoot(`${process.env.HOME ?? ""}/Documents`);
		if (classified.kind === "local") return;
		expect(classified.reason).toBe("file-provider-xattr");
	});
});
