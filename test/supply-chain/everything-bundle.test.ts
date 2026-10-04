import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import {
	ensureEverythingBinaries,
	loadEverythingBundleManifest,
	resolveEverythingBundleDir,
} from "../../src/everything/index.ts";

const repoRoot = join(import.meta.dirname, "..", "..");
const bundleDir = join(repoRoot, "vendor", "everything");

describe("bundled Everything distribution", () => {
	const manifest = loadEverythingBundleManifest(bundleDir);

	it("ships in the npm package together with its licenses", () => {
		const pkg = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8")) as { files: string[] };
		expect(pkg.files).toEqual(expect.arrayContaining(["vendor", "licenses", "NOTICE"]));
		expect(resolveEverythingBundleDir()).toBe(bundleDir);
	});

	it("pins every vendored archive by SHA-256", () => {
		for (const assets of Object.values(manifest.architectures)) {
			for (const [archive, digest] of [
				[assets!.everythingArchive, assets!.everythingArchiveSha256],
				[assets!.esArchive, assets!.esArchiveSha256],
			] as const) {
				const bytes = readFileSync(join(bundleDir, archive));
				expect(createHash("sha256").update(bytes).digest("hex"), archive).toBe(digest);
			}
		}
	});

	it("includes the MIT license text for Everything and ES and references them from NOTICE", () => {
		const notice = readFileSync(join(repoRoot, "NOTICE"), "utf8");
		expect(manifest.licenseFiles.length).toBeGreaterThan(0);
		for (const file of manifest.licenseFiles) {
			const text = readFileSync(join(repoRoot, file), "utf8");
			expect(text, file).toContain("Permission is hereby granted, free of charge");
			expect(notice, file).toContain(file);
		}
		expect(notice).toContain(`Everything ${manifest.everythingVersion}`);
		expect(notice).toContain(`ES ${manifest.esVersion}`);
	});

	it("extracts verified binaries from the real bundle for every bundled architecture", async () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-everything-bundle-"));
		try {
			for (const arch of Object.keys(manifest.architectures)) {
				const result = await ensureEverythingBinaries({ root, platform: "win32", arch });
				expect(result, arch).toMatchObject({ ok: true, source: "installed" });
				if (result.ok) expect(existsSync(result.esPath)).toBe(true);
			}
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	});
});
