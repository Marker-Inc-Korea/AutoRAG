import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { parse } from "yaml";
import { SupplyChainError } from "../../scripts/supply-chain/policy.ts";
import { stageReleaseAssets } from "../../scripts/supply-chain/stage-release-assets.ts";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");
const fixtureRoots: string[] = [];

afterEach(() => {
	for (const root of fixtureRoots.splice(0)) {
		rmSync(root, { recursive: true, force: true });
	}
});

const TAR_ARCHIVE = Uint8Array.from(
	Buffer.from(
		"H4sIAGQXqWoC/+3NsQqDQBCE4a19ioi13K4YIY9zgqBwWtwZnz8XG8FahJD/a2Znmg3Bz77uTV9Pc2EvaYjbEOU6mnVtu2d2TlVrjvu7W2OdykPlBu+0+pjfy3+qStdPi0tjIQAAAAAAAAAAAAAAAACAH/IBLt23OQAoAAA=",
		"base64",
	),
);
const ZIP_ARCHIVE = Uint8Array.from(
	Buffer.from(
		"UEsDBBQAAAAAACSYL11dm7CPAgAAAAIAAAAQAAAAbGxhbWEtc2VydmVyLmV4ZU1aUEsBAhQDFAAAAAAAJJgvXV2bsI8CAAAAAgAAABAAAAAAAAAAAAAAAIABAAAAAGxsYW1hLXNlcnZlci5leGVQSwUGAAAAAAEAAQA+AAAAMAAAAAAA",
		"base64",
	),
);

function writeFixture(opts: {
	readonly withDist: boolean;
	readonly withSbom: boolean;
	readonly withEmbedding?: boolean;
}): {
	readonly root: string;
	readonly outputDir: string;
	readonly sbomDir: string;
} {
	const root = mkdtempSync(join(tmpdir(), "autorag-release-assets-"));
	fixtureRoots.push(root);
	writeFileSync(
		join(root, "package.json"),
		JSON.stringify({ name: "fixture-pkg", version: "1.2.3", files: ["dist"] }, null, 2),
	);
	writeFileSync(join(root, "LICENSE"), "MIT License\n");
	writeFileSync(join(root, "NOTICE"), "NOTICE\n");
	writeFileSync(join(root, "GOVERNANCE.md"), "# Governance\n");
	if (opts.withDist) {
		mkdirSync(join(root, "dist"));
		writeFileSync(join(root, "dist", "index.js"), "export const ok = true;\n");
	}
	if (opts.withEmbedding) {
		const licensesDir = join(root, "licenses");
		mkdirSync(licensesDir);
		writeFileSync(join(licensesDir, "llama.cpp-MIT.txt"), "MIT\n");
		const assets = [
			{
				kind: "runtime",
				id: "mac",
				filename: "runtime.tar.gz",
				url: "https://example.invalid/runtime.tar.gz",
				revision: "test",
				bytes: TAR_ARCHIVE,
				archiveMembers: ["llama-b10951/llama-server"],
			},
			{
				kind: "runtime",
				id: "win-cpu",
				filename: "runtime-cpu.zip",
				url: "https://example.invalid/runtime-cpu.zip",
				revision: "test",
				bytes: ZIP_ARCHIVE,
				archiveMembers: ["llama-server.exe"],
			},
			{
				kind: "runtime",
				id: "win-vulkan",
				filename: "runtime-vulkan.zip",
				url: "https://example.invalid/runtime-vulkan.zip",
				revision: "test",
				bytes: ZIP_ARCHIVE,
				archiveMembers: ["llama-server.exe"],
			},
		] as const;
		writeFileSync(
			join(licensesDir, "embedding-assets.json"),
			JSON.stringify({
				assets: assets.map(({ bytes, ...asset }) => ({
					...asset,
					sha256: createHash("sha256").update(bytes).digest("hex"),
					licenseId: "MIT",
					noticeFile: "licenses/llama.cpp-MIT.txt",
				})),
			}),
		);
	}
	const sbomDir = join(root, "sboms");
	mkdirSync(sbomDir);
	if (opts.withSbom) {
		writeFileSync(join(sbomDir, "autorag.cdx.json"), JSON.stringify({ bomFormat: "CycloneDX" }));
		writeFileSync(join(sbomDir, "notes.json"), JSON.stringify({ not: "sbom" }));
	}
	return { root, outputDir: join(root, "release-assets"), sbomDir };
}

describe("stageReleaseAssets", () => {
	it("stages the npm pack tarball, LICENSE, NOTICE, GOVERNANCE, SBOM, and SHA256SUMS", () => {
		const fixture = writeFixture({ withDist: true, withSbom: true, withEmbedding: true });
		const staged = stageReleaseAssets({
			projectRoot: fixture.root,
			outputDir: fixture.outputDir,
			sbomDir: fixture.sbomDir,
			downloadAsset: (asset) => (asset.id === "mac" ? TAR_ARCHIVE : ZIP_ARCHIVE),
		});
		expect(staged.files).toContain("licenses/llama.cpp-MIT.txt");
		expect(staged.files).toContain("runtime.tar.gz");
		expect(staged.files).toContain("runtime-cpu.zip");
		expect(staged.files).toContain("runtime-vulkan.zip");
		expect(staged.files).toContain("LICENSE");
		expect(staged.files).toContain("NOTICE");
		expect(staged.files).toContain("GOVERNANCE.md");
		expect(staged.files).toContain("autorag.cdx.json");
		expect(staged.files).not.toContain("notes.json");
		expect(staged.files).toContain("SHA256SUMS.txt");
		expect(staged.files.some((name) => name.endsWith(".tgz"))).toBe(true);
		expect(staged.files.some((name) => name.includes("node_modules"))).toBe(false);

		const sums = readFileSync(staged.checksumsPath, "utf8");
		const licenseLine = sums.split("\n").find((line) => line.endsWith("  LICENSE"));
		expect(licenseLine).toBeTruthy();
		const digest = licenseLine?.slice(0, 64);
		expect(digest).toMatch(/^[0-9a-f]{64}$/);
	});

	it("fails closed when dist/index.js is missing", () => {
		const fixture = writeFixture({ withDist: false, withSbom: false });
		expect(() =>
			stageReleaseAssets({
				projectRoot: fixture.root,
				outputDir: fixture.outputDir,
			}),
		).toThrow(SupplyChainError);
	});
});

describe("GitHub release asset contract", () => {
	it("attaches staged release-assets and keeps provenance publish plus supply-chain needs", () => {
		const text = readFileSync(join(repoRoot, ".github/workflows/release.yml"), "utf8");
		expect(text).toContain("stage-release-assets.ts");
		expect(text).toContain("release-assets/");
		expect(text).toContain("npm publish --provenance");
		const attached = text.split("files:")[1] ?? "";
		expect(attached).toContain("release-assets/");
		expect(attached).not.toMatch(/node_modules/);
		const doc = parse(text) as {
			readonly jobs: { readonly publish: { readonly needs?: readonly string[] } };
		};
		expect(doc.jobs.publish.needs).toContain("supply-chain");
	});
});
