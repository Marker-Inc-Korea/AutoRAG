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

function writeFixture(opts: { readonly withDist: boolean; readonly withSbom: boolean }): {
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
		const fixture = writeFixture({ withDist: true, withSbom: true });
		const staged = stageReleaseAssets({
			projectRoot: fixture.root,
			outputDir: fixture.outputDir,
			sbomDir: fixture.sbomDir,
		});
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
