import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { parse } from "yaml";
import { generateCycloneDx, generateNotice } from "../../scripts/supply-chain/inventory.ts";
import { AUTORAG_SUPPLY_CHAIN_POLICY, evaluateReleaseGate } from "../../scripts/supply-chain/policy.ts";

const repoRoot = join(dirname(fileURLToPath(import.meta.url)), "../..");

describe("evaluateReleaseGate", () => {
	it("passes only when SBOM, license, CVE, and NOTICE gates all passed", () => {
		expect(
			evaluateReleaseGate({
				sbomOk: true,
				licenseOk: true,
				cveOk: true,
				noticeOk: true,
			}),
		).toEqual({ ok: true, missing: [] });
	});

	it("fails closed when any required artifact or policy result is missing", () => {
		const result = evaluateReleaseGate({
			sbomOk: false,
			licenseOk: true,
			cveOk: false,
			noticeOk: true,
		});
		expect(result.ok).toBe(false);
		expect(result.missing).toEqual(["sbom", "cve"]);
	});
});

describe("NOTICE and CycloneDX tokens", () => {
	const components = [
		{ name: "jszip", version: "3.10.1", license: "(MIT OR GPL-3.0-or-later)" },
		{ name: "xlsx", version: "0.18.5", license: "Apache-2.0" },
	] as const;

	it("emits dual-license SPDX ids and production package names into NOTICE", () => {
		const notice = generateNotice({
			projectName: "AutoRAG",
			year: "2025",
			holder: "NomaDamas / Marker Inc.",
			rootLicense: "MIT",
			legacyLicense: "Apache-2.0",
			components,
		});
		expect(notice).toContain("MIT");
		expect(notice).toContain("Apache-2.0");
		expect(notice).toContain("jszip@3.10.1");
		expect(notice).toContain("xlsx@0.18.5");
	});

	it("emits CycloneDX JSON with named components", () => {
		const bom = generateCycloneDx({
			name: "@autorag/librarian",
			version: "2.4.2",
			components,
		});
		expect(bom.bomFormat).toBe("CycloneDX");
		expect(bom.components.map((component) => component.name)).toEqual(["jszip", "xlsx"]);
		expect(bom.components[0]?.licenses).toEqual([{ expression: "(MIT OR GPL-3.0-or-later)" }]);
	});
});

describe("GitHub workflow contracts", () => {
	it("requires the supply-chain job before npm publish", () => {
		const doc = parse(readFileSync(join(repoRoot, ".github/workflows/release.yml"), "utf8")) as {
			readonly jobs: { readonly publish: { readonly needs?: readonly string[] } };
		};
		expect(doc.jobs.publish.needs).toContain("supply-chain");
	});

	it("wires free GitHub-provided or well-known OSS scanners and no paid SCA vendors", () => {
		const text = readFileSync(join(repoRoot, ".github/workflows/supply-chain.yml"), "utf8");
		expect(text).toContain("actions/dependency-review-action@");
		expect(text).toContain("anchore/sbom-action@");
		expect(text).toContain("google/osv-scanner-action");
		expect(text).toContain("actions/attest-sbom@");
		expect(text).not.toMatch(/snyk|fossa|mend\/|blackduck|sonatype/i);
		expect(text).toContain("workflow_call:");
		const doc = parse(text) as {
			readonly jobs?: { readonly "license-notice-gate"?: unknown };
		};
		expect(doc.jobs?.["license-notice-gate"]).toBeTruthy();
	});

	it("keeps dependency-review allow-licenses aligned with the AutoRAG policy", () => {
		const doc = parse(readFileSync(join(repoRoot, ".github/dependency-review-config.yml"), "utf8")) as {
			readonly "fail-on-severity": string;
			readonly "allow-licenses": readonly string[];
		};
		expect(doc["fail-on-severity"]).toBe("high");
		expect([...doc["allow-licenses"]].sort()).toEqual([...AUTORAG_SUPPLY_CHAIN_POLICY.allowLicenses].sort());
	});
});
