import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { collectPackageLicenses } from "../../scripts/supply-chain/inventory.ts";
import {
	AUTORAG_SUPPLY_CHAIN_POLICY,
	evaluateCves,
	evaluateLicenses,
	evaluateSpdxExpression,
	severityAtOrAbove,
} from "../../scripts/supply-chain/policy.ts";

const fixtureRoots: string[] = [];

afterEach(() => {
	for (const root of fixtureRoots.splice(0)) {
		rmSync(root, { recursive: true, force: true });
	}
});

function writePackageTree(spec: {
	readonly dependencies: Record<string, string>;
	readonly packages: Record<string, { readonly version: string; readonly license: string }>;
}): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-supply-chain-"));
	fixtureRoots.push(root);
	writeFileSync(
		join(root, "package.json"),
		JSON.stringify({ name: "fixture", dependencies: spec.dependencies }, null, 2),
	);
	for (const [name, meta] of Object.entries(spec.packages)) {
		const dir = join(root, "node_modules", name);
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "package.json"), JSON.stringify({ name, ...meta }, null, 2));
	}
	return root;
}

describe("evaluateSpdxExpression against AutoRAG MIT + Apache-2.0 governance", () => {
	it("allows MIT when the expression is a single permitted identifier", () => {
		expect(evaluateSpdxExpression("MIT", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(true);
	});

	it("denies GPL-3.0-only because copyleft is outside the allowlist", () => {
		expect(evaluateSpdxExpression("GPL-3.0-only", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(false);
	});

	it("allows jszip-style MIT OR GPL-3.0-or-later because one alternative is permitted", () => {
		expect(evaluateSpdxExpression("MIT OR GPL-3.0-or-later", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(true);
	});

	it("denies MIT AND GPL-3.0-only because conjunctive copyleft is not distributable", () => {
		expect(evaluateSpdxExpression("MIT AND GPL-3.0-only", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(false);
	});

	it("denies an empty or unknown license fail-closed", () => {
		expect(evaluateSpdxExpression("", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(false);
		expect(evaluateSpdxExpression("LicenseRef-clearlydefined-OTHER", AUTORAG_SUPPLY_CHAIN_POLICY)).toBe(false);
	});
});

describe("evaluateLicenses", () => {
	it("allows the current production license set including disjunctive SPDX", () => {
		const result = evaluateLicenses(
			[
				{ name: "yaml", version: "2.9.0", license: "ISC" },
				{ name: "jszip", version: "3.10.1", license: "(MIT OR GPL-3.0-or-later)" },
				{ name: "xlsx", version: "0.18.5", license: "Apache-2.0" },
			],
			AUTORAG_SUPPLY_CHAIN_POLICY,
		);
		expect(result.ok).toBe(true);
		expect(result.denials).toEqual([]);
	});

	it("denies a GPL-3.0 production dependency", () => {
		const result = evaluateLicenses(
			[{ name: "evil", version: "1.0.0", license: "GPL-3.0-only" }],
			AUTORAG_SUPPLY_CHAIN_POLICY,
		);
		expect(result.ok).toBe(false);
		expect(result.denials.map((denial) => denial.component.name)).toEqual(["evil"]);
	});

	it("collects licenses from package.json production dependencies", () => {
		const root = writePackageTree({
			dependencies: { allowed: "1.0.0", copyleft: "2.0.0" },
			packages: {
				allowed: { version: "1.0.0", license: "MIT" },
				copyleft: { version: "2.0.0", license: "AGPL-3.0-only" },
			},
		});
		const components = collectPackageLicenses(root);
		const result = evaluateLicenses(components, AUTORAG_SUPPLY_CHAIN_POLICY);
		expect(components.map((component) => component.name).sort()).toEqual(["allowed", "copyleft"]);
		expect(result.ok).toBe(false);
		expect(result.denials.map((denial) => denial.component.name)).toEqual(["copyleft"]);
	});
});

describe("evaluateCves", () => {
	it("passes moderate findings when the gate is high", () => {
		const result = evaluateCves(
			[{ id: "GHSA-moderate", severity: "moderate", packageName: "demo" }],
			AUTORAG_SUPPLY_CHAIN_POLICY,
		);
		expect(result.ok).toBe(true);
		expect(result.blocking).toEqual([]);
	});

	it("fails closed on high and critical findings", () => {
		expect(severityAtOrAbove("high", "high")).toBe(true);
		const result = evaluateCves(
			[
				{ id: "GHSA-high", severity: "high", packageName: "demo" },
				{ id: "CVE-2024-0001", severity: "critical", packageName: "demo" },
			],
			AUTORAG_SUPPLY_CHAIN_POLICY,
		);
		expect(result.ok).toBe(false);
		expect(result.blocking.map((finding) => finding.id).sort()).toEqual(["CVE-2024-0001", "GHSA-high"]);
	});
});
