#!/usr/bin/env bun
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { collectPackageLicenses, generateCycloneDx, generateNotice, isJsonObject } from "./inventory.ts";
import {
	AUTORAG_SUPPLY_CHAIN_POLICY,
	type CveFinding,
	type CveSeverity,
	evaluateCves,
	evaluateLicenses,
	evaluateReleaseGate,
	SupplyChainError,
} from "./policy.ts";

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const NOTICE_PATH = join(repoRoot, "NOTICE");
const NOTICE_META = {
	projectName: "AutoRAG",
	year: "2025",
	holder: "NomaDamas / Marker Inc.",
	rootLicense: "MIT",
	legacyLicense: "Apache-2.0",
} as const;

function main(argv: readonly string[]): number {
	const command = argv[0] && !argv[0].startsWith("-") ? argv[0] : "gate";
	const flags = parseFlags(argv[0] && !argv[0].startsWith("-") ? argv.slice(1) : argv);
	const components = collectPackageLicenses(repoRoot);
	const notice = generateNotice({ ...NOTICE_META, components });
	if (command === "notice") {
		writeFileSync(NOTICE_PATH, notice);
		return 0;
	}
	const manifest: unknown = JSON.parse(readFileSync(join(repoRoot, "package.json"), "utf8"));
	if (!isJsonObject(manifest) || typeof manifest.name !== "string" || typeof manifest.version !== "string") {
		throw new SupplyChainError("package.json must declare name and version");
	}
	const bom = generateCycloneDx({ name: manifest.name, version: manifest.version, components });
	if (command === "sbom" || flags.sbom !== undefined) {
		const sbomPath = flags.sbom ?? join(repoRoot, "sbom.cdx.json");
		writeFileSync(sbomPath, `${JSON.stringify(bom, null, 2)}\n`);
	}
	if (command === "sbom") return 0;
	if (command !== "gate") throw new SupplyChainError(`unknown command ${command}`);

	const licenses = evaluateLicenses(components, AUTORAG_SUPPLY_CHAIN_POLICY);
	let noticeOk = false;
	try {
		noticeOk = readFileSync(NOTICE_PATH, "utf8") === notice;
	} catch (error) {
		if (!(error instanceof Error) || !("code" in error) || error.code !== "ENOENT") throw error;
	}
	const cves =
		flags.cves === undefined
			? { ok: true, blocking: [] }
			: evaluateCves(readCves(flags.cves), AUTORAG_SUPPLY_CHAIN_POLICY);
	const gate = evaluateReleaseGate({
		sbomOk: true,
		licenseOk: licenses.ok,
		cveOk: cves.ok,
		noticeOk,
	});
	const summary = {
		ok: gate.ok,
		missing: gate.missing,
		denials: licenses.denials.map((denial) => `${denial.component.name}: ${denial.reason}`),
		cves: cves.blocking.map((finding) => `${finding.id}:${finding.severity}`),
	};
	process.stdout.write(`${JSON.stringify(summary, null, 2)}\n`);
	if (!gate.ok) {
		process.stderr.write(
			"supply-chain gate failed; run `bun scripts/supply-chain/evaluate.ts notice` if NOTICE is stale\n",
		);
		return 1;
	}
	return 0;
}

function parseFlags(argv: readonly string[]): { readonly sbom?: string; readonly cves?: string } {
	let sbom: string | undefined;
	let cves: string | undefined;
	for (let index = 0; index < argv.length; index += 1) {
		const flag = argv[index];
		const value = argv[index + 1];
		if (flag === "--sbom" && value !== undefined) {
			sbom = value;
			index += 1;
			continue;
		}
		if (flag === "--cves" && value !== undefined) {
			cves = value;
			index += 1;
			continue;
		}
		throw new SupplyChainError(`unknown argument ${flag}`);
	}
	return { sbom, cves };
}

function readCves(path: string): CveFinding[] {
	const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
	if (!Array.isArray(parsed)) throw new SupplyChainError("CVE file must be a JSON array");
	return parsed.map((item, index) => {
		if (!isJsonObject(item) || typeof item.id !== "string" || typeof item.packageName !== "string") {
			throw new SupplyChainError(`CVE entry ${index} is missing id or packageName`);
		}
		return { id: item.id, packageName: item.packageName, severity: parseSeverity(item.severity) };
	});
}

function parseSeverity(value: unknown): CveSeverity {
	if (value === "critical" || value === "high" || value === "moderate" || value === "low" || value === "unknown") {
		return value;
	}
	if (value === "medium") return "moderate";
	return "unknown";
}

try {
	process.exitCode = main(process.argv.slice(2));
} catch (error) {
	const message = error instanceof Error ? error.message : String(error);
	process.stderr.write(`${message}\n`);
	process.exitCode = 1;
}
