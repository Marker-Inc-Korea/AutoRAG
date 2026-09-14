import { readFileSync } from "node:fs";
import { join } from "node:path";
import { type ComponentLicense, SupplyChainError } from "./policy.ts";

export function isJsonObject(value: unknown): value is Record<string, unknown> {
	return value !== null && typeof value === "object" && !Array.isArray(value);
}

export function collectPackageLicenses(projectRoot: string): ComponentLicense[] {
	const manifest = readJsonObject(join(projectRoot, "package.json"));
	const dependencies = recordOfStrings(manifest.dependencies);
	return Object.keys(dependencies)
		.sort()
		.map((name) => {
			const pkg = readJsonObject(join(projectRoot, "node_modules", ...name.split("/"), "package.json"));
			return {
				name,
				version: typeof pkg.version === "string" ? pkg.version : "",
				license: licenseFromPackageJson(pkg),
			};
		});
}

export function generateNotice(input: {
	readonly projectName: string;
	readonly year: string;
	readonly holder: string;
	readonly rootLicense: string;
	readonly legacyLicense: string;
	readonly components: readonly ComponentLicense[];
}): string {
	const thirdParty = [...input.components]
		.sort((left, right) => left.name.localeCompare(right.name))
		.map((component) => `${component.name}@${component.version}\nLicense: ${component.license}`)
		.join("\n\n");
	return [
		`${input.projectName}`,
		`Copyright (c) ${input.year} ${input.holder}`,
		"",
		`Root package license: ${input.rootLicense}`,
		`legacy/ package license: ${input.legacyLicense}`,
		"",
		"Third-party production dependencies:",
		"",
		thirdParty,
		"",
	].join("\n");
}

export type CycloneDxBom = {
	readonly bomFormat: "CycloneDX";
	readonly specVersion: string;
	readonly version: number;
	readonly metadata: {
		readonly component: { readonly type: "library"; readonly name: string; readonly version: string };
	};
	readonly components: readonly {
		readonly type: "library";
		readonly name: string;
		readonly version: string;
		readonly licenses: readonly [{ readonly expression: string }];
	}[];
};

export function generateCycloneDx(input: {
	readonly name: string;
	readonly version: string;
	readonly components: readonly ComponentLicense[];
}): CycloneDxBom {
	return {
		bomFormat: "CycloneDX",
		specVersion: "1.5",
		version: 1,
		metadata: { component: { type: "library", name: input.name, version: input.version } },
		components: input.components.map((component) => ({
			type: "library" as const,
			name: component.name,
			version: component.version,
			licenses: [{ expression: component.license }],
		})),
	};
}

function readJsonObject(path: string): Record<string, unknown> {
	const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
	if (!isJsonObject(parsed)) throw new SupplyChainError(`${path} is not a JSON object`);
	return parsed;
}

function recordOfStrings(value: unknown): Record<string, string> {
	if (value === undefined) return {};
	if (!isJsonObject(value)) throw new SupplyChainError("dependencies must be a string map");
	const entries = Object.entries(value).map(([name, version]) => {
		if (typeof version !== "string") throw new SupplyChainError(`dependency ${name} is not a string range`);
		return [name, version] as const;
	});
	return Object.fromEntries(entries);
}

function licenseFromPackageJson(pkg: Record<string, unknown>): string {
	if (typeof pkg.license === "string") return pkg.license;
	if (isJsonObject(pkg.license)) {
		return typeof pkg.license.type === "string" ? pkg.license.type : "";
	}
	if (!Array.isArray(pkg.licenses)) return "";
	return pkg.licenses
		.flatMap((item) => {
			if (typeof item === "string") return [item];
			if (isJsonObject(item) && typeof item.type === "string") return [item.type];
			return [];
		})
		.join(" OR ");
}
