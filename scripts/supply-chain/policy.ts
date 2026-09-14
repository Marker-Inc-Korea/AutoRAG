export type CveSeverity = "critical" | "high" | "moderate" | "low" | "unknown";

export type SupplyChainPolicy = {
	readonly projectLicenses: readonly string[];
	readonly allowLicenses: readonly string[];
	readonly failOnCveSeverity: "critical" | "high" | "moderate" | "low";
	readonly unknownLicense: "deny" | "allow";
};

export type ComponentLicense = {
	readonly name: string;
	readonly version: string;
	readonly license: string;
};

export type LicenseDenial = {
	readonly decision: "deny";
	readonly reason: string;
	readonly component: ComponentLicense;
};

export type CveFinding = {
	readonly id: string;
	readonly severity: CveSeverity;
	readonly packageName: string;
};

export type ReleaseGateInput = {
	readonly sbomOk: boolean;
	readonly licenseOk: boolean;
	readonly cveOk: boolean;
	readonly noticeOk: boolean;
};

const ALLOW_LICENSES = [
	"0BSD",
	"Apache-1.1",
	"Apache-2.0",
	"Artistic-2.0",
	"BlueOak-1.0.0",
	"BSD-2-Clause",
	"BSD-3-Clause",
	"BSD-3-Clause-Clear",
	"BSL-1.0",
	"CC-BY-3.0",
	"CC-BY-4.0",
	"CC0-1.0",
	"HPND",
	"ISC",
	"MIT",
	"MIT-0",
	"MPL-2.0",
	"MS-PL",
	"NCSA",
	"OpenSSL",
	"PostgreSQL",
	"PSF-2.0",
	"Python-2.0",
	"Unicode-3.0",
	"Unicode-DFS-2016",
	"Unlicense",
	"WTFPL",
	"X11",
	"Zlib",
] as const;

export const AUTORAG_SUPPLY_CHAIN_POLICY: SupplyChainPolicy = {
	projectLicenses: ["MIT", "Apache-2.0"],
	allowLicenses: ALLOW_LICENSES,
	failOnCveSeverity: "high",
	unknownLicense: "deny",
};

const SEVERITY_RANK: Readonly<Record<CveSeverity, number>> = {
	critical: 4,
	high: 3,
	moderate: 2,
	low: 1,
	unknown: 4,
};

const GATE_ORDER = ["sbom", "license", "cve", "notice"] as const;

export class SupplyChainError extends Error {
	readonly name = "SupplyChainError";
}

export function evaluateSpdxExpression(expression: string, policy: SupplyChainPolicy): boolean {
	const trimmed = expression.trim();
	if (trimmed === "") return policy.unknownLicense === "allow";
	const allow = new Set(policy.allowLicenses.map((id) => id.toLowerCase()));
	try {
		const parser = { tokens: tokenize(trimmed), index: 0 };
		const allowed = parseOr(parser, allow);
		return allowed && parser.index === parser.tokens.length;
	} catch (error) {
		if (error instanceof SupplyChainError) return false;
		throw error;
	}
}

export function evaluateLicenses(
	components: readonly ComponentLicense[],
	policy: SupplyChainPolicy,
): { readonly ok: boolean; readonly denials: readonly LicenseDenial[] } {
	const denials = components
		.filter((component) => !evaluateSpdxExpression(component.license, policy))
		.map((component) => ({
			decision: "deny" as const,
			reason: `license ${component.license || "<empty>"} is outside the AutoRAG allowlist`,
			component,
		}));
	return { ok: denials.length === 0, denials };
}

export function severityAtOrAbove(severity: CveSeverity, threshold: SupplyChainPolicy["failOnCveSeverity"]): boolean {
	return SEVERITY_RANK[severity] >= SEVERITY_RANK[threshold];
}

export function evaluateCves(
	findings: readonly CveFinding[],
	policy: SupplyChainPolicy,
): { readonly ok: boolean; readonly blocking: readonly CveFinding[] } {
	const blocking = findings.filter((finding) => severityAtOrAbove(finding.severity, policy.failOnCveSeverity));
	return { ok: blocking.length === 0, blocking };
}

export function evaluateReleaseGate(input: ReleaseGateInput): {
	readonly ok: boolean;
	readonly missing: readonly string[];
} {
	const flags = {
		sbom: input.sbomOk,
		license: input.licenseOk,
		cve: input.cveOk,
		notice: input.noticeOk,
	} as const;
	const missing = GATE_ORDER.filter((name) => !flags[name]);
	return { ok: missing.length === 0, missing };
}

type Parser = { readonly tokens: readonly string[]; index: number };

function tokenize(expression: string): readonly string[] {
	const tokens: string[] = [];
	let index = 0;
	while (index < expression.length) {
		const char = expression[index];
		if (char === undefined || /\s/.test(char)) {
			index += 1;
			continue;
		}
		if (char === "(" || char === ")") {
			tokens.push(char);
			index += 1;
			continue;
		}
		let end = index + 1;
		while (end < expression.length) {
			const current = expression[end];
			if (current === undefined || /\s/.test(current) || current === "(" || current === ")") break;
			end += 1;
		}
		tokens.push(expression.slice(index, end));
		index = end;
	}
	return tokens;
}

function parseOr(parser: Parser, allow: ReadonlySet<string>): boolean {
	let allowed = parseAnd(parser, allow);
	while (peek(parser)?.toUpperCase() === "OR") {
		next(parser);
		allowed = parseAnd(parser, allow) || allowed;
	}
	return allowed;
}

function parseAnd(parser: Parser, allow: ReadonlySet<string>): boolean {
	let allowed = parsePrimary(parser, allow);
	while (peek(parser)?.toUpperCase() === "AND") {
		next(parser);
		allowed = parsePrimary(parser, allow) && allowed;
	}
	return allowed;
}

function parsePrimary(parser: Parser, allow: ReadonlySet<string>): boolean {
	if (peek(parser) === "(") {
		next(parser);
		const inner = parseOr(parser, allow);
		if (peek(parser) !== ")") throw new SupplyChainError("unbalanced SPDX expression");
		next(parser);
		return inner;
	}
	const identifier = next(parser);
	if (peek(parser)?.toUpperCase() === "WITH") {
		next(parser);
		next(parser);
	}
	const normalized = identifier.endsWith("+") ? `${identifier.slice(0, -1)}-or-later` : identifier;
	return allow.has(normalized.toLowerCase());
}

function peek(parser: Parser): string | undefined {
	return parser.tokens[parser.index];
}

function next(parser: Parser): string {
	const token = parser.tokens[parser.index];
	if (token === undefined) throw new SupplyChainError("unexpected end of SPDX expression");
	parser.index += 1;
	return token;
}
