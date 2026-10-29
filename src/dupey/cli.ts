import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { portableSpawnCommand } from "../process/portable-spawn.ts";

export interface DupeyScanFile {
	readonly path: string;
	readonly content_hash?: string;
	readonly [key: string]: unknown;
}

export interface DupeyScanEdge {
	readonly relation: string;
	readonly a: string;
	readonly b: string;
	readonly [key: string]: unknown;
}

export interface DupeyScanFamily {
	readonly id: number;
	readonly relation: string;
	readonly files: readonly string[];
	readonly members: readonly Record<string, unknown>[];
	readonly edges: readonly DupeyScanEdge[];
	readonly pick?: Record<string, unknown>;
	readonly [key: string]: unknown;
}

export interface DupeyScanResult {
	readonly dir: string;
	readonly threshold?: number;
	readonly contains_threshold?: number;
	readonly contains_min_jaccard?: number;
	readonly files: readonly DupeyScanFile[];
	readonly families: readonly DupeyScanFamily[];
	readonly errors: readonly Record<string, unknown>[];
	readonly [key: string]: unknown;
}

export interface DupeyCliOptions {
	readonly executable?: string;
	readonly cwd?: string;
	readonly timeoutMs?: number;
	readonly run?: (args: readonly string[]) => Promise<string>;
}

export class DupeyCliError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "DupeyCliError";
	}
}

function runDupey(args: readonly string[], options: DupeyCliOptions): Promise<string> {
	return options.run ? options.run(args) : spawnDupey(options.executable ?? "dupey", args, options);
}

export async function scanWithDupey(dir: string, options: DupeyCliOptions = {}): Promise<DupeyScanResult> {
	const absoluteDir = resolve(dir);
	const stdout = await runDupey(["scan", absoluteDir, "--json"], options);
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		throw new DupeyCliError("dupey returned invalid JSON.");
	}
	if (!isScanResult(parsed)) throw new DupeyCliError("dupey returned an unsupported scan result.");
	return parsed;
}

/** `dupey scan`'s own defaults: `--threshold`, `--contains-threshold`, `--contains-min-jaccard`. */
const NEAR_JACCARD = 0.9;
const CONTAINS_CONTAINMENT = 0.96;
const CONTAINS_MIN_JACCARD = 0.4;

export type DupeyRelation = "exact" | "near" | "contains" | "different";

export type DupeyCompareResult =
	| {
			readonly relation: DupeyRelation;
			readonly nearScore: number;
			readonly jaccard: number;
			readonly containmentAInB?: number;
			readonly containmentBInA?: number;
	  }
	/** dupey cannot extract text from one of the files (for example HTML or an image), so no family can form. */
	| { readonly relation: "unsupported" };

/**
 * Would `dupey scan` put these two files in one family? Classifies with the
 * same default thresholds the scan uses, so the answer matches what a later
 * scan of a directory holding both will report.
 */
export async function compareWithDupey(
	a: string,
	b: string,
	options: DupeyCliOptions = {},
): Promise<DupeyCompareResult> {
	let stdout: string;
	try {
		stdout = await runDupey(["compare", resolve(a), resolve(b)], options);
	} catch (error) {
		if (error instanceof DupeyCliError && error.message.includes("unsupported format")) {
			return { relation: "unsupported" };
		}
		throw error;
	}
	const fields = new Map<string, string>();
	for (const line of stdout.split(/\r?\n/)) {
		const tab = line.indexOf("\t");
		if (tab > 0) fields.set(line.slice(0, tab), line.slice(tab + 1).trim());
	}
	const nearScore = parseScore(fields.get("near_score"));
	const jaccard = parseScore(fields.get("jaccard"));
	if (nearScore === undefined || jaccard === undefined) {
		throw new DupeyCliError("dupey returned an unsupported compare result.");
	}
	const containmentAInB = parseScore(fields.get("containment_a_in_b"));
	const containmentBInA = parseScore(fields.get("containment_b_in_a"));
	const containment = Math.max(containmentAInB ?? 0, containmentBInA ?? 0);
	let relation: DupeyRelation = "different";
	if (fields.get("exact_equal") === "true") relation = "exact";
	else if (jaccard >= NEAR_JACCARD) relation = "near";
	else if (containment >= CONTAINS_CONTAINMENT && jaccard >= CONTAINS_MIN_JACCARD) relation = "contains";
	return {
		relation,
		nearScore,
		jaccard,
		...(containmentAInB !== undefined ? { containmentAInB } : {}),
		...(containmentBInA !== undefined ? { containmentBInA } : {}),
	};
}

function parseScore(value: string | undefined): number | undefined {
	if (value === undefined) return undefined;
	const parsed = Number(value);
	return Number.isFinite(parsed) ? parsed : undefined;
}

function isScanResult(value: unknown): value is DupeyScanResult {
	if (value === null || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	return Array.isArray(record.files) && Array.isArray(record.families) && Array.isArray(record.errors);
}

function spawnDupey(executable: string, args: readonly string[], options: DupeyCliOptions): Promise<string> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise((resolveOutput, reject) => {
		// Windows cannot spawn shebang scripts or extensionless entrypoints; route
		// script shims through their interpreter the same way MinSync/FSearch do.
		const portable = portableSpawnCommand(executable, args);
		const child = spawn(portable.command, [...portable.args], {
			cwd: options.cwd,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout = "";
		let stderr = "";
		let settled = false;
		const timeout = setTimeout(() => {
			child.kill("SIGTERM");
			if (!settled) {
				settled = true;
				reject(new DupeyCliError("dupey timed out."));
			}
		}, options.timeoutMs ?? 120_000);
		child.stdout.on("data", (chunk: Buffer) => (stdout += chunk.toString()));
		child.stderr.on("data", (chunk: Buffer) => (stderr += chunk.toString()));
		child.once("error", (error) => {
			clearTimeout(timeout);
			if (!settled) {
				settled = true;
				reject(new DupeyCliError(`Unable to run dupey: ${error.message}`));
			}
		});
		child.once("close", (code) => {
			clearTimeout(timeout);
			if (settled) return;
			settled = true;
			if (code !== 0) {
				reject(
					new DupeyCliError(
						`dupey failed with exit code ${code ?? "unknown"}${stderr.trim() ? `: ${stderr.trim()}` : ""}`,
					),
				);
				return;
			}
			resolveOutput(stdout);
		});
	});
}
