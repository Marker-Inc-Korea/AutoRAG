import { spawn } from "node:child_process";
import { resolve } from "node:path";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { DUPEY_MIN_VERSION, isSupportedDupeyVersion, parseDupeyVersion } from "./version.ts";

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

/** One file's identity as `dupey fingerprint` reports it. `exact` equals `dupey scan --json`'s `content_hash`. */
export interface DupeyFingerprint {
	readonly path: string;
	readonly format: string;
	readonly chars: number;
	readonly exact: string;
	readonly minhashWidth: number;
	/** The document's own last-modified time, or `null` when dupey records none (`-`). */
	readonly modified: string | null;
	readonly revision: string | null;
}

export class DupeyCliError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "DupeyCliError";
	}
}

/** One `--version` probe per executable per process; a failed probe is forgotten so a later install is seen. */
const versionChecks = new Map<string, Promise<void>>();

/**
 * Refuse a dupey below {@link DUPEY_MIN_VERSION}: its hashes disagree with the ones AutoRAG stores. A version line
 * AutoRAG cannot read is let through, as Jikji's gate does. Test doubles (`run`) are never probed.
 */
function checkDupeyVersion(executable: string, options: DupeyCliOptions): Promise<void> {
	const cached = versionChecks.get(executable);
	if (cached !== undefined) return cached;
	const check = spawnDupey(executable, ["--version"], options).then((stdout) => {
		const version = parseDupeyVersion(stdout);
		if (version !== undefined && !isSupportedDupeyVersion(version)) {
			throw new DupeyCliError(
				`dupey ${version} at ${executable} is older than the ${DUPEY_MIN_VERSION} AutoRAG requires; run \`cargo install dupey\` to update it.`,
			);
		}
	});
	check.catch(() => versionChecks.delete(executable));
	versionChecks.set(executable, check);
	return check;
}

async function runDupey(args: readonly string[], options: DupeyCliOptions): Promise<string> {
	if (options.run) return options.run(args);
	const executable = options.executable ?? "dupey";
	await checkDupeyVersion(executable, options);
	return spawnDupey(executable, args, options);
}

export async function scanWithDupey(
	dir: string,
	options: DupeyCliOptions & { readonly excludeDirs?: readonly string[] } = {},
): Promise<DupeyScanResult> {
	const absoluteDir = resolve(dir);
	const args = ["scan", absoluteDir, "--json"];
	for (const name of options.excludeDirs ?? []) args.push("--exclude-dir", name);
	const stdout = await runDupey(args, options);
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		throw new DupeyCliError("dupey returned invalid JSON.");
	}
	if (!isScanResult(parsed)) throw new DupeyCliError("dupey returned an unsupported scan result.");
	return parsed;
}

/**
 * Read one file's canonical-text identity with `dupey fingerprint` while it is on disk. This is how a cloud
 * file is judged for exact-duplicate exclusion without ever reading it again after its single download:
 * `exact` is the same hash `dupey scan --json` would report (`content_hash`), so a scanned copy elsewhere
 * and this stored fingerprint group together.
 */
export async function fingerprintWithDupey(path: string, options: DupeyCliOptions = {}): Promise<DupeyFingerprint> {
	const absolute = resolve(path);
	const stdout = await runDupey(["fingerprint", absolute], options);
	const fields = new Map<string, string>();
	for (const line of stdout.split(/\r?\n/)) {
		const tab = line.indexOf("\t");
		if (tab > 0) fields.set(line.slice(0, tab), line.slice(tab + 1).trim());
	}
	const exact = fields.get("exact");
	if (exact === undefined || exact.length === 0) {
		throw new DupeyCliError("dupey fingerprint returned no exact hash.");
	}
	const chars = Number.parseInt(fields.get("chars") ?? "", 10);
	const minhashWidth = Number.parseInt(fields.get("minhash_width") ?? "", 10);
	// `dupey fingerprint` prints `-` for an absent value; normalize that to `null`.
	const nullable = (value: string | undefined): string | null =>
		value === undefined || value === "" || value === "-" ? null : value;
	return {
		path: fields.get("path") ?? absolute,
		format: fields.get("format") ?? "",
		chars: Number.isFinite(chars) ? chars : 0,
		exact,
		minhashWidth: Number.isFinite(minhashWidth) ? minhashWidth : 0,
		modified: nullable(fields.get("modified")),
		revision: nullable(fields.get("revision")),
	};
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
