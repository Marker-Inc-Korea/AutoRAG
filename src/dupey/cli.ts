import { spawn } from "node:child_process";
import { resolve } from "node:path";

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

export async function scanWithDupey(dir: string, options: DupeyCliOptions = {}): Promise<DupeyScanResult> {
	const absoluteDir = resolve(dir);
	const stdout = options.run
		? await options.run(["scan", absoluteDir, "--json"])
		: await spawnDupey(options.executable ?? "dupey", ["scan", absoluteDir, "--json"], options);
	let parsed: unknown;
	try {
		parsed = JSON.parse(stdout);
	} catch {
		throw new DupeyCliError("dupey returned invalid JSON.");
	}
	if (!isScanResult(parsed)) throw new DupeyCliError("dupey returned an unsupported scan result.");
	return parsed;
}

function isScanResult(value: unknown): value is DupeyScanResult {
	if (value === null || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	return Array.isArray(record.files) && Array.isArray(record.families) && Array.isArray(record.errors);
}

function spawnDupey(executable: string, args: readonly string[], options: DupeyCliOptions): Promise<string> {
	return new Promise((resolveOutput, reject) => {
		const child = spawn(executable, [...args], {
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
