import { spawn } from "node:child_process";
import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, dirname, join, resolve } from "node:path";
import { portableSpawnCommand } from "../process/portable-spawn.ts";

/**
 * Conversions can legitimately take a long time (video transcodes, big PDFs,
 * HWP → PDF), so the default budget is generous. Callers may lower it per call.
 */
export const DEFAULT_FILEE_TIMEOUT_MS = 30 * 60 * 1000;

export interface FileeConvertRequest {
	/** Files, folders or wildcards. Relative paths are resolved against `cwd`. Any location is allowed. */
	readonly inputs: readonly string[];
	/** Target format id or extension (`pdf`, `docx`, `webp`, `mp3`, ...). See {@link listFileeFormats}. */
	readonly to: string;
	/** Save here instead of next to each source. */
	readonly outputDir?: string;
	/** Include sub folders of folder inputs. */
	readonly recursive?: boolean;
	/** 1-100, for lossy image formats. */
	readonly quality?: number;
	/** Replace an existing output file. Without it (and without `skipExisting`) a number is added to the new name. */
	readonly overwrite?: boolean;
	/** Leave an existing output file alone. */
	readonly skipExisting?: boolean;
	/** Give the output the modified time of its source. */
	readonly keepDates?: boolean;
}

export interface FileeFileResult {
	readonly source: string;
	readonly state: "done" | "skipped" | "failed" | "cancelled" | (string & {});
	readonly outputs: readonly string[];
	readonly error?: string;
	readonly notice?: string;
}

export interface FileeConvertResult {
	readonly preset: string;
	readonly target: string;
	readonly converted: number;
	readonly skipped: number;
	readonly failed: number;
	readonly files: readonly FileeFileResult[];
}

export interface FileeCliRun {
	readonly stdout: string;
	readonly stderr: string;
	readonly exitCode: number;
}

export interface FileeCliOptions {
	/** Path to `filee` (or `filee-cli`). Default: found on PATH. */
	readonly executable?: string;
	readonly cwd?: string;
	readonly timeoutMs?: number;
	readonly signal?: AbortSignal;
	/**
	 * Folder for Filee's settings and downloaded engines (`FILEE_DATA_DIR`), so a
	 * run neither reads nor writes the user's own Filee app settings.
	 */
	readonly dataDir?: string;
	/** `false` never downloads filee-cli when none is found. Default true. See `resolveFileeOptions`. */
	readonly autoInstall?: boolean;
	/** Test seam: replaces the spawn. */
	readonly run?: (args: readonly string[]) => Promise<FileeCliRun>;
}

export class FileeCliError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "FileeCliError";
	}
}

/**
 * Convert files with the `filee` command. Exit 0 (all converted) and 1 (some
 * failed) both return the per-file report; wrong arguments, an unknown format,
 * a missing input and "nothing to convert" throw {@link FileeCliError}.
 */
export async function convertWithFilee(
	request: FileeConvertRequest,
	options: FileeCliOptions = {},
): Promise<FileeConvertResult> {
	const args = buildConvertArgs(request, options.cwd);
	const run = await runFilee(args, options);
	if (run.exitCode !== 0 && run.exitCode !== 1) throw failure(run);
	let parsed: unknown;
	try {
		parsed = JSON.parse(run.stdout);
	} catch {
		throw new FileeCliError("filee returned invalid JSON.");
	}
	if (!isConvertResult(parsed)) throw new FileeCliError("filee returned an unsupported conversion result.");
	return parsed;
}

/** `filee formats [format]`: every format, or what one format converts to. */
export async function listFileeFormats(format?: string, options: FileeCliOptions = {}): Promise<string> {
	const run = await runFilee(format === undefined ? ["formats"] : ["formats", format], options);
	if (run.exitCode !== 0) throw failure(run);
	return run.stdout.trim();
}

function buildConvertArgs(request: FileeConvertRequest, cwd: string | undefined): string[] {
	if (request.inputs.length === 0) throw new FileeCliError("Give at least one file or folder to convert.");
	const to = request.to.trim();
	if (to.length === 0) throw new FileeCliError("Give a target format.");
	if (
		request.quality !== undefined &&
		!(Number.isInteger(request.quality) && request.quality >= 1 && request.quality <= 100)
	) {
		throw new FileeCliError("quality must be a whole number from 1 to 100.");
	}
	if (request.overwrite === true && request.skipExisting === true) {
		throw new FileeCliError("overwrite and skipExisting cannot both be set.");
	}
	const base = cwd ?? process.cwd();
	const args = ["convert", ...request.inputs.map((input) => resolve(base, input)), "--to", to, "--json"];
	if (request.outputDir !== undefined) args.push("-o", resolve(base, request.outputDir));
	if (request.recursive === true) args.push("--recursive");
	if (request.quality !== undefined) args.push("--quality", String(request.quality));
	if (request.overwrite === true) args.push("--overwrite");
	if (request.skipExisting === true) args.push("--skip");
	if (request.keepDates === true) args.push("--keep-dates");
	return args;
}

function failure(run: FileeCliRun): FileeCliError {
	const detail = run.stderr
		.split(/\r?\n/)
		.map((line) => line.replace(/^filee:\s*/, "").trim())
		.filter((line) => line.length > 0 && !line.startsWith("Run 'filee help'"))
		.join(" ");
	if (run.exitCode === 2 || run.exitCode === 3)
		return new FileeCliError(detail || `filee exited with code ${run.exitCode}.`);
	return new FileeCliError(`filee failed with exit code ${run.exitCode}${detail ? `: ${detail}` : ""}`);
}

function isConvertResult(value: unknown): value is FileeConvertResult {
	if (value === null || typeof value !== "object") return false;
	const record = value as Record<string, unknown>;
	return (
		typeof record.converted === "number" &&
		typeof record.failed === "number" &&
		Array.isArray(record.files) &&
		record.files.every(
			(file) =>
				file !== null &&
				typeof file === "object" &&
				typeof (file as Record<string, unknown>).source === "string" &&
				typeof (file as Record<string, unknown>).state === "string" &&
				Array.isArray((file as Record<string, unknown>).outputs),
		)
	);
}

function runFilee(args: readonly string[], options: FileeCliOptions): Promise<FileeCliRun> {
	if (options.run) return options.run(args);
	const executable = options.executable ?? resolveFileeExecutable() ?? "filee";
	return spawnFilee(executable, args, options);
}

function spawnFilee(executable: string, args: readonly string[], options: FileeCliOptions): Promise<FileeCliRun> {
	return new Promise((resolveRun, reject) => {
		if (options.signal?.aborted) {
			reject(new FileeCliError("filee was aborted."));
			return;
		}
		const portable = portableSpawnCommand(executable, args);
		const env =
			options.dataDir === undefined ? process.env : { ...process.env, FILEE_DATA_DIR: resolve(options.dataDir) };
		const child = spawn(portable.command, [...portable.args], {
			cwd: options.cwd,
			env,
			stdio: ["ignore", "pipe", "pipe"],
		});
		// Decode once at the end: a multibyte character can straddle two chunks.
		const stdout: Buffer[] = [];
		const stderr: Buffer[] = [];
		let settled = false;
		const finish = (action: () => void): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			options.signal?.removeEventListener("abort", onAbort);
			action();
		};
		const timeout = setTimeout(() => {
			child.kill("SIGTERM");
			finish(() => reject(new FileeCliError("filee timed out.")));
		}, options.timeoutMs ?? DEFAULT_FILEE_TIMEOUT_MS);
		const onAbort = (): void => {
			child.kill("SIGTERM");
			finish(() => reject(new FileeCliError("filee was aborted.")));
		};
		options.signal?.addEventListener("abort", onAbort, { once: true });
		child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
		child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
		child.once("error", (error) => finish(() => reject(new FileeCliError(`Unable to run filee: ${error.message}`))));
		child.once("close", (code) =>
			finish(() =>
				resolveRun({
					stdout: Buffer.concat(stdout).toString("utf8"),
					stderr: Buffer.concat(stderr).toString("utf8"),
					exitCode: code ?? -1,
				}),
			),
		);
	});
}

/**
 * Find the Filee command line on `PATH`. POSIX installs link it as `filee`
 * (or leave `filee-cli`). On Windows the PATH entry is the `cli\filee.cmd`
 * shim, which Node cannot spawn directly; its target is `filee-cli.exe` one
 * folder up. `filee.exe` is the tray app, never the command line.
 */
export function resolveFileeExecutable(
	env: NodeJS.ProcessEnv = process.env,
	platform: NodeJS.Platform = process.platform,
): string | undefined {
	const key = Object.keys(env).find((name) =>
		platform === "win32" ? name.toUpperCase() === "PATH" : name === "PATH",
	);
	const pathEnv = key === undefined ? undefined : env[key];
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	for (const dir of pathEnv.split(platform === "win32" ? ";" : delimiter)) {
		if (dir.length === 0) continue;
		const candidates = platform === "win32" ? windowsCandidates(dir) : [join(dir, "filee"), join(dir, "filee-cli")];
		for (const candidate of candidates) {
			try {
				accessSync(candidate, constants.X_OK);
				return candidate;
			} catch {
				// not here, keep looking
			}
		}
	}
	return undefined;
}

/** `filee-cli.exe` in `dir` (portable folder), or one level up when `dir` holds the `filee.cmd` shim. */
function windowsCandidates(dir: string): string[] {
	const candidates = [join(dir, "filee-cli.exe")];
	if (existsSync(join(dir, "filee.cmd"))) candidates.push(join(dirname(dir), "filee-cli.exe"));
	return candidates;
}
