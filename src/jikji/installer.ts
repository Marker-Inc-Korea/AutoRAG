import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { delimiter, join } from "node:path";

/**
 * Jikji CLI installer. Jikji ships as the `jikji-cli` crate on crates.io
 * (binary name `jikji`); there are no prebuilt GitHub release binaries, so
 * installation compiles from source through the user's Rust toolchain.
 * The binary is cached under `<root>/.autorag/bin` — the same directory the
 * MinSync installer uses.
 */

export const JIKJI_CRATE_NAME = "jikji-cli";
export const JIKJI_INSTALL_TIMEOUT_MS = 10 * 60 * 1_000;

export function jikjiExecutableName(platform: NodeJS.Platform = process.platform): string {
	return platform === "win32" ? "jikji.exe" : "jikji";
}

export function cargoExecutableName(platform: NodeJS.Platform = process.platform): string {
	return platform === "win32" ? "cargo.exe" : "cargo";
}

export function cachedJikjiBinaryPath(root: string, platform: NodeJS.Platform = process.platform): string {
	return join(root, ".autorag", "bin", jikjiExecutableName(platform));
}

/** Resolve an executable from PATH directories. Returns the absolute path of the first match or undefined. */
export function lookupExecutableInPath(executable: string, env: NodeJS.ProcessEnv): string | undefined {
	const pathEnv = env.PATH;
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, executable);
		if (existsSync(candidate)) return candidate;
	}
	return undefined;
}

export interface JikjiInstallRunResult {
	readonly code: number | null;
	readonly stderr: string;
}

export type JikjiInstallRunner = (args: readonly string[], timeoutMs: number) => Promise<JikjiInstallRunResult>;

export interface EnsureJikjiBinaryOptions {
	/** Workspace root; the binary is cached at `<root>/.autorag/bin/jikji`. */
	readonly root: string;
	/** crates.io version pin for `jikji-cli`. Defaults to latest. */
	readonly version?: string;
	readonly env?: NodeJS.ProcessEnv;
	readonly timeoutMs?: number;
	/** Injectable process runner for tests. Defaults to spawning `cargo install`. */
	readonly runner?: JikjiInstallRunner;
	/** Injectable cargo lookup for tests. Defaults to PATH lookup. */
	readonly cargoLocator?: (env: NodeJS.ProcessEnv) => string | undefined;
}

export type EnsureJikjiBinaryResult =
	| { readonly ok: true; readonly binaryPath: string; readonly source: "cached" | "installed" }
	| { readonly ok: false; readonly reason: "missing-cargo" | "install-failed"; readonly message: string };

function defaultRunner(cargo: string, env: NodeJS.ProcessEnv): JikjiInstallRunner {
	return (args, timeoutMs) =>
		new Promise((resolvePromise) => {
			const child = spawn(cargo, args, { env, stdio: ["ignore", "ignore", "pipe"] });
			let stderr = "";
			const timeout = setTimeout(() => {
				if (!child.killed) child.kill("SIGTERM");
			}, timeoutMs);
			child.stderr.setEncoding("utf8");
			child.stderr.on("data", (chunk: string) => {
				stderr += chunk;
			});
			child.on("error", (error) => {
				clearTimeout(timeout);
				resolvePromise({ code: null, stderr: error.message });
			});
			child.on("close", (code) => {
				clearTimeout(timeout);
				resolvePromise({ code, stderr });
			});
		});
}

/**
 * Ensure a `jikji` binary exists under `<root>/.autorag/bin`, installing the
 * `jikji-cli` crate through cargo when missing. Never throws: failures are
 * reported as degraded results so callers can fall back to raw discovery.
 */
export async function ensureJikjiBinary(options: EnsureJikjiBinaryOptions): Promise<EnsureJikjiBinaryResult> {
	const env = options.env ?? process.env;
	const cached = cachedJikjiBinaryPath(options.root);
	if (existsSync(cached)) return { ok: true, binaryPath: cached, source: "cached" };

	const cargoLocator = options.cargoLocator ?? ((e) => lookupExecutableInPath(cargoExecutableName(), e));
	const cargo = cargoLocator(env);
	if (cargo === undefined) {
		return {
			ok: false,
			reason: "missing-cargo",
			message:
				"Jikji auto-install needs the Rust toolchain (cargo) on PATH — install via https://rustup.rs, or install jikji manually (`cargo install jikji-cli`).",
		};
	}

	const args = ["install", JIKJI_CRATE_NAME, "--locked", "--root", join(options.root, ".autorag")];
	if (options.version !== undefined) args.push("--version", options.version);
	const runner = options.runner ?? defaultRunner(cargo, env);
	const result = await runner(args, options.timeoutMs ?? JIKJI_INSTALL_TIMEOUT_MS);
	if (result.code === 0 && existsSync(cached)) {
		return { ok: true, binaryPath: cached, source: "installed" };
	}
	return {
		ok: false,
		reason: "install-failed",
		message: `cargo install ${JIKJI_CRATE_NAME} failed (exit ${result.code ?? "unknown"}): ${result.stderr.slice(0, 500)}`,
	};
}
