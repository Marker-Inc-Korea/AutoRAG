/**
 * OS-native file kind — the same string the platform itself reports: Finder's
 * Kind column on macOS (Spotlight `kMDItemKind`) and Explorer's Type column on
 * Windows (the registry's ProgID description). A lookup that the OS cannot
 * answer returns null so the caller keeps the extension-map fallback; it never
 * throws, and a failure is logged verbatim.
 */

import { execFile } from "node:child_process";
import { errorMessage } from "./fs-entry";

export interface KindQuery {
	readonly command: string;
	readonly args: readonly string[];
}

export interface OsKindResolver {
	/** The OS kind for a path; "" extensions are keyed by path. Null when unknown. */
	kindFor(path: string, ext: string): Promise<string | null>;
}

export interface OsKindDeps {
	/** Defaults to process.platform. */
	readonly platform?: NodeJS.Platform;
	/** Defaults to spawning the platform command with execFile. */
	readonly run?: (query: KindQuery) => Promise<string | null>;
	/** Defaults to console.warn. */
	readonly warn?: (message: string) => void;
}

const KIND_TIMEOUT_MS = 5000;
const KIND_MAX_BUFFER = 1 << 20;

function windowsKindScript(path: string): string {
	const literal = path.replace(/'/g, "''");
	return [
		`$path = '${literal}'`,
		"$ext = [System.IO.Path]::GetExtension($path)",
		"if ([string]::IsNullOrEmpty($ext)) { return }",
		"$progId = (Get-ItemProperty -LiteralPath ('Registry::HKEY_CLASSES_ROOT\\' + $ext) -ErrorAction SilentlyContinue).'(default)'",
		"if ([string]::IsNullOrEmpty($progId)) { return }",
		"$typeName = (Get-ItemProperty -LiteralPath ('Registry::HKEY_CLASSES_ROOT\\' + $progId) -ErrorAction SilentlyContinue).'(default)'",
		"if (-not [string]::IsNullOrEmpty($typeName)) { [Console]::Out.Write($typeName) }",
	].join("; ");
}

/** The OS command that reports a path's kind, or null on an unsupported platform. */
export function kindQueryForPlatform(platform: NodeJS.Platform, path: string): KindQuery | null {
	if (platform === "darwin") {
		return { command: "/usr/bin/mdls", args: ["-name", "kMDItemKind", "-raw", path] };
	}
	if (platform === "win32") {
		return {
			command: "powershell.exe",
			args: ["-NoProfile", "-NonInteractive", "-Command", windowsKindScript(path)],
		};
	}
	return null;
}

/** The kind inside a probe's stdout; null when the OS reported nothing. */
export function parseKindStdout(stdout: string): string | null {
	const value = stdout.trim().replace(/^"+|"+$/g, "").trim();
	if (value === "" || value.toLowerCase() === "null" || value === "(null)") return null;
	return value;
}

function defaultRun(query: KindQuery): Promise<string | null> {
	return new Promise((resolve, reject) => {
		execFile(query.command, [...query.args], { timeout: KIND_TIMEOUT_MS, maxBuffer: KIND_MAX_BUFFER }, (error, stdout) => {
			if (error) {
				reject(error);
				return;
			}
			resolve(parseKindStdout(stdout));
		});
	});
}

/**
 * One OS-query process per distinct extension, shared across a session: the
 * kind follows the type identifier, which the extension determines, so a folder
 * of 200 .mp4 files costs a single `mdls`.
 */
export function createOsKindResolver(deps: OsKindDeps = {}): OsKindResolver {
	const platform = deps.platform ?? process.platform;
	const run = deps.run ?? defaultRun;
	const warn = deps.warn ?? ((message: string) => console.warn(message));
	const cache = new Map<string, Promise<string | null>>();

	async function query(path: string): Promise<string | null> {
		const request = kindQueryForPlatform(platform, path);
		if (request === null) return null;
		try {
			return await run(request);
		} catch (error) {
			warn(`file-kind: ${request.command} failed for ${path}: ${errorMessage(error)}`);
			return null;
		}
	}

	return {
		kindFor(path, ext) {
			const key = ext === "" ? path : ext;
			const cached = cache.get(key);
			if (cached !== undefined) return cached;
			const pending = query(path);
			cache.set(key, pending);
			return pending;
		},
	};
}
