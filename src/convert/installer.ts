import { createHash } from "node:crypto";
import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, renameSync, rmSync } from "node:fs";
import { mkdtemp } from "node:fs/promises";
import { join } from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { resolveAutoRAGHome } from "../config/home.ts";
import { spawnProcess } from "../minsync/process.ts";
import { FileeCliError, type FileeCliOptions, resolveFileeExecutable } from "./cli.ts";

/**
 * filee-cli is NomaDamas' CLI-only redistribution of KnifeLemon/Filee
 * (MIT, https://github.com/NomaDamas/filee-cli). It is a self-contained .NET
 * program that must run as a separate process, so AutoRAG downloads the pinned
 * release archive, checks its SHA-256, and unpacks it under
 * `<AUTORAG_HOME>/tools/filee-cli/<version>/`. Nothing is linked or bundled.
 * Filee's own engines (FFmpeg, Pandoc, LibreOffice, ...) are fetched by
 * filee-cli itself on first use.
 */
export const FILEE_CLI_VERSION = "2.0.0-3";

export interface FileeCliAsset {
	readonly name: string;
	readonly url: string;
	readonly sha256: string;
}

const RELEASE_BASE = `https://github.com/NomaDamas/filee-cli/releases/download/v${FILEE_CLI_VERSION}`;

function asset(suffix: string, sha256: string): FileeCliAsset {
	const name = `filee-cli-${FILEE_CLI_VERSION}-${suffix}`;
	return { name, url: `${RELEASE_BASE}/${name}`, sha256 };
}

/** SHA-256 values come from the release's `SHA256SUMS`. Keyed by `<platform>-<arch>`. */
export const FILEE_CLI_ASSETS: Readonly<Record<string, FileeCliAsset>> = {
	"darwin-arm64": asset("osx-arm64.tar.gz", "e5e277e02a200001b2d7e0f63b5039497b1be99354f32e9ff534a4b0286088cc"),
	"darwin-x64": asset("osx-x64.tar.gz", "255d3f06d7df4ad367d99d63cbe542c2cda8b86fef071141c5cf52be973939de"),
	"linux-arm64": asset("linux-arm64.tar.gz", "07123e5cab6fb15a07c493e819635906c0eb4a50cfa54aea53b32b1cb557ce7c"),
	"linux-x64": asset("linux-x64.tar.gz", "3388a2a33b5dea7eec97a35e180073e3e395c52749e0933d8d7bb5709a05b463"),
	"win32-x64": asset("win-x64.zip", "73601ebaa185f047949c031da8ee3eb560ea3a7973a7f067c18e4dbcc0afdabe"),
};

export class FileeInstallError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "FileeInstallError";
	}
}

export function selectFileeAsset(
	platform: NodeJS.Platform,
	arch: NodeJS.Architecture,
	assets: Readonly<Record<string, FileeCliAsset>> = FILEE_CLI_ASSETS,
): FileeCliAsset {
	const found = assets[`${platform}-${arch}`];
	if (found === undefined) throw new FileeInstallError(`Unsupported filee-cli platform: ${platform}/${arch}`);
	return found;
}

export function fileeBinaryName(platform: NodeJS.Platform = process.platform): string {
	return platform === "win32" ? "filee-cli.exe" : "filee-cli";
}

export interface EnsureFileeBinaryOptions {
	/** AutoRAG home; the install lives at `<home>/tools/filee-cli/<version>/`. Default: the resolved AutoRAG home. */
	readonly home?: string;
	readonly platform?: NodeJS.Platform;
	readonly arch?: NodeJS.Architecture;
	/** Test seam: pinned assets. */
	readonly assets?: Readonly<Record<string, FileeCliAsset>>;
	/** Test seam: fetch `url` into `destination`. */
	readonly download?: (url: string, destination: string) => Promise<void>;
	/** Test seam: unpack `archive` into `directory`. */
	readonly extract?: (archive: string, directory: string) => Promise<void>;
}

export interface InstalledFileeBinary {
	readonly binaryPath: string;
	readonly source: "installed" | "cached";
}

const inFlight = new Map<string, Promise<InstalledFileeBinary>>();

/**
 * Make sure the pinned filee-cli is unpacked in the AutoRAG home. Idempotent;
 * concurrent callers in one process share a single install, and concurrent
 * processes race on an atomic rename (the loser discards its copy).
 */
export function ensureFileeBinary(options: EnsureFileeBinaryOptions = {}): Promise<InstalledFileeBinary> {
	const home = options.home ?? resolveAutoRAGHome();
	const platform = options.platform ?? process.platform;
	const key = `${home}\0${platform}`;
	const running = inFlight.get(key);
	if (running !== undefined) return running;
	const started = install(home, options).finally(() => inFlight.delete(key));
	inFlight.set(key, started);
	return started;
}

async function install(home: string, options: EnsureFileeBinaryOptions): Promise<InstalledFileeBinary> {
	const platform = options.platform ?? process.platform;
	const arch = options.arch ?? process.arch;
	const root = join(home, "tools", "filee-cli");
	const target = join(root, FILEE_CLI_VERSION);
	const binaryPath = join(target, fileeBinaryName(platform));
	if (existsSync(binaryPath)) return { binaryPath, source: "cached" };

	const pinned = selectFileeAsset(platform, arch, options.assets);
	// A network call in a unit test is always a bug; installer tests inject their own seams.
	if (process.env.NODE_ENV === "test" && options.download === undefined) {
		throw new FileeInstallError("filee-cli download is disabled in the test runtime.");
	}
	mkdirSync(root, { recursive: true });
	const stage = await mkdtemp(join(root, ".install-"));
	try {
		const archive = join(stage, pinned.name);
		await (options.download ?? downloadFile)(pinned.url, archive);
		await verifySha256(archive, pinned.sha256, pinned.name);
		const unpacked = join(stage, "unpacked");
		mkdirSync(unpacked);
		await (options.extract ?? extractArchive)(archive, unpacked);
		const folder = findFolderWith(unpacked, fileeBinaryName(platform));
		if (folder === undefined) {
			throw new FileeInstallError(`${pinned.name} did not contain ${fileeBinaryName(platform)}.`);
		}
		try {
			renameSync(folder, target);
		} catch (error) {
			// Another process finished first; its complete install wins.
			if (existsSync(binaryPath)) return { binaryPath, source: "cached" };
			throw new FileeInstallError(`Could not place filee-cli at ${target}: ${(error as Error).message}`);
		}
		return { binaryPath, source: "installed" };
	} finally {
		rmSync(stage, { recursive: true, force: true });
	}
}

async function downloadFile(url: string, destination: string): Promise<void> {
	const response = await fetch(url, { headers: { "User-Agent": "AutoRAG-filee-cli" }, redirect: "follow" });
	if (!response.ok || response.body === null) {
		throw new FileeInstallError(`filee-cli download failed with HTTP ${response.status}: ${url}`);
	}
	await pipeline(Readable.fromWeb(response.body as never), createWriteStream(destination));
}

async function verifySha256(path: string, expected: string, name: string): Promise<void> {
	const hash = createHash("sha256");
	await pipeline(createReadStream(path), hash);
	const actual = hash.digest("hex");
	if (actual !== expected) {
		throw new FileeInstallError(`${name} sha256 mismatch: expected ${expected}, got ${actual}`);
	}
}

/** `tar` unpacks .tar.gz everywhere and .zip on Windows (bsdtar). */
async function extractArchive(archive: string, directory: string): Promise<void> {
	const result = await spawnProcess("tar", ["-xf", archive, "-C", directory], directory);
	if (!result.ok) throw new FileeInstallError(result.stderr || `Could not extract ${archive}`);
}

/** The package is one top-level folder; accept the binary at the top or one level down. */
function findFolderWith(directory: string, binary: string): string | undefined {
	if (existsSync(join(directory, binary))) return directory;
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		if (entry.isDirectory() && existsSync(join(directory, entry.name, binary))) return join(directory, entry.name);
	}
	return undefined;
}

export interface ResolveFileeBinaryOptions {
	readonly home?: string;
	/** `filee.binaryPath` from config. */
	readonly executable?: string;
	readonly env?: NodeJS.ProcessEnv;
	/** `false` never downloads. Default true. */
	readonly autoInstall?: boolean;
	readonly installer?: Omit<EnsureFileeBinaryOptions, "home">;
}

export interface ResolvedFileeBinary {
	readonly path: string;
	readonly source: "configured" | "path" | "installed" | "cached";
}

/**
 * Where the Filee command line comes from, in order: the configured path, a
 * `filee`/`filee-cli` on PATH, then the managed copy (installed on demand).
 */
export async function resolveFileeBinary(options: ResolveFileeBinaryOptions = {}): Promise<ResolvedFileeBinary> {
	if (options.executable !== undefined) {
		if (!existsSync(options.executable)) {
			throw new FileeCliError(`filee.binaryPath does not exist: ${options.executable}`);
		}
		return { path: options.executable, source: "configured" };
	}
	const onPath = resolveFileeExecutable(options.env ?? process.env);
	if (onPath !== undefined) return { path: onPath, source: "path" };
	const home = options.home ?? resolveAutoRAGHome(options.env ?? process.env);
	const cached = join(
		home,
		"tools",
		"filee-cli",
		FILEE_CLI_VERSION,
		fileeBinaryName(options.installer?.platform ?? process.platform),
	);
	if (existsSync(cached)) return { path: cached, source: "cached" };
	if (options.autoInstall === false) {
		throw new FileeCliError(
			"filee-cli is not installed. Install it (brew install NomaDamas/filee-cli/filee-cli, or a release from https://github.com/NomaDamas/filee-cli/releases) or set filee.autoInstall to true in config.json.",
		);
	}
	try {
		const installed = await ensureFileeBinary({ ...options.installer, home });
		return { path: installed.binaryPath, source: installed.source };
	} catch (error) {
		throw new FileeCliError(
			`Could not install filee-cli automatically: ${error instanceof Error ? error.message : String(error)}`,
		);
	}
}

/**
 * Fill in `executable` for a conversion call: the configured path, a PATH
 * install, or the managed copy (downloaded on first use unless `autoInstall`
 * is false). Options that already carry a `run` seam are returned unchanged.
 */
export async function resolveFileeOptions(
	options: FileeCliOptions,
	deps: Pick<ResolveFileeBinaryOptions, "home" | "env" | "installer"> = {},
): Promise<FileeCliOptions> {
	if (options.run !== undefined) return options;
	const resolved = await resolveFileeBinary({
		...deps,
		...(options.executable !== undefined ? { executable: options.executable } : {}),
		...(options.autoInstall !== undefined ? { autoInstall: options.autoInstall } : {}),
	});
	return { ...options, executable: resolved.path };
}

export type WarmFileeOutcome =
	| { readonly state: ResolvedFileeBinary["source"]; readonly path: string }
	| { readonly state: "skipped" }
	| { readonly state: "failed"; readonly reason: string };

/**
 * Start-of-session prefetch so the first `convert_files` call does not wait for
 * an 85 MB download. Never throws: a failure here resurfaces, with its reason,
 * on the first real conversion.
 */
export async function warmFileeBinary(
	options: FileeCliOptions | false,
	deps: Pick<ResolveFileeBinaryOptions, "home" | "env" | "installer"> = {},
): Promise<WarmFileeOutcome> {
	if (options === false || options.autoInstall === false || options.run !== undefined) return { state: "skipped" };
	try {
		const resolved = await resolveFileeBinary({
			...deps,
			...(options.executable !== undefined ? { executable: options.executable } : {}),
		});
		return { state: resolved.source, path: resolved.path };
	} catch (error) {
		return { state: "failed", reason: error instanceof Error ? error.message : String(error) };
	}
}
