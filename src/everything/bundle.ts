import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import JSZip from "jszip";

/**
 * voidtools Everything (MIT, closed source) and its MIT-licensed ES command
 * line client ship inside the AutoRAG package as pinned portable ZIPs under
 * `vendor/everything`. On Windows they are extracted, SHA-256 verified, into
 * `<workspace>/.autorag/everything/<version>/<arch>/`. Nothing is downloaded
 * and nothing is installed system-wide.
 */

export type EverythingArch = "x64" | "arm64";

/**
 * `"install"` extracts and verifies the pinned binaries when the workspace
 * cache is missing or invalid; `"cached"` returns only an already extracted,
 * hash-verified cache entry and never writes to disk.
 */
export type EverythingBinaryResolutionMode = "cached" | "install";

export interface EverythingArchAssets {
	readonly everythingArchive: string;
	readonly everythingArchiveSha256: string;
	readonly everythingMember: string;
	readonly everythingSha256: string;
	readonly esArchive: string;
	readonly esArchiveSha256: string;
	readonly esMember: string;
	readonly esSha256: string;
}

export interface EverythingBundleManifest {
	readonly everythingVersion: string;
	readonly esVersion: string;
	readonly licenseFiles: readonly string[];
	readonly architectures: Partial<Record<EverythingArch, EverythingArchAssets>>;
}

export interface EnsureEverythingBinariesOptions {
	/** Workspace root; binaries are cached under `<root>/.autorag/everything`. */
	readonly root: string;
	readonly platform?: NodeJS.Platform;
	readonly arch?: string;
	/** Directory holding the bundled ZIPs. Defaults to the packaged `vendor/everything`. */
	readonly bundleDir?: string;
	/** Bundle manifest. Defaults to `<bundleDir>/manifest.json`. */
	readonly manifest?: EverythingBundleManifest;
	/** Defaults to `"install"`; `"cached"` is the read-only resolution a query uses. */
	readonly mode?: EverythingBinaryResolutionMode;
}

export type EnsureEverythingBinariesResult =
	| {
			readonly ok: true;
			readonly everythingPath: string;
			readonly esPath: string;
			readonly source: "cached" | "installed";
	  }
	| {
			readonly ok: false;
			readonly reason: "unsupported-platform" | "bundle-missing" | "install-failed" | "not-installed";
			readonly message: string;
	  };

const MANIFEST_FILENAME = "manifest.json";

/**
 * Locate the packaged `vendor/everything` directory by walking up from this
 * module. The source tree (`src/everything/`) and the bundled outputs
 * (`dist/index.js`, `dist/cli/index.js`) sit at different depths.
 */
export function resolveEverythingBundleDir(): string | undefined {
	let dir = dirname(fileURLToPath(import.meta.url));
	for (let depth = 0; depth < 6; depth += 1) {
		const candidate = join(dir, "vendor", "everything");
		if (existsSync(join(candidate, MANIFEST_FILENAME))) return candidate;
		const parent = dirname(dir);
		if (parent === dir) break;
		dir = parent;
	}
	return undefined;
}

export function everythingArch(arch: string): EverythingArch | undefined {
	if (arch === "x64" || arch === "arm64") return arch;
	return undefined;
}

/** Deterministic cache location of the extracted binaries for one workspace. */
export function everythingBinaryPaths(
	root: string,
	version: string,
	arch: EverythingArch,
): { readonly everythingPath: string; readonly esPath: string } {
	const dir = join(root, ".autorag", "everything", version, arch);
	return { everythingPath: join(dir, "everything.exe"), esPath: join(dir, "es.exe") };
}

export function loadEverythingBundleManifest(bundleDir: string): EverythingBundleManifest {
	return JSON.parse(readFileSync(join(bundleDir, MANIFEST_FILENAME), "utf8")) as EverythingBundleManifest;
}

/**
 * Resolve the bundled Everything + ES binaries for this Windows host. Never
 * throws: every failure is reported with its underlying message. `mode:
 * "install"` (default) extracts and verifies them from the bundled ZIPs when
 * the workspace cache is missing or invalid; `mode: "cached"` is read-only and
 * reports `not-installed` instead of writing so a query can only use verified,
 * already-extracted binaries.
 */
export async function ensureEverythingBinaries(
	options: EnsureEverythingBinariesOptions,
): Promise<EnsureEverythingBinariesResult> {
	const platform = options.platform ?? process.platform;
	const rawArch = options.arch ?? process.arch;
	if (platform !== "win32") {
		return {
			ok: false,
			reason: "unsupported-platform",
			message: `Everything is Windows-only; this host is ${platform}.`,
		};
	}
	const arch = everythingArch(rawArch);
	const bundleDir = options.bundleDir ?? resolveEverythingBundleDir();
	if (bundleDir === undefined) {
		return {
			ok: false,
			reason: "bundle-missing",
			message: `Bundled Everything assets (vendor/everything/${MANIFEST_FILENAME}) were not found above ${dirname(fileURLToPath(import.meta.url))}.`,
		};
	}
	let manifest: EverythingBundleManifest;
	try {
		manifest = options.manifest ?? loadEverythingBundleManifest(bundleDir);
	} catch (error) {
		return { ok: false, reason: "bundle-missing", message: errorMessage(error) };
	}
	const assets = arch === undefined ? undefined : manifest.architectures[arch];
	if (arch === undefined || assets === undefined) {
		return {
			ok: false,
			reason: "unsupported-platform",
			message: `No bundled Everything build for Windows architecture ${rawArch}; bundled: ${Object.keys(manifest.architectures).join(", ")}.`,
		};
	}

	const { everythingPath, esPath } = everythingBinaryPaths(options.root, manifest.everythingVersion, arch);
	const everythingCached = fileHasSha256(everythingPath, assets.everythingSha256);
	const esCached = fileHasSha256(esPath, assets.esSha256);
	if (everythingCached && esCached) {
		return { ok: true, everythingPath, esPath, source: "cached" };
	}
	if ((options.mode ?? "install") === "cached") {
		const state =
			everythingCached || esCached
				? "are incomplete or do not match their pinned SHA-256 digests"
				: "are not installed";
		return {
			ok: false,
			reason: "not-installed",
			message: `Everything ${manifest.everythingVersion} (${arch}) binaries ${state} under ${dirname(everythingPath)}; run \`autorag refresh\` to (re)install and index them.`,
		};
	}
	try {
		mkdirSync(dirname(everythingPath), { recursive: true });
		const everythingBytes = await extractVerified(
			join(bundleDir, assets.everythingArchive),
			assets.everythingArchiveSha256,
			assets.everythingMember,
			assets.everythingSha256,
		);
		const esBytes = await extractVerified(
			join(bundleDir, assets.esArchive),
			assets.esArchiveSha256,
			assets.esMember,
			assets.esSha256,
		);
		writeAtomic(everythingPath, everythingBytes);
		writeAtomic(esPath, esBytes);
	} catch (error) {
		return { ok: false, reason: "install-failed", message: errorMessage(error) };
	}
	return { ok: true, everythingPath, esPath, source: "installed" };
}

async function extractVerified(
	archivePath: string,
	archiveSha256: string,
	member: string,
	memberSha256: string,
): Promise<Buffer> {
	const archive = readFileSync(archivePath);
	const actualArchive = sha256(archive);
	if (actualArchive !== archiveSha256) {
		throw new Error(`${archivePath} SHA-256 ${actualArchive} does not match pinned ${archiveSha256}`);
	}
	const zip = await JSZip.loadAsync(archive);
	const entry = zip.file(member);
	if (entry === null) throw new Error(`${archivePath} does not contain ${member}`);
	const bytes = await entry.async("nodebuffer");
	const actualMember = sha256(bytes);
	if (actualMember !== memberSha256) {
		throw new Error(`${member} from ${archivePath} SHA-256 ${actualMember} does not match pinned ${memberSha256}`);
	}
	return bytes;
}

function fileHasSha256(path: string, expected: string): boolean {
	return existsSync(path) && sha256(readFileSync(path)) === expected;
}

function writeAtomic(path: string, bytes: Buffer): void {
	const temp = `${path}.${process.pid}.tmp`;
	try {
		writeFileSync(temp, bytes);
		renameSync(temp, path);
	} finally {
		rmSync(temp, { force: true });
	}
}

function sha256(bytes: Buffer): string {
	return createHash("sha256").update(bytes).digest("hex");
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
