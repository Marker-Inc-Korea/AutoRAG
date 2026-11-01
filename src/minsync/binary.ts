import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, join } from "node:path";
import { minSyncBinaryMissingError, minSyncInstallFailedError, minSyncOutdatedError } from "./errors.ts";
import { type EnsureMinSyncBinaryOptions, ensureMinSyncBinary, executableName } from "./installer.ts";
import { type OutdatedMinSyncBinary, outdatedMinSyncVersion } from "./version.ts";

export interface ResolveMinSyncBinaryOptions {
	/** Workspace project root; the cached binary lives under `<root>/.autorag/bin`. */
	readonly root: string;
	/** Test/embedded override; CLI config intentionally does not persist this. */
	readonly binaryPath?: string;
	readonly autoInstall?: boolean;
	readonly installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	/** False for query paths: they use an existing binary or fail, and never wait on `cargo install`. */
	readonly install?: boolean;
}

/**
 * Resolve the `minsync` executable from PATH directories: the first executable that is not older than
 * {@link MINSYNC_MIN_VERSION}. Older ones are skipped and, when `outdated` is given, recorded there.
 */
export function lookupMinSyncInPath(env: NodeJS.ProcessEnv, outdated?: OutdatedMinSyncBinary[]): string | undefined {
	const pathEnv = env.PATH;
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	const execName = executableName(process.platform);
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, execName);
		try {
			accessSync(candidate, constants.X_OK);
		} catch {
			continue; // not executable in this dir
		}
		const version = outdatedMinSyncVersion(candidate);
		if (version === undefined) return candidate;
		outdated?.push({ path: candidate, version });
	}
	return undefined;
}

/**
 * Resolve the MinSync CLI from the user's global PATH first, then the workspace cache, and finally the
 * verified auto-install fallback. A binary older than {@link MINSYNC_MIN_VERSION} is never run: it is skipped on
 * PATH, and a cached one is replaced by the install. Throws {@link MinSyncRequiredError}: MinSync is required, so
 * there is no "no binary" value to degrade on. An explicit `binaryPath` is used as given and never probed.
 */
export async function resolveMinSyncBinary(options: ResolveMinSyncBinaryOptions): Promise<string> {
	if (options.binaryPath && existsSync(options.binaryPath)) return options.binaryPath;
	if (options.binaryPath !== undefined) throw minSyncBinaryMissingError();
	const outdated: OutdatedMinSyncBinary[] = [];
	const pathBinary = lookupMinSyncInPath(process.env, outdated);
	if (pathBinary) return pathBinary;
	const cachedBinary = join(options.root, ".autorag", "bin", executableName(process.platform));
	if (existsSync(cachedBinary)) {
		const version = outdatedMinSyncVersion(cachedBinary);
		if (version === undefined) return cachedBinary;
		outdated.push({ path: cachedBinary, version });
	}
	if ((options.autoInstall ?? true) && options.install !== false) {
		try {
			const installed = await ensureMinSyncBinary({ ...options.installer, root: options.root });
			return installed.binaryPath;
		} catch (error) {
			throw minSyncInstallFailedError(error);
		}
	}
	throw outdated.length > 0 ? minSyncOutdatedError(outdated) : minSyncBinaryMissingError();
}

/** True when no usable `minsync` is resolvable right now without installing one. */
export function isMinSyncBinaryMissing(options: Pick<ResolveMinSyncBinaryOptions, "root" | "binaryPath">): boolean {
	if (options.binaryPath !== undefined) return !existsSync(options.binaryPath);
	if (lookupMinSyncInPath(process.env) !== undefined) return false;
	const cachedBinary = join(options.root, ".autorag", "bin", executableName(process.platform));
	return !existsSync(cachedBinary) || outdatedMinSyncVersion(cachedBinary) !== undefined;
}
