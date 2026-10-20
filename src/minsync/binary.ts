import { accessSync, constants, existsSync } from "node:fs";
import { delimiter, join } from "node:path";
import { minSyncBinaryMissingError, minSyncInstallFailedError } from "./errors.ts";
import { type EnsureMinSyncBinaryOptions, ensureMinSyncBinary, executableName } from "./installer.ts";

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

/** Resolve the `minsync` executable from PATH directories. Returns the first match or undefined. */
export function lookupMinSyncInPath(env: NodeJS.ProcessEnv): string | undefined {
	const pathEnv = env.PATH;
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	const execName = executableName(process.platform);
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, execName);
		try {
			accessSync(candidate, constants.X_OK);
			return candidate;
		} catch {
			// not executable in this dir, continue
		}
	}
	return undefined;
}

/**
 * Resolve the MinSync CLI from the user's global PATH first, then the workspace cache, and finally the
 * verified auto-install fallback. Throws {@link MinSyncRequiredError}: MinSync is required, so there is
 * no "no binary" value to degrade on.
 */
export async function resolveMinSyncBinary(options: ResolveMinSyncBinaryOptions): Promise<string> {
	if (options.binaryPath && existsSync(options.binaryPath)) return options.binaryPath;
	if (options.binaryPath !== undefined) throw minSyncBinaryMissingError();
	const pathBinary = lookupMinSyncInPath(process.env);
	if (pathBinary) return pathBinary;
	const cachedBinary = join(options.root, ".autorag", "bin", executableName(process.platform));
	if (existsSync(cachedBinary)) return cachedBinary;
	if ((options.autoInstall ?? true) && options.install !== false) {
		try {
			const installed = await ensureMinSyncBinary({ ...options.installer, root: options.root });
			return installed.binaryPath;
		} catch (error) {
			throw minSyncInstallFailedError(error);
		}
	}
	throw minSyncBinaryMissingError();
}

/** True when no `minsync` is resolvable right now without installing one. */
export function isMinSyncBinaryMissing(options: Pick<ResolveMinSyncBinaryOptions, "root" | "binaryPath">): boolean {
	if (options.binaryPath !== undefined) return !existsSync(options.binaryPath);
	if (lookupMinSyncInPath(process.env) !== undefined) return false;
	return !existsSync(join(options.root, ".autorag", "bin", executableName(process.platform)));
}
