import { spawnSync } from "node:child_process";
import { statSync } from "node:fs";

/**
 * Oldest `minsync` AutoRAG will run. 0.6.0 replaced the BM25 full-text index with SPLADE sparse retrieval: AutoRAG
 * queries `--mode sparse` and no longer passes `init --language`, which older builds reject. 0.7.1 answers queries
 * from every committed batch, also while the first sync is still running or was interrupted.
 */
export const MINSYNC_MIN_VERSION = "0.7.1";

/** `minsync --version` answers instantly; a binary that does not within this is treated as unprobeable. */
const VERSION_PROBE_TIMEOUT_MS = 5_000;

const VERSION_LINE = /^minsync\s+(\d+\.\d+\.\d+)(?:[-+][^\s]*)?\s*$/mu;

/** A `minsync` that was found but is older than {@link MINSYNC_MIN_VERSION}, so it is never run. */
export interface OutdatedMinSyncBinary {
	readonly path: string;
	readonly version: string;
}

/** Probe results keyed by path, invalidated when the file's size or mtime changes (an upgrade in place). */
const probeCache = new Map<string, { readonly stamp: string; readonly version: string | undefined }>();

/** Version from `minsync --version` output, or `undefined` when the output is not a MinSync version line. */
export function parseMinSyncVersion(output: string): string | undefined {
	return VERSION_LINE.exec(output)?.[1];
}

/** True when `version` is at or above {@link MINSYNC_MIN_VERSION}. Numeric, per component. */
export function isSupportedMinSyncVersion(version: string): boolean {
	const have = version.split(".").map((part) => Number.parseInt(part, 10));
	const need = MINSYNC_MIN_VERSION.split(".").map((part) => Number.parseInt(part, 10));
	for (let index = 0; index < need.length; index += 1) {
		const left = have[index] ?? 0;
		const right = need[index] ?? 0;
		if (left !== right) return left > right;
	}
	return true;
}

/**
 * The version of the `minsync` at `binaryPath` when it is older than {@link MINSYNC_MIN_VERSION}, else undefined.
 * A binary whose `--version` cannot be run or is not a MinSync version line is not called outdated, so a custom
 * build is never locked out; the stage protocol check still guards it.
 */
export function outdatedMinSyncVersion(binaryPath: string): string | undefined {
	let stamp: string;
	try {
		const stat = statSync(binaryPath);
		stamp = `${stat.size}:${stat.mtimeMs}`;
	} catch {
		return undefined;
	}
	const cached = probeCache.get(binaryPath);
	let version: string | undefined;
	if (cached?.stamp === stamp) {
		version = cached.version;
	} else {
		const probed = spawnSync(binaryPath, ["--version"], { encoding: "utf8", timeout: VERSION_PROBE_TIMEOUT_MS });
		version = probed.status === 0 ? parseMinSyncVersion(probed.stdout) : undefined;
		probeCache.set(binaryPath, { stamp, version });
	}
	return version !== undefined && !isSupportedMinSyncVersion(version) ? version : undefined;
}
