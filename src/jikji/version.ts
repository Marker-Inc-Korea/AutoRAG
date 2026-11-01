/**
 * Jikji version policy.
 *
 * AutoRAG talks to the `jikji` CLI through a fixed argv, so it must refuse a binary that
 * cannot accept that argv instead of spawning it and reporting a generic failure.
 */

/**
 * Oldest `jikji` AutoRAG will run. `jikji find` only accepts `--no-background-refresh`
 * from 0.2.0 (0.1.1 exits 2 with `unexpected argument`), and AutoRAG passes it on every find.
 */
export const JIKJI_MIN_VERSION = "0.2.0";

/**
 * Version AutoRAG installs and pins. 0.2.1 is the first 0.2.x with the Windows rooted
 * cache-path rejection (a path such as `\secret` or `C:\x` is no longer joined onto the store).
 */
export const JIKJI_INSTALL_VERSION = "0.2.1";

const VERSION_LINE = /^jikji(?:-cli)?\s+(\d+\.\d+\.\d+)(?:[-+][^\s]*)?\s*$/mu;

/** Version from `jikji --version` output, or `undefined` when the output is not a jikji version line. */
export function parseJikjiVersion(output: string): string | undefined {
	return VERSION_LINE.exec(output)?.[1];
}

function parts(version: string): readonly number[] {
	return version.split(".").map((part) => Number.parseInt(part, 10));
}

/** True when `version` is at or above {@link JIKJI_MIN_VERSION}. Numeric, per component. */
export function isSupportedJikjiVersion(version: string): boolean {
	const have = parts(version);
	const need = parts(JIKJI_MIN_VERSION);
	for (let index = 0; index < need.length; index += 1) {
		const left = have[index] ?? 0;
		const right = need[index] ?? 0;
		if (left !== right) return left > right;
	}
	return true;
}
