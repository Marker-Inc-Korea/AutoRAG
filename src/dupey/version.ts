/**
 * dupey version policy.
 *
 * AutoRAG groups files by dupey's exact hash and stores cloud files' hashes across refreshes, so it must refuse a
 * binary whose hashes would disagree with the ones it stores instead of silently grouping by them.
 */

/**
 * Oldest `dupey` AutoRAG will run. Measured against 0.1.2: 0.1.2 misreads HWP 5.x record lengths, so an HWP
 * hashes incomplete or empty text (11,199 extracted characters with 0.2.0 came out as 45 with 0.1.2, and other
 * files as 0), and its `fingerprint` hashes empty text where `scan` hashes the file bytes. From 0.2.0 an HWP
 * hashes its full body text, and a file with no extracted text hashes its bytes in both `scan` and `fingerprint`.
 */
export const DUPEY_MIN_VERSION = "0.2.0";

const VERSION_LINE = /^dupey\s+(\d+\.\d+\.\d+)(?:[-+][^\s]*)?\s*$/mu;

/** Version from `dupey --version` output, or `undefined` when the output is not a dupey version line. */
export function parseDupeyVersion(output: string): string | undefined {
	return VERSION_LINE.exec(output)?.[1];
}

/** True when `version` is at or above {@link DUPEY_MIN_VERSION}. Numeric, per component. */
export function isSupportedDupeyVersion(version: string): boolean {
	const have = version.split(".").map((part) => Number.parseInt(part, 10));
	const need = DUPEY_MIN_VERSION.split(".").map((part) => Number.parseInt(part, 10));
	for (let index = 0; index < need.length; index += 1) {
		const left = have[index] ?? 0;
		const right = need[index] ?? 0;
		if (left !== right) return left > right;
	}
	return true;
}
