/**
 * dupey version policy.
 *
 * AutoRAG groups files by dupey's exact hash and stores cloud files' hashes across refreshes, so it must refuse a
 * binary whose hashes would disagree with the ones it stores instead of silently grouping by them.
 */

/**
 * Oldest `dupey` AutoRAG will run.
 *
 * 0.2.0 over 0.1.2: 0.1.2 misreads HWP 5.x record lengths, so an HWP hashes incomplete or empty text (11,199
 * extracted characters came out as 45), and its `fingerprint` hashes empty text where `scan` hashes the file bytes.
 *
 * 0.3.1 over 0.2.0 and 0.3.0, measured on this machine's Documents, Downloads and Desktop:
 * - `scan` discarded every directory it could not read, so a root with a folder it was denied looked free of
 *   duplicates. From 0.3.1 that folder is reported in `errors`.
 * - A UTF-16 text file (Notepad's "Unicode") was an extraction error, and a text file with a UTF-8 byte-order
 *   mark hashed the BOM as content, so it never matched an identical copy saved without one (3 of 8,914 `.txt`
 *   files here). From 0.3.1 both hash their text.
 * 0.3.0 added images (decoded pixels plus a perceptual hash) and skipping cloud placeholders; those are the
 * reason `scan` output now also lists image files and image families.
 */
export const DUPEY_MIN_VERSION = "0.3.1";

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
