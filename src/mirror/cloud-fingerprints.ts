import { existsSync, readFileSync } from "node:fs";
import { extname, join } from "node:path";
import type { ExactDuplicateFingerprint } from "../dupey/filter.ts";
import { writeFileAtomic } from "./atomic-write.ts";

export const CLOUD_FINGERPRINTS_FILE = "cloud-fingerprints.json";

/**
 * A cloud or network file's dupey identity, taken once while the file was on disk during its single download
 * (or its single read over the network) and reused afterwards. This is what lets the exact-duplicate exclusion
 * group such a file with a scanned copy without reading the released placeholder or the share again.
 */
export interface CloudFingerprint {
	/** The mirror's own source identity at fingerprint time (`size:mtimeNs`), so an unchanged file is not re-read. */
	readonly sourceVersion: string;
	/** Exact hash; equals `content_hash` from `dupey scan --json` (dupey's file-byte hash when it extracts no text). */
	readonly exact: string;
	/** The document's own last-modified time, or `null` when dupey records none. */
	readonly modified: string | null;
	readonly revision: string | null;
}

/**
 * AutoRAG-owned store of cloud fingerprints, keyed by absolute source path. Not part of the MinSync stage schema.
 * Version 2 holds hashes from dupey 0.2.0 or later (the floor AutoRAG enforces); version 1 was written by 0.1.2.
 */
export interface CloudFingerprintStore {
	readonly version: 2;
	readonly entries: Readonly<Record<string, CloudFingerprint>>;
}

export function cloudFingerprintStorePath(root: string): string {
	return join(root, ".autorag", CLOUD_FINGERPRINTS_FILE);
}

export function emptyCloudFingerprints(): CloudFingerprintStore {
	return { version: 2, entries: {} };
}

/**
 * A stored entry, or `undefined` when it is malformed or its hash no longer matches what dupey computes now.
 * A version-1 entry was hashed by dupey 0.1.2, which disagrees with 0.2.0 on every HWP file and on files with no
 * extracted text (`chars` 0); those are dropped, so the file takes no part in exact-duplicate exclusion until it
 * changes and is fingerprinted again. Every other version-1 hash is unchanged and kept.
 */
function readCloudFingerprint(sourcePath: string, value: unknown, storeVersion: unknown): CloudFingerprint | undefined {
	if (typeof value !== "object" || value === null) return undefined;
	const record = value as Record<string, unknown>;
	const { sourceVersion, exact, modified, revision } = record;
	if (
		typeof sourceVersion !== "string" ||
		typeof exact !== "string" ||
		(modified !== null && typeof modified !== "string") ||
		(revision !== null && typeof revision !== "string")
	) {
		return undefined;
	}
	if (storeVersion !== 2 && (record.chars === 0 || extname(sourcePath).toLowerCase() === ".hwp")) return undefined;
	return { sourceVersion, exact, modified, revision };
}

/** Load the stored fingerprints; a missing or unreadable store reads as empty, never an error. */
export function loadCloudFingerprints(root: string): CloudFingerprintStore {
	const path = cloudFingerprintStorePath(root);
	if (!existsSync(path)) return emptyCloudFingerprints();
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
		if (typeof parsed !== "object" || parsed === null || !("entries" in parsed)) return emptyCloudFingerprints();
		const { entries } = parsed;
		if (typeof entries !== "object" || entries === null) return emptyCloudFingerprints();
		const storeVersion = "version" in parsed ? parsed.version : undefined;
		const valid: Record<string, CloudFingerprint> = {};
		for (const [sourcePath, value] of Object.entries(entries)) {
			const fingerprint = readCloudFingerprint(sourcePath, value, storeVersion);
			if (fingerprint !== undefined) valid[sourcePath] = fingerprint;
		}
		return { version: 2, entries: valid };
	} catch {
		return emptyCloudFingerprints();
	}
}

export function saveCloudFingerprints(root: string, entries: Readonly<Record<string, CloudFingerprint>>): void {
	writeFileAtomic(cloudFingerprintStorePath(root), `${JSON.stringify({ version: 2, entries }, null, 2)}\n`);
}

/** View the store as the exclusion merge's fingerprint map (absolute path to exact hash and modified time). */
export function cloudFingerprintExclusions(
	store: CloudFingerprintStore,
): ReadonlyMap<string, ExactDuplicateFingerprint> {
	const map = new Map<string, ExactDuplicateFingerprint>();
	for (const [sourcePath, fingerprint] of Object.entries(store.entries)) {
		map.set(sourcePath, { exact: fingerprint.exact, modified: fingerprint.modified });
	}
	return map;
}
