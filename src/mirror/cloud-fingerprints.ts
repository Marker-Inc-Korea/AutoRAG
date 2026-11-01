import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
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
	/** Canonical-text hash; equals `content_hash` from `dupey scan --json`. */
	readonly exact: string;
	/** Characters dupey extracted; `0` means no text, so {@link exact} identifies nothing. */
	readonly chars: number;
	/** The document's own last-modified time, or `null` when dupey records none. */
	readonly modified: string | null;
	readonly revision: string | null;
}

/** AutoRAG-owned store of cloud fingerprints, keyed by absolute source path. Not part of the MinSync stage schema. */
export interface CloudFingerprintStore {
	readonly version: 1;
	readonly entries: Readonly<Record<string, CloudFingerprint>>;
}

export function cloudFingerprintStorePath(root: string): string {
	return join(root, ".autorag", CLOUD_FINGERPRINTS_FILE);
}

export function emptyCloudFingerprints(): CloudFingerprintStore {
	return { version: 1, entries: {} };
}

function isCloudFingerprint(value: unknown): value is CloudFingerprint {
	if (typeof value !== "object" || value === null) return false;
	const record = value as Record<string, unknown>;
	return (
		typeof record.sourceVersion === "string" &&
		typeof record.exact === "string" &&
		typeof record.chars === "number" &&
		(record.modified === null || typeof record.modified === "string") &&
		(record.revision === null || typeof record.revision === "string")
	);
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
		const valid: Record<string, CloudFingerprint> = {};
		for (const [sourcePath, fingerprint] of Object.entries(entries)) {
			if (isCloudFingerprint(fingerprint)) valid[sourcePath] = fingerprint;
		}
		return { version: 1, entries: valid };
	} catch {
		return emptyCloudFingerprints();
	}
}

export function saveCloudFingerprints(root: string, entries: Readonly<Record<string, CloudFingerprint>>): void {
	writeFileAtomic(cloudFingerprintStorePath(root), `${JSON.stringify({ version: 1, entries }, null, 2)}\n`);
}

/** View the store as the exclusion merge's fingerprint map (absolute path to exact hash and modified time). */
export function cloudFingerprintExclusions(
	store: CloudFingerprintStore,
): ReadonlyMap<string, ExactDuplicateFingerprint> {
	const map = new Map<string, ExactDuplicateFingerprint>();
	for (const [sourcePath, fingerprint] of Object.entries(store.entries)) {
		map.set(sourcePath, { exact: fingerprint.exact, modified: fingerprint.modified, chars: fingerprint.chars });
	}
	return map;
}
