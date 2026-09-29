/**
 * Persisted dupey scan results — the app's single source of truth for version
 * stacks. Lives under Electron's userData directory so a restart serves the
 * last scan from disk instead of re-running the dupey CLI.
 */

import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { join } from "node:path";
import type { FsVersionFamily } from "../shared/fs-contract";

export interface StoredVersionFamilies {
	readonly version: 1;
	/** ISO timestamp of the scan that produced this snapshot. */
	readonly scannedAt: string;
	/** Location roots the snapshot covers. */
	readonly locations: readonly string[];
	readonly families: readonly FsVersionFamily[];
}

export interface VersionFamilyStore {
	read(): Promise<StoredVersionFamilies | null>;
	write(value: StoredVersionFamilies): Promise<void>;
}

function isStored(value: unknown): value is StoredVersionFamilies {
	if (typeof value !== "object" || value === null) return false;
	const candidate = value as Partial<StoredVersionFamilies>;
	return (
		candidate.version === 1 &&
		typeof candidate.scannedAt === "string" &&
		Array.isArray(candidate.locations) &&
		Array.isArray(candidate.families)
	);
}

export function createFileVersionFamilyStore(directory: string): VersionFamilyStore {
	const path = join(directory, "version-families.json");
	const tempPath = `${path}.tmp`;
	return {
		async read(): Promise<StoredVersionFamilies | null> {
			try {
				const parsed: unknown = JSON.parse(await readFile(path, "utf8"));
				return isStored(parsed) ? parsed : null;
			} catch {
				// Missing or unreadable snapshot: the service scans and rewrites it.
				return null;
			}
		},
		async write(value: StoredVersionFamilies): Promise<void> {
			await mkdir(directory, { recursive: true });
			await writeFile(tempPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
			await rename(tempPath, path);
		},
	};
}
