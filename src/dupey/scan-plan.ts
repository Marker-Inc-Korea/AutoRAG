import { basename } from "node:path";
import { type CloudFileProbe, type CloudRoot, cloudScope } from "../filesystem/cloud-placeholder.ts";

/** One `dupey scan` to run: the directory and the cloud or network folder names nested in it that must not be walked. */
export interface DupeyScanTarget {
	readonly path: string;
	readonly excludeDirs: readonly string[];
}

export interface DupeyScanPlan {
	readonly targets: readonly DupeyScanTarget[];
	/** Roots inside a cloud or network folder: never scanned, because `dupey scan` opens every file's contents. */
	readonly skipped: readonly string[];
}

/**
 * The machine's cloud and network roots for the roots about to be scanned. A probe that is missing or fails
 * reads as no roots, never an error: an unknown root is scanned as before.
 */
export async function listCloudRoots(
	probe: CloudFileProbe | undefined,
	roots: readonly string[],
): Promise<readonly CloudRoot[]> {
	if (probe === undefined) return [];
	try {
		return await probe.cloudRoots(roots);
	} catch {
		return [];
	}
}

/**
 * Which roots `dupey scan` may walk. A root inside a cloud or network folder is skipped whole and a cloud or
 * network folder nested under a root is excluded by name: `dupey scan` reads file contents, which re-downloads
 * a released placeholder (measured: a OneDrive folder of 1.1k placeholders ran past the 120 s scan timeout and
 * every read failed with ETIMEDOUT) and re-reads a network file over the wire.
 */
export function planDupeyScans(roots: readonly string[], cloudRoots: readonly CloudRoot[]): DupeyScanPlan {
	const targets: DupeyScanTarget[] = [];
	const skipped: string[] = [];
	for (const path of roots) {
		const scope = cloudScope(path, cloudRoots);
		if (scope.insideCloudRoot) {
			skipped.push(path);
			continue;
		}
		targets.push({ path, excludeDirs: scope.nestedCloudRoots.map((root) => basename(root.path)) });
	}
	return { targets, skipped };
}
