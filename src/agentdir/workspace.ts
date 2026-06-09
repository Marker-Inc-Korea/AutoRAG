import { basename, join } from "node:path";
import { type RefreshSummary, Workspace } from "@nomadamas/agentdir";

/** Materialization strategy passed to agentdir on first workspace init. */
export type MaterializationStrategy = "reflink" | "symlink" | "virtual";

/** Default subdirectory (under a project root) where the agentdir workspace lives. */
export const WORKSPACE_SUBDIR = join(".autorag", "workspace");

/** Resolve the on-disk agentdir workspace root for a given project root. */
export function workspaceRoot(root: string): string {
	return join(root, WORKSPACE_SUBDIR);
}

const cache = new Map<string, Workspace>();

/**
 * Open-or-init the agentdir Workspace rooted at `<root>/.autorag/workspace`.
 *
 * Workspaces are cached per resolved path so repeated calls in a session reuse
 * the same handle. Strategy defaults to "reflink" (agentdir falls back to
 * byte-copy on filesystems without copy-on-write support).
 */
export function getWorkspace(root: string, options: { strategy?: MaterializationStrategy } = {}): Workspace {
	const wsRoot = workspaceRoot(root);
	const cached = cache.get(wsRoot);
	if (cached) return cached;

	let ws: Workspace;
	try {
		ws = Workspace.open(wsRoot);
	} catch {
		ws = Workspace.init(wsRoot, options.strategy ?? "reflink");
	}
	cache.set(wsRoot, ws);
	return ws;
}

/** Drop a cached workspace handle (test/teardown helper). */
export function clearWorkspaceCache(): void {
	cache.clear();
}

export interface PlannedMount {
	source: string;
	mount: string;
}

/**
 * Deterministically plan virtual mount points for source directories.
 *
 * Each source maps to `/<basename>`. Basename collisions are resolved with a
 * stable numeric suffix (`-2`, `-3`, ...) over a sorted copy of the inputs, so
 * the same input set always yields the same mount assignment.
 */
export function planMounts(searchPaths: string[]): PlannedMount[] {
	const sorted = [...searchPaths].sort();
	const used = new Set<string>();
	const plan: PlannedMount[] = [];
	for (const source of sorted) {
		const base = basename(source.replace(/[/\\]+$/, "")) || "root";
		let mount = `/${base}`;
		let n = 2;
		while (used.has(mount)) {
			mount = `/${base}-${n}`;
			n += 1;
		}
		used.add(mount);
		plan.push({ source, mount });
	}
	return plan;
}

export interface BootstrapResult {
	mounts: PlannedMount[];
	entriesAdded: number;
}

/**
 * Map source directories into the workspace virtual tree using the deterministic
 * mount plan. Mounts that already exist are skipped, so this is safe to call on
 * every session start. Uses a single `map` per source because agentdir's batch
 * map accepts only files, whereas search paths are directories.
 */
export async function bootstrapMappings(ws: Workspace, searchPaths: string[]): Promise<BootstrapResult> {
	const mounts = planMounts(searchPaths);
	let entriesAdded = 0;
	for (const { source, mount } of mounts) {
		if (await ws.exists(mount)) continue;
		const summary = await ws.map(source, mount);
		entriesAdded += summary.entriesAdded;
	}
	return { mounts, entriesAdded };
}

/**
 * Refresh the workspace against its source roots.
 *
 * With `verifyHashes: false` (default) only mtime+size are compared. With
 * `verifyHashes: true`, files whose mtime+size are unchanged are additionally
 * SHA-256 verified, which detects same-size/same-mtime content swaps (agentdir
 * issue #2 — spoofed mtime, NFS clock skew, coarse FAT32 granularity).
 */
export async function refreshWorkspace(
	ws: Workspace,
	options: { verifyHashes?: boolean } = {},
): Promise<RefreshSummary> {
	return options.verifyHashes ? ws.refreshWithHashVerification(true) : ws.refresh();
}
