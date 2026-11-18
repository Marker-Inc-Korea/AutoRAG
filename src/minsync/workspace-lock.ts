import { existsSync } from "node:fs";
import { join } from "node:path";

/**
 * MinSync allows one operation per workspace: a second concurrent operation fails
 * with "another sync is in progress" while the first holds its lock. Everything
 * AutoRAG runs against one workspace (the local-file vector and hybrid methods,
 * or one Obsidian vault) goes through here, so operations for one workspace queue
 * instead of colliding, and a query never reads an index a sync is rewriting.
 *
 * One exception: the first sync of a workspace. MinSync's `query` takes no lock and reads a
 * snapshot of the store, and every batch the first sync commits is final (its dense rows and
 * their sparse vectors), so the batches committed so far answer queries while the rest are still
 * being read. That is what lets a first index be searched, and a question be proven against it,
 * before it ends.
 */
const workspaceOperationTails = new Map<string, Promise<unknown>>();
/** Queries of one workspace queue behind each other (one `minsync query` process, hence one set of loaded models, at a time) but not behind a first sync. */
const workspaceQueryTails = new Map<string, Promise<unknown>>();
const workspaceSyncInFlight = new Map<string, number>();
/** Syncs in flight that began as the first sync of an index nothing was ever built for, and that keep what they stored. */
const workspaceFirstSyncInFlight = new Map<string, number>();

function chain<T>(tails: Map<string, Promise<unknown>>, workspacePath: string, run: () => Promise<T>): Promise<T> {
	const previous = tails.get(workspacePath) ?? Promise.resolve();
	const result = previous.then(run, run);
	tails.set(
		workspacePath,
		result.then(
			() => undefined,
			() => undefined,
		),
	);
	return result;
}

function withWorkspaceLock<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	return chain(workspaceOperationTails, workspacePath, run);
}

function bump(counts: Map<string, number>, workspacePath: string, by: 1 | -1): void {
	const next = (counts.get(workspacePath) ?? 0) + by;
	if (next <= 0) counts.delete(workspacePath);
	else counts.set(workspacePath, next);
}

/**
 * Whether a sync that is about to start is the first one of an index that was never built (no `cursor.json`) and a
 * stale `txn.json` of an earlier, interrupted first sync does not exist. A leftover marker means rows from a run that
 * may have used another embedder, which MinSync does not check while there is no cursor, so such a run is not
 * one a query may read beside.
 */
function startsAsCleanFirstSync(workspacePath: string): boolean {
	const state = join(workspacePath, ".minsync");
	return !existsSync(join(state, "cursor.json")) && !existsSync(join(state, "txn.json"));
}

/**
 * Run a sync (anything that rewrites the workspace). The in-flight count is raised
 * before this returns, so a query issued right after the call already sees the sync.
 */
export function runWorkspaceSync<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	bump(workspaceSyncInFlight, workspacePath, 1);
	return withWorkspaceLock(workspacePath, () => {
		// Decided when this sync really starts, after the ones queued ahead of it have finished.
		const clean = startsAsCleanFirstSync(workspacePath);
		if (clean) bump(workspaceFirstSyncInFlight, workspacePath, 1);
		return run().finally(() => {
			if (clean) bump(workspaceFirstSyncInFlight, workspacePath, -1);
		});
	}).finally(() => bump(workspaceSyncInFlight, workspacePath, -1));
}

/** Whether every sync in flight for `workspacePath` is a clean first sync, whose committed batches a query may read. */
function onlyCleanFirstSyncsInFlight(workspacePath: string): boolean {
	const all = workspaceSyncInFlight.get(workspacePath) ?? 0;
	return all > 0 && all === (workspaceFirstSyncInFlight.get(workspacePath) ?? 0);
}

export interface WorkspaceQueryOptions {
	/**
	 * Allow the query beside a clean first sync (the default). A caller whose own state is not readable until a sync
	 * completes (the Obsidian vault checks `cursor.json`) turns this off, so it keeps the accurate "another sync is in
	 * progress" instead of a misleading "not indexed yet".
	 */
	readonly duringFirstSync?: boolean;
}

/**
 * Run a query. During the first sync of an index it does not wait for the sync: it runs beside it and sees the
 * batches committed so far, one query at a time. Any later sync rewrites an index that already answers, and a first
 * sync that follows an interrupted one may be changing the embedder, so a query is refused while one is pending.
 */
export async function runWorkspaceQuery<T>(
	workspacePath: string,
	run: () => Promise<T>,
	options: WorkspaceQueryOptions = {},
): Promise<T> {
	if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) {
		if (options.duringFirstSync === false || !onlyCleanFirstSyncsInFlight(workspacePath)) {
			throw new Error(`another sync is in progress (${workspacePath})`);
		}
		return chain(workspaceQueryTails, workspacePath, run);
	}
	return withWorkspaceLock(workspacePath, run);
}
