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
const workspaceSyncInFlight = new Map<string, number>();

function withWorkspaceLock<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	const previous = workspaceOperationTails.get(workspacePath) ?? Promise.resolve();
	const result = previous.then(run, run);
	workspaceOperationTails.set(
		workspacePath,
		result.then(
			() => undefined,
			() => undefined,
		),
	);
	return result;
}

/**
 * Run a sync (anything that rewrites the workspace). The in-flight count is raised
 * before this returns, so a query issued right after the call already sees the sync.
 */
export function runWorkspaceSync<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	workspaceSyncInFlight.set(workspacePath, (workspaceSyncInFlight.get(workspacePath) ?? 0) + 1);
	return withWorkspaceLock(workspacePath, run).finally(() => {
		const remaining = (workspaceSyncInFlight.get(workspacePath) ?? 1) - 1;
		if (remaining === 0) workspaceSyncInFlight.delete(workspacePath);
		else workspaceSyncInFlight.set(workspacePath, remaining);
	});
}

/**
 * Whether `workspacePath` holds an index only its first sync has written so far: no sync has
 * completed (no `cursor.json`) and the first one has opened its transaction (`txn.json`), after
 * which every committed batch is searchable. MinSync decides what is in it; this only decides
 * whether the query may run now.
 */
function isFirstSyncReadable(workspacePath: string): boolean {
	const state = join(workspacePath, ".minsync");
	return !existsSync(join(state, "cursor.json")) && existsSync(join(state, "txn.json"));
}

/**
 * Run a query. A first sync does not hold it back: it runs beside the sync and sees the batches committed
 * so far. Any later sync rewrites an index that already answers, so a query is refused while one is pending.
 */
export async function runWorkspaceQuery<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) {
		if (!isFirstSyncReadable(workspacePath)) throw new Error(`another sync is in progress (${workspacePath})`);
		return run();
	}
	return withWorkspaceLock(workspacePath, run);
}
