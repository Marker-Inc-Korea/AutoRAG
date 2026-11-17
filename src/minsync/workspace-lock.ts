import { existsSync } from "node:fs";
import { join } from "node:path";

/**
 * One workspace, two kinds of operation.
 *
 * A sync rewrites the index and the staged files, so syncs for one workspace queue behind each other
 * instead of colliding on MinSync's lock ("another sync is in progress"). Everything AutoRAG runs against
 * one workspace (the local-file vector and hybrid methods, or one Obsidian vault) goes through here.
 *
 * A query does not wait for a sync and is not refused by one. MinSync's `query` takes no lock: it reads the
 * store at the table version that is current when it opens, and a sync prunes only versions older than ten
 * minutes. So a query beside a refresh answers from what is committed, and a first sync's committed batches
 * are final (its dense rows and their sparse vectors), so they answer too. That is what keeps search
 * available while an index is built or refreshed, and lets a question be proven against a first index
 * before it ends.
 *
 * The one refusal is an index with nothing to read yet; see {@link runWorkspaceQuery}.
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
 * Whether a query has anything to read: a sync has completed (`cursor.json`), or a first sync has opened its
 * transaction (`txn.json`), after which every committed batch is searchable. MinSync decides what the index
 * holds; this only decides whether asking it is worthwhile. A forced rebuild (`init --force`) moves both
 * files and the store aside, which is the moment a query has nothing to read.
 */
function hasReadableIndex(workspacePath: string): boolean {
	const state = join(workspacePath, ".minsync");
	return existsSync(join(state, "cursor.json")) || existsSync(join(state, "txn.json"));
}

/**
 * Run a query. It runs beside a sync that is pending and answers from what is committed. It is refused only
 * while a sync is pending and the index has nothing to read yet, because there is nothing to answer from.
 */
export async function runWorkspaceQuery<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) {
		if (!hasReadableIndex(workspacePath)) throw new Error(`another sync is in progress (${workspacePath})`);
		return run();
	}
	return withWorkspaceLock(workspacePath, run);
}
