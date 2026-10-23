/**
 * MinSync allows one operation per workspace: a second concurrent operation fails
 * with "another sync is in progress" while the first holds its lock. Everything
 * AutoRAG runs against one workspace (the local-file vector and hybrid methods,
 * or one Obsidian vault) goes through here, so operations for one workspace queue
 * instead of colliding, and a query never reads an index a sync is rewriting.
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

/** Run a query. It is refused while a sync is pending, because the index is mid-rewrite. */
export async function runWorkspaceQuery<T>(workspacePath: string, run: () => Promise<T>): Promise<T> {
	if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) {
		throw new Error(`another sync is in progress (${workspacePath})`);
	}
	return withWorkspaceLock(workspacePath, run);
}
