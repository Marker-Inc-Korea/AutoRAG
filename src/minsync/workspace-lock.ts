import { existsSync } from "node:fs";
import { join } from "node:path";

/**
 * One workspace, two kinds of operation.
 *
 * A sync rewrites the index and the staged files, so syncs for one workspace queue behind each other
 * instead of colliding on MinSync's lock ("another sync is in progress"). Everything AutoRAG runs against
 * one workspace (the local-file vector and hybrid methods, or one Obsidian vault) goes through here.
 *
 * A query does not wait for a sync. MinSync's `query` takes no lock: it reads the store at the table version
 * that is current when it opens, and a sync prunes only versions older than ten minutes. So a query beside a
 * sync of an index that already has a `cursor.json` answers from what is committed (MinSync 0.7.4 checks every
 * such query against the embedder the cursor records), and so do the committed batches of a clean first sync
 * or of a first-build session (final: dense rows and their sparse vectors). That keeps search available while
 * an index is built or refreshed, and lets a question be proven against a first index before it ends. Queries
 * of one workspace run one at a time.
 *
 * Refused: a query while the index has nothing to read yet, and while a sync is pending that follows an
 * interrupted first sync (no cursor to check against, rows that may be another embedder's) or has not started
 * yet and so is not known to be safe; see {@link runWorkspaceQuery}.
 */
const workspaceOperationTails = new Map<string, Promise<unknown>>();
/** Queries of one workspace queue behind each other (one `minsync query` process, hence one set of loaded models, at a time) but not behind a first sync. */
const workspaceQueryTails = new Map<string, Promise<unknown>>();
const workspaceSyncInFlight = new Map<string, number>();
/** Syncs in flight that began as the first sync of an index nothing was ever built for, and that keep what they stored. */
const workspaceFirstSyncInFlight = new Map<string, number>();
/** Syncs in flight that began on an index that already had a `cursor.json`: MinSync checks its embedder against that cursor on every query. */
const workspaceBuiltSyncInFlight = new Map<string, number>();

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

/** Syncs in flight that belong to a first-build session (counted when registered, so one that has not started yet counts too). */
const workspaceSessionSyncs = new Map<string, number>();
/** How many syncs without a session have started on a workspace; a session notes it when it opens. */
const workspaceForeignStarts = new Map<string, number>();
/** The one first-build session open on a workspace, if any. */
const workspaceOpenSession = new Map<string, FirstBuildSession>();
/** What each session saw when it opened, for telling whether a sync from outside it has run since. */
const sessionMarks = new WeakMap<FirstBuildSession, number>();

/**
 * One refresh that builds a workspace's index from nothing. Only the syncs that carry it (`runWorkspaceSync`'s `session` option)
 * count as the refresh's own: each runs with the embedder that wrote the earlier ones, over rows it wrote itself, so a query may read
 * beside every one of them and not only the first. A sync from anywhere else (a second instance, a forced or embedder-switching
 * refresh) carries no session and keeps the refusal, whatever else is running. And once such a sync has started, the session is over:
 * the rows are no longer only the refresh's own, so its later syncs are ordinary ones.
 */
export interface FirstBuildSession {
	readonly workspacePath: string;
	isOpen(): boolean;
	/** Call when the refresh is over; calling it again does nothing. */
	end(): void;
}

/**
 * Open a session when `workspacePath` holds nothing a refresh could be resuming: no `cursor.json`, and none of what an interrupted
 * first sync leaves behind, and no sync of any caller registered or running (one that is still preparing leaves the files looking
 * clean while it is about to write them). `undefined` when the workspace already holds an index (or an interrupted build left rows),
 * which a query is never allowed to read while it is being rewritten, or when another session is open (two refreshes could be
 * building with different embedders).
 */
export function beginFirstBuild(workspacePath: string): FirstBuildSession | undefined {
	if (workspaceOpenSession.has(workspacePath) || (workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) return undefined;
	if (!startsAsCleanFirstSync(workspacePath)) return undefined;
	let open = true;
	const session: FirstBuildSession = {
		workspacePath,
		isOpen: () => open,
		end: () => {
			if (!open) return;
			open = false;
			if (workspaceOpenSession.get(workspacePath) === session) workspaceOpenSession.delete(workspacePath);
		},
	};
	workspaceOpenSession.set(workspacePath, session);
	sessionMarks.set(session, workspaceForeignStarts.get(workspacePath) ?? 0);
	return session;
}

/**
 * Whether a sync that is about to start is the first one of an index that was never built: no `cursor.json`, and none of what an
 * earlier first sync leaves behind (`txn.json`, and the journal `progress.jsonl`, which MinSync creates right after the
 * transaction and removes only when a sync succeeds). Rows from an interrupted run may have been written by another embedder, which
 * MinSync does not check while there is no cursor, so a sync that finds them is not one a query may read beside.
 */
function startsAsCleanFirstSync(workspacePath: string): boolean {
	const state = join(workspacePath, ".minsync");
	return !["cursor.json", "txn.json", "progress.jsonl"].some((name) => existsSync(join(state, name)));
}

/**
 * Whether MinSync would answer a query now from what is committed, given that every sync in flight is a kind a query may read beside.
 * With a `cursor.json` it does, from the last completed sync. Without one only a first sync's open transaction (`txn.json`) is
 * readable (every committed batch is final), and a caller that cannot use that (`firstSyncReadable: false`) is refused. With neither
 * file (the librarian is still installing, staging, or initializing, or a forced rebuild has moved the store aside) `minsync query`
 * answers "never synced", which the caller would misreport, so the query is refused as before.
 */
function indexIsReadable(workspacePath: string, firstSyncReadable: boolean): boolean {
	const state = join(workspacePath, ".minsync");
	if (existsSync(join(state, "cursor.json"))) return true;
	return firstSyncReadable && existsSync(join(state, "txn.json"));
}

function hasCursor(workspacePath: string): boolean {
	return existsSync(join(workspacePath, ".minsync", "cursor.json"));
}

function refusal(workspacePath: string): Error {
	return new Error(`another sync is in progress (${workspacePath})`);
}

/**
 * Run a sync (anything that rewrites the workspace). The in-flight count is raised
 * before this returns, so a query issued right after the call already sees the sync. It starts only once the queries
 * that ran beside an earlier first sync are done, so it never rewrites the index or the config under one.
 */
export function runWorkspaceSync<T>(
	workspacePath: string,
	run: () => Promise<T>,
	options: { readonly session?: FirstBuildSession } = {},
): Promise<T> {
	const session = options.session;
	// Registered as the session's own now, so a query between two syncs of the refresh is not refused; settled again at the start.
	let owned = session !== undefined && session.workspacePath === workspacePath && session.isOpen();
	// A sync that finds a `cursor.json` at registration is a sync of a built index at its start too (the cursor only goes when a sync
	// re-initializes the store, which the start checks again), so a query issued before it starts need not be refused.
	let built = !owned && hasCursor(workspacePath);
	bump(workspaceSyncInFlight, workspacePath, 1);
	if (owned) bump(workspaceSessionSyncs, workspacePath, 1);
	if (built) bump(workspaceBuiltSyncInFlight, workspacePath, 1);
	return withWorkspaceLock(workspacePath, async () => {
		await (workspaceQueryTails.get(workspacePath) ?? Promise.resolve());
		if (owned && session !== undefined) {
			// A sync from outside the session has started since it opened (or the refresh ended): this one cannot rely on rows
			// being the refresh's own, so it is decided like any other sync.
			const foreign = (workspaceForeignStarts.get(workspacePath) ?? 0) !== (sessionMarks.get(session) ?? 0);
			if (foreign || !session.isOpen()) {
				session.end();
				owned = false;
				bump(workspaceSessionSyncs, workspacePath, -1);
			}
		}
		if (!owned) workspaceForeignStarts.set(workspacePath, (workspaceForeignStarts.get(workspacePath) ?? 0) + 1);
		// Decided when this sync really starts, after the ones queued ahead of it have finished. A session's sync is counted already.
		const clean = !owned && startsAsCleanFirstSync(workspacePath);
		// An index that was already built: its sync rewrites it in place, but every query is checked against the cursor's embedder.
		const builtNow = !owned && !clean && hasCursor(workspacePath);
		if (built !== builtNow) bump(workspaceBuiltSyncInFlight, workspacePath, builtNow ? 1 : -1);
		built = builtNow;
		if (clean) bump(workspaceFirstSyncInFlight, workspacePath, 1);
		try {
			return await run();
		} finally {
			if (clean) bump(workspaceFirstSyncInFlight, workspacePath, -1);
		}
	}).finally(() => {
		if (built) bump(workspaceBuiltSyncInFlight, workspacePath, -1);
		bump(workspaceSyncInFlight, workspacePath, -1);
		if (owned) bump(workspaceSessionSyncs, workspacePath, -1);
	});
}

/**
 * Whether every sync in flight for `workspacePath` is one whose committed state a query may read: a clean first sync, a
 * sync of a first-build session, or a sync that started on an index that already had a `cursor.json`. A sync that is queued
 * and has not started yet is none of these (it is classified when it really starts), and neither is one that follows an
 * interrupted first sync, whose rows may be another embedder's.
 */
function onlyReadableSyncsInFlight(workspacePath: string): boolean {
	const all = workspaceSyncInFlight.get(workspacePath) ?? 0;
	const own =
		(workspaceFirstSyncInFlight.get(workspacePath) ?? 0) +
		(workspaceSessionSyncs.get(workspacePath) ?? 0) +
		(workspaceBuiltSyncInFlight.get(workspacePath) ?? 0);
	return all > 0 && all === own;
}

export interface WorkspaceQueryOptions {
	/**
	 * Allow the query beside a first sync, before any sync has completed (the default). A caller whose own state is not
	 * readable until a sync completes (the Obsidian vault checks `cursor.json`) turns this off, so during a first sync it
	 * keeps the accurate "another sync is in progress" instead of a misleading "not indexed yet". It still answers beside
	 * a sync of an index that already has its `cursor.json`.
	 */
	readonly duringFirstSync?: boolean;
}

/**
 * Run a query. It does not wait for a sync: beside a clean first sync, a first-build session, or a sync of an index that
 * already has a `cursor.json`, it runs beside the sync and answers from what is committed, one query at a time. It is refused
 * while another kind of sync is pending (one that follows an interrupted first sync may be changing the embedder, and one
 * that has not started yet is not known to be safe), or while the index has nothing to read yet.
 * The answer is decided again when the query's turn comes, since the sync it was queued beside may have ended meanwhile.
 */
export async function runWorkspaceQuery<T>(
	workspacePath: string,
	run: () => Promise<T>,
	options: WorkspaceQueryOptions = {},
): Promise<T> {
	const allowedBesideSync = (): boolean =>
		onlyReadableSyncsInFlight(workspacePath) && indexIsReadable(workspacePath, options.duringFirstSync !== false);
	if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0) {
		if (!allowedBesideSync()) throw refusal(workspacePath);
		return chain(workspaceQueryTails, workspacePath, () => {
			if ((workspaceSyncInFlight.get(workspacePath) ?? 0) > 0 && !allowedBesideSync())
				return Promise.reject(refusal(workspacePath));
			return run();
		});
	}
	// No sync pending: a query waits for the queries still running beside an earlier first sync, then takes its turn behind the other operations.
	return withWorkspaceLock(workspacePath, async () => {
		await (workspaceQueryTails.get(workspacePath) ?? Promise.resolve());
		return run();
	});
}
