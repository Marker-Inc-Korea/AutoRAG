import { existsSync } from "node:fs";
import { basename, extname, join } from "node:path";
import { resolveMinSyncBinary } from "../../../minsync/binary.ts";
import type { MinSyncQueryMode } from "../../../minsync/client.ts";
import { MinSyncClient } from "../../../minsync/client.ts";
import { MinSyncRequiredError } from "../../../minsync/errors.ts";
import type { MinSyncVectorMethodOptions } from "../../../minsync/method.ts";
import { runWorkspaceQuery, runWorkspaceSync } from "../../../minsync/workspace-lock.ts";
import type { RetrievalResult } from "../../../retrieval/types.ts";
import { boundDiagnosticText } from "../../connector.ts";
import { matchesDatasourceScope } from "../../scope.ts";
import type { DatasourceDiagnostic, DatasourceDiagnosticCode } from "../../types.ts";
import {
	OBSIDIAN_DATASOURCE_ID,
	obsidianMinSyncWorkspace,
	obsidianNoteSource,
	obsidianStagingStatePath,
} from "./paths.ts";
import { listVaultNotes, loadStagingState, stageNotes, VaultUnavailableError } from "./staging.ts";

/** The MinSync settings a vault shares with the rest of the install: binary, embedder, chunking. */
export type ObsidianMinSyncOptions = Pick<
	MinSyncVectorMethodOptions,
	"binaryPath" | "autoInstall" | "installer" | "embedder" | "maxChunkSize"
>;

export interface ObsidianVaultIndexOptions {
	readonly instanceId: string;
	readonly vaultPath: string | undefined;
	readonly workspaceRoot: string | undefined;
	readonly minSync?: ObsidianMinSyncOptions;
}

export type ObsidianSearchMode = "bm25" | "semantic";

export interface ObsidianSearchOptions {
	readonly topK?: number;
	readonly scope?: string;
	readonly signal?: AbortSignal;
}

export type VaultSyncOutcome =
	| { readonly ok: true; readonly notes: number; readonly diagnostics: readonly DatasourceDiagnostic[] }
	| { readonly ok: false; readonly code: DatasourceDiagnosticCode; readonly message: string };

const DEFAULT_TOP_K = 20;
/** A folder scope cannot be pushed into MinSync, so the fetch grows until enough in-scope notes turn up. */
const MIN_SCOPED_FETCH = 50;
const MAX_SCOPED_FETCH = 1000;
const QUERY_MODES: Readonly<Record<ObsidianSearchMode, MinSyncQueryMode>> = { bm25: "bm25", semantic: "vector" };

/**
 * One vault, indexed by MinSync in a workspace of its own. Notes are staged under the workspace's
 * `files/` folder (see `staging.ts`), `minsync sync` embeds and indexes them incrementally, and a
 * query runs against that index and maps each chunk back to its note.
 */
export class ObsidianVaultIndex {
	private readonly instanceId: string;
	private readonly vaultPath: string | undefined;
	private readonly workspaceRoot: string | undefined;
	private readonly minSync: ObsidianMinSyncOptions;

	constructor(options: ObsidianVaultIndexOptions) {
		this.instanceId = options.instanceId;
		this.vaultPath = options.vaultPath;
		this.workspaceRoot = options.workspaceRoot;
		this.minSync = options.minSync ?? {};
	}

	/** Mirror the vault's notes into the MinSync workspace and sync it. Never throws except on abort. */
	sync(signal?: AbortSignal): Promise<VaultSyncOutcome> {
		const { vaultPath, workspaceRoot } = this;
		if (vaultPath === undefined) {
			return Promise.resolve(
				failure("datasource-unavailable", "no vaultPath is configured for this Obsidian datasource"),
			);
		}
		if (workspaceRoot === undefined) {
			return Promise.resolve(
				failure("datasource-unavailable", "no workspace is configured to hold the vault index"),
			);
		}
		const workspace = obsidianMinSyncWorkspace(workspaceRoot, this.instanceId);
		// The whole run holds the workspace, listing included, so a query issued while it runs is refused
		// instead of reading notes that are half staged.
		return runWorkspaceSync(workspace, () => this.syncLocked(vaultPath, workspaceRoot, workspace, signal));
	}

	private async syncLocked(
		vaultPath: string,
		workspaceRoot: string,
		workspace: string,
		signal: AbortSignal | undefined,
	): Promise<VaultSyncOutcome> {
		try {
			signal?.throwIfAborted();
			const listing = await listVaultNotes(vaultPath);
			// A vault that lists nothing is far more likely moved, unmounted, or unreadable than emptied on
			// purpose; syncing it would wipe the index, so the previous one is kept.
			if (listing.notes.size === 0) {
				return failure("datasource-empty", "the Obsidian vault has no markdown notes; the previous index was kept");
			}
			const binaryPath = await resolveMinSyncBinary({
				root: workspaceRoot,
				binaryPath: this.minSync.binaryPath,
				autoInstall: this.minSync.autoInstall,
				installer: this.minSync.installer,
			});
			const staged = stageNotes(
				join(workspace, "files"),
				obsidianStagingStatePath(workspaceRoot, this.instanceId),
				listing,
			);
			const client = new MinSyncClient({
				binaryPath,
				workspacePath: workspace,
				embedder: this.minSync.embedder,
				maxChunkSize: this.minSync.maxChunkSize,
			});
			const result = await client.sync(false, signal);
			if (!result.ok) {
				const detail = result.diagnostic?.message ?? result.reason ?? "MinSync sync failed";
				const code =
					result.diagnostic?.code === "embedder-unavailable"
						? "datasource-unavailable"
						: "datasource-index-failed";
				return failure(code, `${result.reason ?? "sync-failed"}: ${detail}`);
			}
			const diagnostics: DatasourceDiagnostic[] = [];
			const warn = (message: string): void => {
				diagnostics.push({
					code: "datasource-index-partial",
					severity: "warning",
					message: boundDiagnosticText(message),
					instanceId: this.instanceId,
					source: OBSIDIAN_DATASOURCE_ID,
				});
			};
			if (listing.unreadableFolders.length > 0) {
				warn(
					`${listing.unreadableFolders.length} vault folder(s) could not be read; their notes keep their previous index`,
				);
			}
			if (listing.collisions.length > 0) {
				warn(`${listing.collisions.length} note(s) share a normalized path with another note and were not indexed`);
			}
			if (staged.failed.length > 0) warn(`${staged.failed.length} note(s) could not be copied for indexing`);
			if ((result.filesFailed?.length ?? 0) > 0) {
				warn(`MinSync skipped ${result.filesFailed?.length} note(s); the next refresh retries them`);
			}
			return { ok: true, notes: staged.staged, diagnostics };
		} catch (error) {
			if (signal?.aborted) throw error;
			if (error instanceof VaultUnavailableError || error instanceof MinSyncRequiredError) {
				return failure("datasource-unavailable", error.message);
			}
			return failure("datasource-index-failed", error instanceof Error ? error.message : String(error));
		}
	}

	/** Search the indexed vault. Rejects, with the reason, when the vault was never indexed or a sync is running. */
	async query(
		mode: ObsidianSearchMode,
		text: string,
		options: ObsidianSearchOptions = {},
	): Promise<RetrievalResult[]> {
		const query = text.trim();
		if (query.length === 0) return [];
		const workspaceRoot = this.workspaceRoot;
		if (workspaceRoot === undefined) throw new Error("no workspace is configured to hold the vault index");
		const workspace = obsidianMinSyncWorkspace(workspaceRoot, this.instanceId);
		return runWorkspaceQuery(workspace, () => this.queryLocked(mode, query, options, workspaceRoot, workspace));
	}

	private async queryLocked(
		mode: ObsidianSearchMode,
		query: string,
		options: ObsidianSearchOptions,
		workspaceRoot: string,
		workspace: string,
	): Promise<RetrievalResult[]> {
		if (!existsSync(join(workspace, ".minsync", "cursor.json"))) {
			throw new Error(
				`The Obsidian vault "${this.instanceId}" is not indexed yet: run \`autorag refresh --method datasources\` first.`,
			);
		}
		// Queries never install: they use an existing binary or fail, so a turn never waits on a download.
		const binaryPath = await resolveMinSyncBinary({
			root: workspaceRoot,
			binaryPath: this.minSync.binaryPath,
			autoInstall: this.minSync.autoInstall,
			installer: this.minSync.installer,
			install: false,
		});
		const client = new MinSyncClient({
			binaryPath,
			workspacePath: workspace,
			embedder: this.minSync.embedder,
			maxChunkSize: this.minSync.maxChunkSize,
		});
		const staged = loadStagingState(obsidianStagingStatePath(workspaceRoot, this.instanceId));
		const topK = options.topK ?? DEFAULT_TOP_K;
		const scope = options.scope?.normalize("NFC");
		let fetch = scope === undefined ? topK : Math.max(topK * 5, MIN_SCOPED_FETCH);
		for (;;) {
			const hits = await client.query(query, fetch, QUERY_MODES[mode], options.signal);
			const results: RetrievalResult[] = [];
			for (const hit of hits) {
				const relativePath = hit.path.startsWith("files/")
					? hit.path.slice("files/".length).normalize("NFC")
					: undefined;
				const note = relativePath === undefined ? undefined : staged.entries[relativePath];
				if (relativePath === undefined || note === undefined) continue;
				const source = obsidianNoteSource(this.instanceId, relativePath);
				if (!matchesDatasourceScope(source, scope)) continue;
				results.push({
					id: `obsidian:${this.instanceId}:${relativePath}:${hit.docId ?? results.length}`,
					content: hit.text,
					source,
					score: hit.score,
					metadata: {
						method: `obsidian-${mode}`,
						datasourceId: OBSIDIAN_DATASOURCE_ID,
						instanceId: this.instanceId,
						mode,
						file: relativePath,
						title: basename(relativePath, extname(relativePath)),
						path: note.sourcePath,
					},
				});
				if (results.length >= topK) break;
			}
			// MinSync cannot filter by folder, so a narrow scope may find too few notes in the first window.
			if (results.length >= topK || scope === undefined || hits.length < fetch || fetch >= MAX_SCOPED_FETCH) {
				return results;
			}
			fetch = Math.min(fetch * 2, MAX_SCOPED_FETCH);
		}
	}
}

function failure(code: DatasourceDiagnosticCode, message: string): VaultSyncOutcome {
	return { ok: false, code, message: boundDiagnosticText(message) };
}
