import { existsSync, realpathSync } from "node:fs";
import { basename, join, normalize } from "node:path";
import { abortReason } from "../process/abort-reason.ts";
import { matchesVirtualPathScope } from "../retrieval/scope.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { isMinSyncBinaryMissing, resolveMinSyncBinary } from "./binary.ts";
import type { MinSyncQueryMode } from "./client.ts";
import { MinSyncClient } from "./client.ts";
import type { EnsureMinSyncBinaryOptions } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "./workspace.ts";
import { runWorkspaceQuery, runWorkspaceSync } from "./workspace-lock.ts";

export interface MinSyncVectorMethodOptions {
	readonly root: string;
	/** Test/embedded override; CLI config intentionally does not persist this. */
	readonly binaryPath?: string;
	readonly workspacePath?: string;
	readonly installer?: Omit<EnsureMinSyncBinaryOptions, "root">;
	readonly autoInstall?: boolean;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
	readonly mode?: MinSyncQueryMode;
	/** Default `topK` used when the caller omits one. Default 50. */
	readonly defaultTopK?: number;
	/** Fetch cap applied when a scope narrows the query. Default 100. */
	readonly scopedTopK?: number;
}

export type MinSyncHybridMethodOptions = Omit<MinSyncVectorMethodOptions, "mode">;

export class MinSyncVectorMethod implements RetrievalMethod {
	private readonly root: string;
	private readonly binaryPath: string | undefined;
	private readonly workspacePath: string;
	private readonly installer: Omit<EnsureMinSyncBinaryOptions, "root"> | undefined;
	private readonly autoInstall: boolean;
	private readonly embedder: MinSyncEmbedderConfig | undefined;
	private readonly maxChunkSize: number | undefined;
	private readonly mode: MinSyncQueryMode;
	private readonly defaultTopK: number;
	private readonly scopedTopK: number;

	constructor(options: MinSyncVectorMethodOptions) {
		this.root = options.root;
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(options.root);
		this.installer = options.installer;
		this.autoInstall = options.autoInstall ?? true;
		this.embedder = options.embedder;
		this.maxChunkSize = options.maxChunkSize;
		this.defaultTopK = options.defaultTopK ?? 50;
		this.scopedTopK = options.scopedTopK ?? 100;
		this.mode = options.mode ?? "vector";
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "minsync",
			type: this.mode === "bm25" ? "bm25" : this.mode === "hybrid" ? "hybrid" : "vector",
			description:
				this.mode === "bm25"
					? "MinSync-backed BM25 lexical retrieval over parsed markdown mirrors"
					: this.mode === "hybrid"
						? "MinSync-backed hybrid retrieval over parsed markdown mirrors"
						: "MinSync-backed semantic vector retrieval over parsed markdown mirrors",
			status: "active",
			capabilities: [
				...(this.mode === "bm25" ? ["lexical"] : this.mode === "hybrid" ? ["semantic", "lexical"] : ["semantic"]),
				"incremental",
				"parsed-mirrors",
				"virtual-paths",
			],
		};
	}

	async sync(force = false, signal?: AbortSignal): Promise<MinSyncSyncResult> {
		return runWorkspaceSync(this.workspacePath, () => this.syncUnlocked(force, signal));
	}

	private async syncUnlocked(force: boolean, signal: AbortSignal | undefined): Promise<MinSyncSyncResult> {
		if (signal?.aborted) throw abortReason(signal);
		const staging = syncMinSyncWorkspace(this.root, { workspacePath: this.workspacePath });
		const withExcluded = (result: MinSyncSyncResult): MinSyncSyncResult =>
			staging.excluded.length === 0 ? result : { ...result, stagingExcluded: staging.excluded };
		const binaryPath = await this.resolveBinary();
		const client = new MinSyncClient({
			binaryPath,
			workspacePath: this.workspacePath,
			embedder: this.embedder,
			maxChunkSize: this.maxChunkSize,
		});
		return withExcluded(await client.sync(force, signal));
	}

	/** Report unavailable binaries without treating auto-install as already failed. */
	isBinaryMissing(): boolean {
		return isMinSyncBinaryMissing({ root: this.root, binaryPath: this.binaryPath });
	}

	/** Status distinguishes an explicit missing override from deferred resolution. */
	isExplicitBinaryMissing(): boolean {
		return this.binaryPath !== undefined && !existsSync(this.binaryPath);
	}

	/** A vector index is ready only after MinSync has written its cursor. */
	isReady(): boolean {
		return existsSync(join(this.workspacePath, ".minsync", "cursor.json"));
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		return runWorkspaceQuery(this.workspacePath, () => this.retrieveUnlocked(query, options));
	}

	private async retrieveUnlocked(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const topK = options.topK ?? this.defaultTopK;
		const queryK = options.scope ? Math.min(Math.max(topK * 5, topK + 20), this.scopedTopK) : topK;
		const byPath = buildMinSyncPathMap(this.root, this.workspacePath);
		// Queries never install: they use an existing binary or fail loudly.
		const binaryPath = await this.resolveBinary({ install: false });
		const client = new MinSyncClient({
			binaryPath,
			workspacePath: this.workspacePath,
			embedder: this.embedder,
			maxChunkSize: this.maxChunkSize,
		});
		let hits: readonly MinSyncQueryHit[];
		try {
			hits = await client.query(query, queryK, this.mode, options.signal);
		} catch (error) {
			if (this.mode !== "hybrid" || options.signal?.aborted) throw error;
			hits = await client.query(query, queryK, "bm25", options.signal);
		}
		const results: RetrievalResult[] = [];
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry || !matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			// `path` is the parsed mirror, shared by every chunk of one document, so
			// MinSync's per-chunk `docId` is what keeps distinct passages distinct.
			// Falling back to the mirror basename would give them one identity and the
			// merger would treat them as the same evidence.
			const chunkDiscriminator = hit.docId ?? basename(hit.path);
			const chunkId = `minsync:${entry.virtualPath}:${chunkDiscriminator}`;
			results.push({
				id: chunkId,
				content: hit.text,
				source: realpathSync(normalize(entry.sourcePath)),
				score: hit.score,
				metadata: {
					method: this.mode === "vector" ? "minsync" : `minsync-${this.mode}`,
					virtualPath: entry.virtualPath,
					minsyncChunkId: chunkId,
					minsyncPath: hit.path,
				},
			});
			if (results.length >= topK) break;
		}
		return results;
	}

	/**
	 * Queries pass `install: false` so a user's turn never waits on `cargo install`; only `sync` (refresh)
	 * may install. See {@link resolveMinSyncBinary}.
	 */
	resolveBinary(options: { readonly install?: boolean } = {}): Promise<string> {
		return resolveMinSyncBinary({
			root: this.root,
			binaryPath: this.binaryPath,
			autoInstall: this.autoInstall,
			installer: this.installer,
			...(options.install === undefined ? {} : { install: options.install }),
		});
	}
}

export class MinSyncHybridMethod extends MinSyncVectorMethod {
	constructor(options: MinSyncHybridMethodOptions) {
		super({ ...options, mode: "hybrid" });
	}

	override describe(): RetrievalMethodDescriptor {
		return {
			name: "hybrid",
			type: "hybrid",
			description: "MinSync-backed hybrid BM25+vector retrieval over shared CDC chunks",
			status: "active",
			capabilities: ["semantic", "lexical", "incremental", "parsed-mirrors", "virtual-paths"],
		};
	}
}
