import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { basename, join, normalize } from "node:path";
import { abortReason } from "../process/abort-reason.ts";
import { type ChunkLocation, locateChunk } from "../retrieval/chunk-location.ts";
import { OVERFETCH_ABSOLUTE_CAP, type RecordCut } from "../retrieval/record-cut.ts";
import { matchesVirtualPathScope } from "../retrieval/scope.ts";
import { describeRetrievalError } from "../retrieval/skip.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { isMinSyncBinaryMissing, resolveMinSyncBinary } from "./binary.ts";
import type { MinSyncQueryMode } from "./client.ts";
import { MinSyncClient } from "./client.ts";
import { isSparseOnlyEmbedder } from "./embedder-config.ts";
import type { EnsureMinSyncBinaryOptions } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncProgress, MinSyncSyncResult } from "./types.ts";
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
	/**
	 * Fuse deeper and cut the hits by document, so one long document cannot take every one of `topK` slots
	 * (see {@link RecordCut}). Unset asks MinSync for `topK` and cuts by chunk.
	 */
	readonly recordCut?: RecordCut;
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
	private readonly recordCut: RecordCut | undefined;

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
		this.recordCut = options.recordCut;
		// A sparse-only index has no dense vectors, so MinSync answers only `--mode sparse` for it.
		this.mode = isSparseOnlyEmbedder(options.embedder) ? "sparse" : (options.mode ?? "vector");
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "minsync",
			type: this.mode,
			description:
				this.mode === "sparse"
					? "MinSync-backed SPLADE sparse retrieval over parsed markdown mirrors"
					: this.mode === "hybrid"
						? "MinSync-backed hybrid retrieval over parsed markdown mirrors"
						: "MinSync-backed semantic vector retrieval over parsed markdown mirrors",
			status: "active",
			capabilities: [
				...(this.mode === "sparse" ? ["lexical"] : this.mode === "hybrid" ? ["semantic", "lexical"] : ["semantic"]),
				"incremental",
				"parsed-mirrors",
				"virtual-paths",
			],
		};
	}

	async sync(
		force = false,
		signal?: AbortSignal,
		onProgress?: (progress: MinSyncSyncProgress) => void,
	): Promise<MinSyncSyncResult> {
		return runWorkspaceSync(this.workspacePath, () => this.syncUnlocked(force, signal, onProgress));
	}

	private async syncUnlocked(
		force: boolean,
		signal: AbortSignal | undefined,
		onProgress?: (progress: MinSyncSyncProgress) => void,
	): Promise<MinSyncSyncResult> {
		if (signal?.aborted) throw abortReason(signal);
		const binaryPath = await this.resolveBinary();
		const staging = syncMinSyncWorkspace(this.root, {
			workspacePath: this.workspacePath,
			minSyncBinaryPath: binaryPath,
		});
		const withExcluded = (result: MinSyncSyncResult): MinSyncSyncResult =>
			staging.excluded.length === 0 ? result : { ...result, stagingExcluded: staging.excluded };
		const client = new MinSyncClient({
			binaryPath,
			workspacePath: this.workspacePath,
			embedder: this.embedder,
			maxChunkSize: this.maxChunkSize,
		});
		return withExcluded(await client.sync(force, signal, onProgress));
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
		const scopedK = options.scope ? Math.min(Math.max(topK * 5, topK + 20), this.scopedTopK) : topK;
		// The deeper fusion widens whatever the scope already asked for; the cap bounds only the extra depth.
		const queryK =
			this.recordCut === undefined ? scopedK : this.recordCut.fusedLimit(scopedK, OVERFETCH_ABSOLUTE_CAP);
		const binaryPath = await this.resolveBinary({ install: false });
		const byPath = buildMinSyncPathMap(this.root, this.workspacePath, binaryPath);
		// Queries never install: they use an existing binary or fail loudly.
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
			hits = await client.query(query, queryK, "sparse", options.signal);
			// Hybrid is the only MinSync method searched, so a semantic half that is down (embedder
			// unreachable, stale index) would vanish from the result unless the pipeline is told.
			options.onFallback?.({
				method: this.describe().name,
				fallback: "sparse",
				reason: describeRetrievalError(error),
			});
		}
		const results: RetrievalResult[] = [];
		const mirrorTexts = new Map<string, string | undefined>();
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry || !matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			// `path` is the parsed mirror, shared by every chunk of one document, so
			// MinSync's per-chunk `docId` is what keeps distinct passages distinct.
			// Falling back to the mirror basename would give them one identity and the
			// merger would treat them as the same evidence.
			const chunkDiscriminator = hit.docId ?? basename(hit.path);
			const chunkId = `minsync:${entry.virtualPath}:${chunkDiscriminator}`;
			const location = locateChunkInMirror(mirrorTexts, entry, hit.text);
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
					...(location !== undefined
						? {
								lineNumber: location.lineStart,
								lineEnd: location.lineEnd,
								...(location.section !== undefined ? { section: location.section } : {}),
							}
						: {}),
				},
			});
			// With a record cut the whole in-scope pool is collected, then cut by document below.
			if (this.recordCut === undefined && results.length >= topK) break;
		}
		return this.recordCut === undefined ? results : this.recordCut.cut(results, topK, (result) => result.source);
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
	private readonly sparseOnly: boolean;

	constructor(options: MinSyncHybridMethodOptions) {
		super({ ...options, mode: "hybrid" });
		this.sparseOnly = isSparseOnlyEmbedder(options.embedder);
	}

	override describe(): RetrievalMethodDescriptor {
		// The registered name stays `hybrid` so tools and the skip report keep one MinSync surface; a sparse-only
		// index ranks by SPLADE alone and says so.
		return this.sparseOnly
			? {
					name: "hybrid",
					type: "sparse",
					description: "MinSync-backed SPLADE sparse retrieval over shared CDC chunks (sparse-only index)",
					status: "active",
					capabilities: ["lexical", "incremental", "parsed-mirrors", "virtual-paths"],
				}
			: {
					name: "hybrid",
					type: "hybrid",
					description: "MinSync-backed hybrid SPLADE sparse + dense vector retrieval over shared CDC chunks",
					status: "active",
					capabilities: ["semantic", "lexical", "incremental", "parsed-mirrors", "virtual-paths"],
				};
	}
}

/** Parsed mirrors bigger than this are not searched for a chunk's line range. */
const MAX_MIRROR_BYTES = 16 * 1024 * 1024;

/**
 * Where `chunk` sits in the parsed markdown MinSync indexed for `entry`. The
 * staged copy is what MinSync chunked; the parsed mirror is the same text and is
 * the fallback before the first sync. Undefined when the document cannot be read
 * or the chunk is not found in exactly one place, so a line is never guessed.
 */
function locateChunkInMirror(
	cache: Map<string, string | undefined>,
	entry: { readonly minSyncPath: string; readonly parsedOutputPath: string },
	chunk: string,
): ChunkLocation | undefined {
	let text: string | undefined;
	const path = existsSync(entry.minSyncPath) ? entry.minSyncPath : entry.parsedOutputPath;
	if (cache.has(path)) {
		text = cache.get(path);
	} else {
		try {
			text = statSync(path).size <= MAX_MIRROR_BYTES ? readFileSync(path, "utf8") : undefined;
		} catch {
			text = undefined;
		}
		cache.set(path, text);
	}
	return text === undefined ? undefined : locateChunk(text, chunk);
}
