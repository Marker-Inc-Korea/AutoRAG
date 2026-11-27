import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { basename, join, normalize } from "node:path";
import { readPagesSidecar } from "../mirror/pages-sidecar.ts";
import type { ParsedPages } from "../parser/pages.ts";
import { abortReason } from "../process/abort-reason.ts";
import { type ChunkLocation, locateChunk } from "../retrieval/chunk-location.ts";
import { OVERFETCH_ABSOLUTE_CAP, type RecordCut } from "../retrieval/record-cut.ts";
import { matchesVirtualPathScope } from "../retrieval/scope.ts";
import { describeRetrievalError } from "../retrieval/skip.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalMissingSource,
	RetrievalOptions,
	RetrievalResult,
} from "../retrieval/types.ts";
import { isMinSyncBinaryMissing, resolveMinSyncBinary } from "./binary.ts";
import type { MinSyncQueryMode } from "./client.ts";
import { MinSyncClient } from "./client.ts";
import { isMultimodalEmbedder, isSparseOnlyEmbedder } from "./embedder-config.ts";
import type { EnsureMinSyncBinaryOptions } from "./installer.ts";
import { minSyncWorkspaceRoot } from "./paths.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncProgress, MinSyncSyncResult } from "./types.ts";
import { buildMinSyncPathMap, syncMinSyncWorkspace } from "./workspace.ts";
import { beginFirstBuild, type FirstBuildSession, runWorkspaceQuery, runWorkspaceSync } from "./workspace-lock.ts";

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

function fileSystemErrorCode(error: unknown): string | undefined {
	const code = error instanceof Error ? (error as NodeJS.ErrnoException).code : undefined;
	return typeof code === "string" ? code : undefined;
}

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

	/**
	 * Mark the start of a refresh that builds this workspace's index from nothing. Syncs run with the returned session (`sync`'s
	 * `session` option) may be read by queries while they run; end it when the refresh is over. `undefined` when an index already
	 * exists (or an interrupted build left rows), which a query is never allowed to read while it is being rewritten.
	 */
	beginFirstBuild(): FirstBuildSession | undefined {
		return beginFirstBuild(this.workspacePath);
	}

	async sync(
		force = false,
		signal?: AbortSignal,
		onProgress?: (progress: MinSyncSyncProgress) => void,
		session?: FirstBuildSession,
	): Promise<MinSyncSyncResult> {
		return runWorkspaceSync(this.workspacePath, () => this.syncUnlocked(force, signal, onProgress), {
			...(session === undefined ? {} : { session }),
		});
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
			stageRawImages: isMultimodalEmbedder(this.embedder),
		});
		const withStaging = (result: MinSyncSyncResult): MinSyncSyncResult => ({
			...result,
			...(staging.excluded.length > 0 ? { stagingExcluded: staging.excluded } : {}),
			...(staging.stagingFailed.length > 0 ? { stagingFailed: staging.stagingFailed } : {}),
		});
		const client = new MinSyncClient({
			binaryPath,
			workspacePath: this.workspacePath,
			embedder: this.embedder,
			maxChunkSize: this.maxChunkSize,
		});
		return withStaging(await client.sync(force, signal, onProgress));
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
		const baseQueryK =
			this.recordCut === undefined ? scopedK : this.recordCut.fusedLimit(scopedK, OVERFETCH_ABSOLUTE_CAP);
		const binaryPath = await this.resolveBinary({ install: false });
		const byPath = buildMinSyncPathMap(this.root, this.workspacePath, binaryPath);
		let hasVisualMirrors = false;
		for (const entry of byPath.values()) {
			if (entry.mirrorKind === "visual" || entry.mirrorKind === "image") {
				hasVisualMirrors = true;
				break;
			}
		}
		// When supplementary visual mirrors exist, add a small buffer so coalescing same-page text+visual hits
		// does not reduce topK without distorting document-level recordCut balance.
		const queryK = hasVisualMirrors ? Math.min(baseQueryK + 8, OVERFETCH_ABSOLUTE_CAP) : baseQueryK;
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
		const missingSources: RetrievalMissingSource[] = [];
		const mirrorTexts = new Map<string, MirrorDocument | undefined>();
		for (const hit of hits) {
			const entry = byPath.get(hit.path);
			if (!entry || !matchesVirtualPathScope(entry.virtualPath, options.scope)) continue;
			// `path` is the parsed mirror, shared by every chunk of one document, so
			// MinSync's per-chunk `docId` is what keeps distinct passages distinct.
			// Falling back to the mirror basename would give them one identity and the
			// merger would treat them as the same evidence.
			// A stale index entry (the file moved or was deleted since the last refresh) must not fail the
			// whole method; skip the hit so later hits can fill topK, and report it below.
			let resolvedSource: string;
			try {
				resolvedSource = realpathSync(normalize(entry.sourcePath));
			} catch (error) {
				const code = fileSystemErrorCode(error);
				if (code !== "ENOENT" && code !== "ENOTDIR") throw error;
				missingSources.push({ virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, code });
				continue;
			}
			const chunkDiscriminator = hit.docId ?? basename(hit.path);
			const chunkId = `minsync:${entry.virtualPath}:${chunkDiscriminator}`;
			const isImageHit = entry.mirrorKind === "image";
			const location = isImageHit
				? undefined
				: locateChunkInMirror(mirrorTexts, entry, withoutHeading(hit.text, hit.headingPath ?? hit.path));
			const content = isImageHit
				? `Image file: ${basename(entry.sourcePath)} (direct multimodal visual embedding match)`
				: hit.text;
			results.push({
				id: chunkId,
				content,
				source: resolvedSource,
				score: hit.score,
				metadata: {
					method: this.mode === "vector" ? "minsync" : `minsync-${this.mode}`,
					virtualPath: entry.virtualPath,
					minsyncChunkId: chunkId,
					minsyncPath: hit.path,
					...(entry.mirrorKind === "visual" || entry.mirrorKind === "image"
						? { mirrorKind: entry.mirrorKind }
						: {}),
					// The window's fallback time for a local file: when the source was last modified.
					sourceMtimeMs: entry.sourceMtimeMs,
					...(location !== undefined
						? {
								lineNumber: location.lineStart,
								lineEnd: location.lineEnd,
								...(location.pageStart !== undefined && location.pageEnd !== undefined
									? { pageStart: location.pageStart, pageEnd: location.pageEnd }
									: {}),
								...(location.section !== undefined ? { section: location.section } : {}),
							}
						: {}),
				},
			});
			// With a record cut or visual mirrors the whole in-scope pool is collected, then coalesced/cut below.
			if (!hasVisualMirrors && this.recordCut === undefined && results.length >= topK) break;
		}
		if (missingSources.length > 0)
			options.onSourceMissing?.({ method: this.describe().name, missing: missingSources });
		const merged = hasVisualMirrors ? coalesceCrossMirrorPageHits(results) : results;
		if (this.recordCut === undefined) return merged.slice(0, topK);
		return this.recordCut.cut(merged, topK, (result) => result.source);
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

/**
 * Maximum characters merged into one slot when coalescing same-page hits across text and visual mirrors,
 * keeping each chunk concise while preserving both prose and chart/table extractions on the same page.
 */
const MAX_COALESCED_CONTENT_CHARS = 8_000;

/**
 * Coalesce duplicate hits that land on the exact same `(source, page)` across a document's text mirror
 * and supplementary visual mirror (or multiple visual sub-chunks of the same page). Without coalescing,
 * a visually-rich page occupies 2–3 slots in `topK` and crowds out other relevant pages from the same
 * document. Merging keeps the higher-ranked hit's score and appends the complementary mirror excerpt so
 * both text and visual evidence travel together in one slot.
 */
export function coalesceCrossMirrorPageHits(results: readonly RetrievalResult[]): RetrievalResult[] {
	const out: RetrievalResult[] = [];
	const bySinglePage = new Map<string, number>();
	const byStandaloneImage = new Map<string, number>();

	for (const result of results) {
		const mirrorKind = result.metadata?.mirrorKind;
		const pageStart = typeof result.metadata?.pageStart === "number" ? result.metadata.pageStart : undefined;
		const pageEnd = typeof result.metadata?.pageEnd === "number" ? result.metadata.pageEnd : undefined;
		const isVisual = mirrorKind === "visual";
		const isImage = mirrorKind === "image";

		// Coalesce standalone image hits across direct pixel embedding (`image`), VLM/OCR mirror (`visual`),
		// and lightweight metadata mirror (`text`) so one image file occupies a single slot with its richest description.
		const isStandaloneImage = /\.(?:png|jpe?g|webp)$/iu.test(result.source);
		if (isImage || (isStandaloneImage && (isVisual || mirrorKind === undefined))) {
			const existingImgIdx = byStandaloneImage.get(result.source);
			if (existingImgIdx !== undefined) {
				const existing = out[existingImgIdx]!;
				const existingIsRichest = existing.metadata?.mirrorKind === "visual";
				const nextContent =
					!existingIsRichest && isVisual && result.content.trim().length > 0 ? result.content : existing.content;
				out[existingImgIdx] = {
					...existing,
					content: nextContent,
					metadata: {
						...existing.metadata,
						...(isVisual || existingIsRichest ? { hasVisualComplement: true } : {}),
					},
				};
				continue;
			}
			byStandaloneImage.set(result.source, out.length);
			out.push(result);
			continue;
		}

		if (pageStart === undefined || pageEnd === undefined || pageStart !== pageEnd) {
			out.push(result);
			continue;
		}
		const key = `${result.source}\0${pageStart}`;
		const existingIndex = bySinglePage.get(key);
		if (existingIndex === undefined) {
			bySinglePage.set(key, out.length);
			out.push(result);
			continue;
		}
		const existing = out[existingIndex]!;
		const existingIsVisual = existing.metadata?.mirrorKind === "visual";
		// Two ordinary base-text chunks on the same page keep their original behavior (multiple passages per page).
		if (!isVisual && !existingIsVisual) {
			out.push(result);
			continue;
		}
		// One is visual (or both are visual sub-chunks on the exact same page): merge into the higher-ranked slot.
		const label = isVisual ? `[Visual extraction — p.${pageStart}]` : `[Base text — p.${pageStart}]`;
		const addition = result.content.trim();
		const combinedContent =
			existing.content.length < MAX_COALESCED_CONTENT_CHARS &&
			addition.length > 0 &&
			!existing.content.includes(addition)
				? `${existing.content}\n\n${label}\n${addition}`.slice(0, MAX_COALESCED_CONTENT_CHARS)
				: existing.content;
		out[existingIndex] = {
			...existing,
			content: combinedContent,
			metadata: {
				...existing.metadata,
				...(isVisual || existingIsVisual ? { hasVisualComplement: true } : {}),
			},
		};
	}

	return out;
}

/**
 * MinSync starts a chunk's `text` with its `heading_path` line (the document's workspace-relative path,
 * truncated). That line is not part of the document, so a lookup that includes it never matches. Only a first
 * line equal to the hit's own heading is dropped; the result's `content` stays exactly as MinSync returned it.
 */
function withoutHeading(text: string, heading: string): string {
	return text.startsWith(`${heading}\n`) ? text.slice(heading.length + 1) : text;
}

/** Parsed mirrors bigger than this are not searched for a chunk's line range. */
const MAX_MIRROR_BYTES = 16 * 1024 * 1024;

interface MirrorDocument {
	readonly text: string;
	/** Page index measured on exactly `text`; absent for a document without pages. */
	readonly pages: ParsedPages | undefined;
}

/**
 * Where `chunk` sits in the parsed markdown MinSync indexed for `entry`. The
 * staged copy is what MinSync chunked; the parsed mirror is the same text and is
 * the fallback before the first sync. Undefined when the document cannot be read
 * or the chunk is not found in exactly one place, so a line is never guessed.
 * The page comes from the parser's page index for the parsed mirror, kept only
 * while it was measured on this very text.
 */
function locateChunkInMirror(
	cache: Map<string, MirrorDocument | undefined>,
	entry: { readonly minSyncPath: string; readonly parsedOutputPath: string },
	chunk: string,
): ChunkLocation | undefined {
	const path = existsSync(entry.minSyncPath) ? entry.minSyncPath : entry.parsedOutputPath;
	let document: MirrorDocument | undefined;
	if (cache.has(path)) {
		document = cache.get(path);
	} else {
		try {
			if (statSync(path).size <= MAX_MIRROR_BYTES) {
				const text = readFileSync(path, "utf8");
				document = { text, pages: readPagesSidecar(entry.parsedOutputPath, text) };
			}
		} catch {
			document = undefined;
		}
		cache.set(path, document);
	}
	return document === undefined ? undefined : locateChunk(document.text, chunk, document.pages);
}
