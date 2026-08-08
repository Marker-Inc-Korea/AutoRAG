import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { loadMirrorIndex } from "../../mirror/index-store.ts";
import { normalizeMarkdown } from "../../parser/text.ts";
import { matchesVirtualPathScope } from "../scope.ts";
import type { RetrievalMethod, RetrievalMethodDescriptor, RetrievalOptions, RetrievalResult } from "../types.ts";

export type BM25ReadinessState =
	| "disabled"
	| "dependency_unavailable"
	| "index_missing"
	| "indexing"
	| "ready"
	| "degraded_fallback"
	| "error";

export type BM25Engine = "tantivy" | "typescript-fallback" | "none";
export type BM25FallbackMode = "typescript" | "disabled";

export interface BM25MethodOptions {
	readonly root: string;
	readonly indexPath?: string;
	readonly enabled?: boolean;
	readonly fallback?: BM25FallbackMode;
	readonly forceEngine?: Exclude<BM25Engine, "none">;
	readonly importBinding?: () => Promise<TantivyBinding>;
}

export interface BM25SyncOptions {
	/** Rebuild even when the stored fingerprint already matches the current mirror content. */
	readonly force?: boolean;
}

export interface BM25SyncResult {
	readonly indexPath: string;
	readonly indexedChunks: number;
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
	/** True when the rebuild was skipped because the stored fingerprint matched. */
	readonly skipped?: boolean;
}

export interface BM25Status {
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
	readonly message?: string;
}

type TantivyBinding = typeof import("@pngwasi/node-tantivy-binding");
// INDEX_SEMANTIC_REGION_START
// Persistence schemas: what a stored fallback-index.json and index-fingerprint.json contain and
// mean. The sidecar records the identity a stored index claims to have been built from, so its
// shape decides trust as much as the artifact itself.

type IndexedChunk = {
	readonly id: string;
	readonly virtualPath: string;
	readonly chunkIndex: number;
	readonly content: string;
};

type FallbackIndex = {
	readonly version: 1;
	readonly chunks: readonly IndexedChunk[];
};

type IndexFingerprint = {
	readonly version: 1;
	readonly fingerprint: string;
	readonly engine: Exclude<BM25Engine, "none">;
	readonly indexedChunks: number;
	/**
	 * Byte length of the fallback artifact when it was written.
	 *
	 * Skipping a rebuild trusts an artifact it never opens, so a truncated file would otherwise pass
	 * the existence check and only fail later at query time. Comparing the recorded size costs one
	 * stat. Absent for the Tantivy engine, whose artifact is a directory.
	 */
	readonly artifactBytes?: number;
};
// INDEX_SEMANTIC_REGION_END

/** Per-chunk tokenization plus term postings, derived once per built artifact. */
type PreparedFallback = {
	readonly entries: readonly { readonly chunk: IndexedChunk; readonly terms: string[] }[];
	/** term -> positions in `entries`. One entry per chunk containing the term. */
	readonly postings: ReadonlyMap<string, number[]>;
	readonly totalTermCount: number;
};

/** An opened Tantivy index and its searcher, reused while the artifact is unchanged. */
type PreparedTantivy = {
	readonly index: ReturnType<TantivyBinding["Index"]["open"]>;
	readonly searcher: ReturnType<ReturnType<TantivyBinding["Index"]["open"]>["searcher"]>;
};

export const BM25_SUBDIR = join(".autorag", "bm25");
const TANTIVY_SUBDIR = "tantivy";
const FALLBACK_INDEX_FILE = "fallback-index.json";
const FINGERPRINT_FILE = "index-fingerprint.json";
/**
 * Upper bound on the bytes retained by the query-time fallback cache.
 *
 * Token count alone does not bound retained memory: the prepared structure also holds the chunk
 * content strings (2 B/char, plus the lowercased buffer the term views share), the per-term string
 * objects, and the posting lists. Measured on Bun 1.3.14 with 450k-token corpora of different
 * shapes - short English tokens, 64-char tokens, Korean, punctuation-heavy, repetitive - a
 * fresh-process RSS differential retains 88-155 MB for the same token count, so per-token cost
 * ranges from ~200 B to ~330 B. The estimate in `preparedFallback` covers those measurements as an
 * upper bound (2 B per content char, 300 B per term occurrence, 8 B per posting, 300 B per unique
 * term for its posting array and map entry, 100 B per chunk for the entry objects), keeping the
 * cache near this 60 MB budget: corpora of roughly 150-200k typical tokens. Above it the cache is
 * not retained and each query recomputes, which is slower but returns identical results.
 */
const MAX_CACHED_BYTES = 60_000_000;
const DEFAULT_TOP_K = 20;
const MAX_CHUNK_CHARS = 2_400;
/**
 * BM25 scoring parameters, hoisted out of `bm25Score` for readability. Scoring runs at query time
 * against the raw chunks stored in the artifact (retrieveFallback), so tuning these applies to an
 * existing index immediately. They are deliberately NOT part of the index fingerprint: including
 * them would force a global rebuild for a change that needs none.
 */
const BM25_K1 = 1.2;
const BM25_B = 0.75;

/**
 * Bumped whenever the fingerprint material changes: the fingerprint hashes this version, the
 * chunking constant, and the mirror entries, so a bump invalidates every stored fingerprint and
 * forces one rebuild.
 *
 * What is covered, and how:
 * - Automatically at runtime: mirror entries (membership, content digest, parser name) and
 *   MAX_CHUNK_CHARS are hashed into the fingerprint, so they invalidate stored indexes with no
 *   manual step. INDEX_SEMANTICS_VERSION itself is part of that material.
 * - Automatically in CI: the INDEX_SEMANTIC_REGION blocks (fallback persistence schema, tantivy
 *   schema and document construction, fallback serialization/deserialization, chunk creation,
 *   fingerprint material, chunk-id hashing) in this file, plus normalizeMarkdown in
 *   src/parser/text.ts, are digested by test/bench/index-semantics-guard.test.ts, which fails
 *   until this version is bumped or the recorded digest is refreshed.
 * - Still a manual decision: parser behavior. The mirror index re-parses only when parserName,
 *   source mtime, or source size changes (src/mirror/sync.ts), so a new parser implementation
 *   that keeps its name and sees the same source identity is not re-mirrored; a bump here alone
 *   cannot produce new parser artifacts for BM25 to index.
 *
 * Query-time behavior (tokenizer regex, BM25_K1/BM25_B, the scoring formula) is deliberately NOT
 * covered: it runs against the already-stored raw chunks at query time, so changes apply to
 * existing artifacts immediately and never require a rebuild.
 */
export const INDEX_SEMANTICS_VERSION = 2;

export class BM25UnavailableError extends Error {
	readonly readiness: BM25ReadinessState;

	constructor(readiness: BM25ReadinessState, message: string) {
		super(message);
		this.name = "BM25UnavailableError";
		this.readiness = readiness;
	}
}

export class BM25Method implements RetrievalMethod {
	private readonly root: string;
	private readonly indexPath: string;
	private readonly fallback: BM25FallbackMode;
	private readonly forceEngine: BM25Engine | undefined;
	private readonly importBinding: () => Promise<TantivyBinding>;
	private status: BM25Status;
	/** Derived query-time state, keyed by the fingerprint of the artifact it was derived from. */
	private fallbackCache: { readonly stamp: string | undefined; readonly prepared: PreparedFallback } | undefined;
	private tantivyCache: { readonly stamp: string | undefined; readonly prepared: PreparedTantivy } | undefined;

	constructor(options: BM25MethodOptions) {
		this.root = options.root;
		this.indexPath = options.indexPath ?? join(options.root, BM25_SUBDIR);
		this.fallback = options.fallback ?? "typescript";
		this.forceEngine = options.forceEngine;
		this.importBinding = options.importBinding ?? (() => import("@pngwasi/node-tantivy-binding"));
		if (options.enabled === false) {
			this.status = { readiness: "disabled", engine: "none", message: "BM25 is disabled" };
		} else {
			// Reflect an on-disk index immediately so fresh processes/status after refresh
			// do not report index_missing until the next in-process sync().
			const tantivyDir = join(this.indexPath, TANTIVY_SUBDIR);
			const hasTantivy = existsSync(join(tantivyDir, "meta.json"));
			const hasFallback = existsSync(join(this.indexPath, FALLBACK_INDEX_FILE));
			if (hasTantivy) {
				this.status = { readiness: "ready", engine: "tantivy" };
			} else if (hasFallback && this.fallback !== "disabled") {
				this.status = { readiness: "degraded_fallback", engine: "typescript-fallback" };
			} else {
				this.status = { readiness: "index_missing", engine: "none", message: "BM25 index has not been built" };
			}
		}
	}

	describe(): RetrievalMethodDescriptor {
		return {
			name: "bm25",
			type: "bm25",
			description: "BM25 lexical retrieval over parsed markdown mirror chunks",
			status: this.status.readiness === "ready" || this.status.readiness === "degraded_fallback" ? "active" : "stub",
			capabilities: [
				"lexical",
				"parsed-mirrors",
				"chunked",
				"virtual-paths",
				"scoped",
				`readiness:${this.status.readiness}`,
				`engine:${this.status.engine}`,
			],
		};
	}

	getStatus(): BM25Status {
		return this.status;
	}

	/**
	 * Build the lexical index from the parsed mirrors.
	 *
	 * The rebuild is skipped when the stored fingerprint already matches the current mirror content
	 * and the built artifact is still present. Computing the fingerprint reads only the mirror index
	 * file, so an unchanged refresh never opens a single mirror document.
	 *
	 * In auto engine mode a failed Tantivy build commits the fallback artifact (when the fallback
	 * engine is enabled) and records the fallback engine in the fingerprint, so later non-force
	 * refreshes skip with `skipped: true` and keep serving the fallback even after Tantivy recovers.
	 * That is deliberate: the sidecar records the engine the artifact was actually built with, and
	 * retrying Tantivy on every refresh would rebuild an unchanged workspace. Recovery is explicit:
	 * `autorag index rebuild` (force), or a config change to `forceEngine: "tantivy"`, which
	 * changes the fingerprint material and invalidates the stored artifact.
	 */
	async sync(options: BM25SyncOptions = {}): Promise<BM25SyncResult> {
		if (this.status.readiness === "disabled") return this.syncResult(0);

		const fingerprint = this.computeFingerprint();
		if (!options.force) {
			const reused = this.reuseExistingIndex(fingerprint);
			if (reused) return reused;
		}

		// About to replace the artifact: drop derived state before the old files disappear, so a
		// cached handle can never outlive the index it was opened against.
		this.fallbackCache = undefined;
		this.tantivyCache = undefined;
		this.status = { readiness: "indexing", engine: this.status.engine };
		const chunks = loadChunks(this.root);

		// The catch below covers engine selection/build failure only. Committing the chosen artifact
		// and recording its fingerprint happen after it, exactly once, so a failure to record the
		// fingerprint can never be mistaken for an engine failure - which would rewrite the artifact
		// and write the fingerprint twice.
		let engine: Exclude<BM25Engine, "none">;
		try {
			if (this.forceEngine === "typescript-fallback") {
				this.writeFallbackIndex(chunks);
				engine = "typescript-fallback";
			} else {
				await this.writeTantivyIndex(chunks);
				engine = "tantivy";
			}
		} catch (error) {
			if (this.fallback === "typescript") {
				this.writeFallbackIndex(chunks);
				this.status = {
					readiness: "degraded_fallback",
					engine: "typescript-fallback",
					message: error instanceof Error ? error.message : String(error),
				};
				this.writeFingerprint(fingerprint, "typescript-fallback", chunks.length);
				return this.syncResult(chunks.length);
			}
			this.status = {
				readiness: error instanceof BM25UnavailableError ? error.readiness : "error",
				engine: "none",
				message: error instanceof Error ? error.message : String(error),
			};
			// No fingerprint is written on hard failure, so the next sync retries the build.
			return this.syncResult(0);
		}

		if (engine === "tantivy") {
			this.status = { readiness: "ready", engine: "tantivy" };
		} else {
			this.status = { readiness: "degraded_fallback", engine: "typescript-fallback" };
		}
		this.writeFingerprint(fingerprint, engine, chunks.length);
		return this.syncResult(chunks.length);
	}

	// INDEX_SEMANTIC_REGION_START
	// Fingerprint material and sidecar trust path: the exact identity a stored index claims to
	// have been built from, plus how that identity is read, compared, and recorded. Weakening any
	// of it - the material, the equality check, the engine check, or the artifact-size check -
	// lets a stale artifact pass the skip and be trusted against new input, so this block is
	// guarded exactly like the artifact formats above.
	/**
	 * Identity of the inputs the index is built from.
	 *
	 * Covers mirror membership, per-entry parsed-content digest, the parser that produced it, the
	 * chunking constant, and this version. Query-time semantics (tokenizer, scoring parameters) are
	 * deliberately absent: they run against the stored raw chunks at query time, so tuning them
	 * applies to an existing artifact immediately and never needs a rebuild. Entries whose digest
	 * is still absent contribute a sentinel so a legacy index can never be mistaken for a
	 * fingerprinted one.
	 */
	private computeFingerprint(): string {
		const index = loadMirrorIndex(this.root);
		const entries = Object.values(index.entries)
			.map((entry) => `${entry.virtualPath}\u0000${entry.contentSha256 ?? "-"}\u0000${entry.parserName}`)
			.sort();
		const material = [
			`semantics:${INDEX_SEMANTICS_VERSION}`,
			`chunk:${MAX_CHUNK_CHARS}`,
			`engine:${this.forceEngine ?? "auto"}`,
			`fallback:${this.fallback}`,
			`entries:${entries.length}`,
			...entries,
		].join("\n");
		return createHash("sha256").update(material, "utf8").digest("hex");
	}

	/** Completed result when the on-disk artifact already matches `fingerprint`, else undefined. */
	private reuseExistingIndex(fingerprint: string): BM25SyncResult | undefined {
		const stored = this.readFingerprint();
		if (!stored || stored.fingerprint !== fingerprint) return undefined;
		if (stored.engine === "tantivy" && existsSync(join(this.tantivyIndexPath(), "meta.json"))) {
			this.status = { readiness: "ready", engine: "tantivy" };
			return { ...this.syncResult(stored.indexedChunks), skipped: true };
		}
		if (
			stored.engine === "typescript-fallback" &&
			this.fallback !== "disabled" &&
			this.hasFallbackIndex() &&
			this.fallbackArtifactBytes() === stored.artifactBytes
		) {
			this.status = { readiness: "degraded_fallback", engine: "typescript-fallback" };
			return { ...this.syncResult(stored.indexedChunks), skipped: true };
		}
		return undefined;
	}

	private readFingerprint(): IndexFingerprint | undefined {
		const path = this.fingerprintPath();
		if (!existsSync(path)) return undefined;
		try {
			const parsed = JSON.parse(readFileSync(path, "utf8")) as Partial<IndexFingerprint>;
			if (parsed.version !== 1) return undefined;
			if (typeof parsed.fingerprint !== "string" || typeof parsed.indexedChunks !== "number") return undefined;
			if (parsed.engine !== "tantivy" && parsed.engine !== "typescript-fallback") return undefined;
			return {
				version: 1,
				fingerprint: parsed.fingerprint,
				engine: parsed.engine,
				indexedChunks: parsed.indexedChunks,
				...(typeof parsed.artifactBytes === "number" ? { artifactBytes: parsed.artifactBytes } : {}),
			};
		} catch {
			return undefined;
		}
	}

	/**
	 * Record the fingerprint after the artifact is on disk.
	 *
	 * Written tmp-then-rename so a crash mid-write leaves either the old fingerprint or none, never
	 * a torn one. Ordering matters: the artifact is committed first, so a fingerprint can never
	 * claim an index that was not built. When the write fails, the stale fingerprint is removed and
	 * the error propagates: a fresh artifact must never sit under an old fingerprint, because the
	 * next sync would trust it and skip the rebuild the new artifact needs. A missing fingerprint
	 * only costs one rebuild next time; a stale one silently serves the wrong index.
	 */
	private writeFingerprint(fingerprint: string, engine: Exclude<BM25Engine, "none">, indexedChunks: number): void {
		const artifactBytes = engine === "typescript-fallback" ? this.fallbackArtifactBytes() : undefined;
		const payload: IndexFingerprint = {
			version: 1,
			fingerprint,
			engine,
			indexedChunks,
			...(artifactBytes === undefined ? {} : { artifactBytes }),
		};
		const tmp = join(this.indexPath, `${FINGERPRINT_FILE}.${randomUUID()}.tmp`);
		try {
			mkdirSync(this.indexPath, { recursive: true });
			writeFileSync(tmp, `${JSON.stringify(payload)}\n`);
			renameSync(tmp, this.fingerprintPath());
		} catch (error) {
			// The new fingerprint could not be recorded. Remove whatever occupies the fingerprint
			// path so no stale identity survives next to the freshly built artifact, then surface
			// the original error instead of hiding it. Removal is best-effort: if it fails too the
			// rethrow below still makes the failure loud.
			try {
				rmSync(this.fingerprintPath(), { recursive: true, force: true });
			} catch {
				// Fall through to the rethrow; the write failure is the primary signal.
			}
			throw error;
		} finally {
			// A failed write or rename must not leave the tmp behind occupying space (e.g. under
			// ENOSPC); on success the rename already consumed it. Removal is best-effort so it can
			// never mask the original error.
			try {
				rmSync(tmp, { force: true });
			} catch {
				// Fall through; the write failure is the primary signal.
			}
		}
	}

	/** Size of the fallback artifact, or undefined when it cannot be stat'ed. */
	private fallbackArtifactBytes(): number | undefined {
		try {
			return statSync(this.fallbackIndexPath()).size;
		} catch {
			return undefined;
		}
	}
	// INDEX_SEMANTIC_REGION_END
	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmedQuery = query.trim();
		if (trimmedQuery.length === 0) return [];
		const topK = options.topK ?? DEFAULT_TOP_K;
		if (this.status.readiness === "disabled" || this.status.readiness === "dependency_unavailable") {
			throw new BM25UnavailableError(this.status.readiness, this.status.message ?? "BM25 is unavailable");
		}
		if (this.status.readiness === "ready") {
			return this.retrieveTantivy(trimmedQuery, topK, options.scope);
		}
		if (this.status.readiness === "degraded_fallback") {
			return this.retrieveFallback(trimmedQuery, topK, options.scope);
		}
		if (this.hasFallbackIndex()) {
			this.status = { readiness: "degraded_fallback", engine: "typescript-fallback" };
			return this.retrieveFallback(trimmedQuery, topK, options.scope);
		}
		throw new BM25UnavailableError("index_missing", "BM25 index has not been built; call refresh() first");
	}

	// INDEX_SEMANTIC_REGION_START
	// Tantivy schema, analyzer, and document construction: the stored tantivy index means exactly
	// what is written here. Adding a field or changing a tokenizerName/indexOption invalidates
	// every stored index.
	private async writeTantivyIndex(chunks: readonly IndexedChunk[]): Promise<void> {
		if (this.forceEngine === "typescript-fallback") throw new Error("Tantivy engine disabled by configuration");
		let binding: TantivyBinding;
		try {
			binding = await this.importBinding();
		} catch (cause) {
			throw new BM25UnavailableError(
				"dependency_unavailable",
				cause instanceof Error ? cause.message : "Tantivy binding is unavailable",
			);
		}
		const indexDir = this.tantivyIndexPath();
		rmSync(indexDir, { recursive: true, force: true });
		mkdirSync(indexDir, { recursive: true });
		const schema = new binding.SchemaBuilder()
			.addTextField("virtualPath", { stored: true, indexOption: "basic", tokenizerName: "raw" })
			.addTextField("chunkId", { stored: true, indexOption: "basic", tokenizerName: "raw" })
			.addTextField("content", { stored: true, indexOption: "position" })
			.build();
		const index = new binding.Index(schema, indexDir, false);
		const writer = index.writer(30_000_000, 1);
		for (const chunk of chunks) {
			const doc = new binding.Document();
			doc.addText("virtualPath", chunk.virtualPath);
			doc.addText("chunkId", String(chunk.chunkIndex));
			doc.addText("content", chunk.content);
			writer.addDocument(doc);
		}
		writer.commit();
	}
	// INDEX_SEMANTIC_REGION_END

	private async retrieveTantivy(
		queryText: string,
		topK: number,
		scope: string | undefined,
	): Promise<RetrievalResult[]> {
		const binding = await this.importBinding();
		const indexDir = this.tantivyIndexPath();
		if (!existsSync(indexDir) || !binding.Index.exists(indexDir)) {
			this.status = { readiness: "index_missing", engine: "none", message: "Tantivy BM25 index is missing" };
			throw new BM25UnavailableError("index_missing", "Tantivy BM25 index is missing; call refresh() first");
		}
		// Opening the index and creating a searcher is per-artifact work, not per-query work. Both are
		// reused until the fingerprint changes; a searcher observes a fixed commit, so reusing one
		// returns exactly what a freshly opened one would for the same artifact.
		const stamp = this.artifactStamp();
		let prepared = this.tantivyCache?.stamp === stamp ? this.tantivyCache?.prepared : undefined;
		if (!prepared) {
			const opened = binding.Index.open(indexDir);
			prepared = { index: opened, searcher: opened.searcher() };
			// Without a fingerprint the stamp is undefined and could never change, so a retained
			// handle would outlive a replacement of the index directory; reopen per query instead.
			this.tantivyCache = stamp === undefined ? undefined : { stamp, prepared };
		}
		const { index, searcher } = prepared;
		const query = index.parseQueryLenient(queryText, ["content"])[0];
		const pageSize = scope ? 100 : topK;
		let offset = 0;
		let totalCount: number | undefined;
		const results: RetrievalResult[] = [];
		while (results.length < topK && (totalCount === undefined || offset < totalCount)) {
			const page = searcher.search(query, pageSize, true, undefined, offset);
			totalCount = page.count ?? page.hits.length;
			if (page.hits.length === 0) break;
			for (const hit of page.hits) {
				const doc = searcher.doc(hit.docAddress).toDict() as Record<string, unknown[]>;
				const virtualPath = firstString(doc.virtualPath);
				const content = firstString(doc.content);
				const chunkId = firstString(doc.chunkId) ?? "0";
				if (!virtualPath || !content || !matchesVirtualPathScope(virtualPath, scope)) continue;
				results.push({
					id: `bm25:${virtualPath}:${chunkId}`,
					content,
					source: virtualPath,
					score: hit.score ?? 0,
					metadata: {
						method: "bm25",
						chunkIndex: Number(chunkId),
						readiness: this.status.readiness,
						engine: "tantivy",
					},
				});
				if (results.length >= topK) break;
			}
			offset += page.hits.length;
		}
		return results;
	}

	// INDEX_SEMANTIC_REGION_START
	// Fallback serialization: the exact bytes of fallback-index.json are decided here.
	private writeFallbackIndex(chunks: readonly IndexedChunk[]): void {
		const index: FallbackIndex = { version: 1, chunks };
		const path = this.fallbackIndexPath();
		mkdirSync(dirname(path), { recursive: true });
		// Written tmp-then-rename, matching src/mirror/sync.ts writeAtomic: a crash mid-write can
		// never leave a torn artifact, and a read-only directory cannot silently accept an in-place
		// overwrite of the existing file. The tmp is removed in `finally` so a failed write or
		// rename (e.g. ENOSPC) cannot leave it behind occupying space; removal failure must not
		// mask the original error.
		const tmp = `${path}.${randomUUID()}.tmp`;
		try {
			writeFileSync(tmp, `${JSON.stringify(index, null, 2)}\n`);
			renameSync(tmp, path);
		} finally {
			try {
				rmSync(tmp, { force: true });
			} catch {
				// Fall through; the write failure is the primary signal.
			}
		}
	}
	// INDEX_SEMANTIC_REGION_END

	private retrieveFallback(query: string, topK: number, scope: string | undefined): RetrievalResult[] {
		const queryTerms = tokenize(query);
		if (queryTerms.length === 0) return [];
		const prepared = this.preparedFallback();

		// Scope selects the subset that defines the corpus statistics. Document frequency and average
		// length are therefore computed over the scoped subset, exactly as before; only the
		// per-chunk tokenization is reused instead of being redone on every query.
		const tokenized =
			scope === undefined
				? prepared.entries
				: prepared.entries.filter(({ chunk }) => matchesVirtualPathScope(chunk.virtualPath, scope));

		const documentFrequencies = new Map<string, number>();
		for (const term of new Set(queryTerms)) {
			if (scope === undefined) {
				// Unscoped: the posting list length is the document frequency.
				documentFrequencies.set(term, prepared.postings.get(term)?.length ?? 0);
				continue;
			}
			// Scoped: count only postings inside the subset. This visits the term's postings rather
			// than every chunk, but yields the identical count.
			let count = 0;
			for (const position of prepared.postings.get(term) ?? []) {
				const entry = prepared.entries[position];
				if (entry && matchesVirtualPathScope(entry.chunk.virtualPath, scope)) count += 1;
			}
			documentFrequencies.set(term, count);
		}

		const totalTermCount =
			scope === undefined ? prepared.totalTermCount : tokenized.reduce((sum, entry) => sum + entry.terms.length, 0);
		const avgLength = totalTermCount / Math.max(tokenized.length, 1);
		// Only chunks that contain at least one query term can score above zero: bm25Score skips terms
		// with zero frequency, and its idf term is positive for every df <= N. Chunks outside the
		// posting union would therefore score exactly 0 and be dropped by the filter below, so
		// visiting them changes nothing except cost.
		const candidates: number[] = [];
		const seen = new Set<number>();
		for (const term of new Set(queryTerms)) {
			for (const position of prepared.postings.get(term) ?? []) {
				if (seen.has(position)) continue;
				seen.add(position);
				const entry = prepared.entries[position];
				if (!entry) continue;
				if (scope !== undefined && !matchesVirtualPathScope(entry.chunk.virtualPath, scope)) continue;
				candidates.push(position);
			}
		}

		return candidates
			.map((position) => {
				const entry = prepared.entries[position] as { chunk: IndexedChunk; terms: string[] };
				return {
					chunk: entry.chunk,
					score: bm25Score(queryTerms, entry.terms, documentFrequencies, tokenized.length, avgLength),
				};
			})
			.filter((entry) => entry.score > 0)
			.sort((a, b) => b.score - a.score || a.chunk.virtualPath.localeCompare(b.chunk.virtualPath))
			.slice(0, topK)
			.map(({ chunk, score }) => ({
				id: chunk.id,
				content: chunk.content,
				source: chunk.virtualPath,
				score,
				metadata: {
					method: "bm25",
					chunkIndex: chunk.chunkIndex,
					readiness: this.status.readiness,
					engine: "typescript-fallback",
				},
			}));
	}

	/**
	 * Tokenized view of the fallback index, reused across queries.
	 *
	 * Rebuilt whenever the stored fingerprint changes, so a cache entry can never outlive the index
	 * it describes. Tokenization is deterministic, so reusing it cannot change a score; it only
	 * stops the same work from being repeated on every query.
	 */
	private preparedFallback(): PreparedFallback {
		const stamp = this.artifactStamp();
		const cached = this.fallbackCache;
		if (cached && cached.stamp === stamp) return cached.prepared;

		const index = this.readFallbackIndex();
		const entries = index.chunks.map((chunk) => ({ chunk, terms: tokenize(chunk.content) }));
		const postings = new Map<string, number[]>();
		let totalTermCount = 0;
		let contentChars = 0;
		let postingsCount = 0;
		for (const [position, entry] of entries.entries()) {
			totalTermCount += entry.terms.length;
			contentChars += entry.chunk.content.length;
			// One posting per distinct term per chunk: the count matches "chunks containing term".
			for (const term of new Set(entry.terms)) {
				const list = postings.get(term);
				if (list) list.push(position);
				else postings.set(term, [position]);
				postingsCount += 1;
			}
		}
		const prepared: PreparedFallback = { entries, postings, totalTermCount };
		// The retention bound is an estimate of the bytes the structure actually pins, not a token
		// count: content strings and term objects dominate and vary with corpus shape (see
		// MAX_CACHED_BYTES). Above the bound the structure is used for this query but not retained,
		// so a large corpus cannot pin tens of megabytes for the process lifetime. Results are
		// unaffected: the next query simply rebuilds the same structure. Without a fingerprint the
		// stamp can never change, so a retained entry would outlive the artifact it was derived
		// from; the cache is not retained either.
		const estimatedBytes =
			contentChars * 2 + totalTermCount * 300 + postingsCount * 8 + postings.size * 300 + entries.length * 100;
		this.fallbackCache = stamp === undefined || estimatedBytes > MAX_CACHED_BYTES ? undefined : { stamp, prepared };
		return prepared;
	}

	/**
	 * Identity of the currently built artifact.
	 *
	 * The fingerprint sidecar is authoritative and is rewritten on every successful build, including
	 * builds performed by another process. Reading it costs one small file read per query, which is
	 * orders of magnitude cheaper than the work it guards. When no fingerprint exists the stamp is
	 * `undefined`, and the caches never retain an entry with an undefined stamp: such an entry
	 * could never be invalidated by an artifact replacement, so a sidecar-less artifact must be
	 * re-derived on every query.
	 */
	private artifactStamp(): string | undefined {
		return this.readFingerprint()?.fingerprint;
	}

	// INDEX_SEMANTIC_REGION_START
	// Fallback deserialization: how a stored fallback-index.json is read back.
	private readFallbackIndex(): FallbackIndex {
		const path = this.fallbackIndexPath();
		if (!existsSync(path)) throw new BM25UnavailableError("index_missing", "BM25 fallback index is missing");
		const parsed = JSON.parse(readFileSync(path, "utf8")) as FallbackIndex;
		return parsed.version === 1 && Array.isArray(parsed.chunks) ? parsed : { version: 1, chunks: [] };
	}
	// INDEX_SEMANTIC_REGION_END

	private hasFallbackIndex(): boolean {
		return existsSync(this.fallbackIndexPath());
	}

	private tantivyIndexPath(): string {
		return join(this.indexPath, TANTIVY_SUBDIR);
	}

	private fingerprintPath(): string {
		return join(this.indexPath, FINGERPRINT_FILE);
	}

	private fallbackIndexPath(): string {
		return join(this.indexPath, FALLBACK_INDEX_FILE);
	}

	private syncResult(indexedChunks: number): BM25SyncResult {
		return {
			indexPath: this.indexPath,
			indexedChunks,
			readiness: this.status.readiness,
			engine: this.status.engine,
		};
	}
}

// INDEX_SEMANTIC_REGION_START
// Chunk creation: loadChunks reads each mirror document, normalizes it, and chunkMarkdown splits
// it into the exact text that is stored in the artifact. `hash` derives the chunk id stored with
// each chunk, so its truncation length is part of the artifact's meaning.
function loadChunks(root: string): IndexedChunk[] {
	const index = loadMirrorIndex(root);
	const chunks: IndexedChunk[] = [];
	for (const entry of Object.values(index.entries).sort((a, b) => a.virtualPath.localeCompare(b.virtualPath))) {
		if (!existsSync(entry.outputPath)) continue;
		const content = normalizeMarkdown(readFileSync(entry.outputPath, "utf8"));
		for (const [chunkIndex, chunkContent] of chunkMarkdown(content).entries()) {
			chunks.push({
				id: `bm25:${entry.virtualPath}:${chunkIndex}:${hash(chunkContent)}`,
				virtualPath: entry.virtualPath,
				chunkIndex,
				content: chunkContent,
			});
		}
	}
	return chunks;
}

function chunkMarkdown(markdown: string): string[] {
	const paragraphs = markdown
		.split(/\n{2,}/u)
		.map((part) => part.trim())
		.filter(Boolean);
	const chunks: string[] = [];
	let current = "";
	for (const paragraph of paragraphs.length > 0 ? paragraphs : [markdown.trim()].filter(Boolean)) {
		if (current.length > 0 && current.length + paragraph.length + 2 > MAX_CHUNK_CHARS) {
			chunks.push(current);
			current = paragraph;
		} else {
			current = current.length === 0 ? paragraph : `${current}\n\n${paragraph}`;
		}
	}
	if (current.length > 0) chunks.push(current);
	return chunks;
}
function hash(value: string): string {
	return createHash("sha256").update(value).digest("hex").slice(0, 12);
}
// INDEX_SEMANTIC_REGION_END

// Query-time scoring semantics: `tokenize` and `bm25Score` run only at query time against the raw
// chunks already stored in the artifact (retrieveFallback / preparedFallback). A change here
// applies to existing artifacts immediately and never invalidates them, so these functions are
// deliberately kept OUTSIDE the INDEX_SEMANTIC_REGION: guarding them would force a global rebuild
// for a change that needs none.

function tokenize(value: string): string[] {
	return value.toLowerCase().match(/[\p{Letter}\p{Number}_]+/gu) ?? [];
}

function bm25Score(
	queryTerms: readonly string[],
	documentTerms: readonly string[],
	documentFrequencies: ReadonlyMap<string, number>,
	documentCount: number,
	avgDocumentLength: number,
): number {
	const termCounts = new Map<string, number>();
	for (const term of documentTerms) termCounts.set(term, (termCounts.get(term) ?? 0) + 1);
	let score = 0;
	for (const term of queryTerms) {
		const frequency = termCounts.get(term) ?? 0;
		if (frequency === 0) continue;
		const docsWithTerm = documentFrequencies.get(term) ?? 0;
		const idf = Math.log(1 + (documentCount - docsWithTerm + 0.5) / (docsWithTerm + 0.5));
		const lengthNorm = documentTerms.length / Math.max(avgDocumentLength, 1);
		const denominator = frequency + BM25_K1 * (1 - BM25_B + BM25_B * lengthNorm);
		score += idf * ((frequency * (BM25_K1 + 1)) / denominator);
	}
	return score;
}

function firstString(values: unknown[] | undefined): string | undefined {
	const first = values?.[0];
	return typeof first === "string" ? first : undefined;
}
