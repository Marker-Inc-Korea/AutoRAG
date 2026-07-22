import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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

export interface BM25SyncResult {
	readonly indexPath: string;
	readonly indexedChunks: number;
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
}

export interface BM25Status {
	readonly readiness: BM25ReadinessState;
	readonly engine: BM25Engine;
	readonly message?: string;
}

type TantivyBinding = typeof import("@pngwasi/node-tantivy-binding");

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

export const BM25_SUBDIR = join(".autorag", "bm25");
const TANTIVY_SUBDIR = "tantivy";
const FALLBACK_INDEX_FILE = "fallback-index.json";
const DEFAULT_TOP_K = 20;
const MAX_CHUNK_CHARS = 2_400;

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

	async sync(): Promise<BM25SyncResult> {
		if (this.status.readiness === "disabled") return this.syncResult(0);
		this.status = { readiness: "indexing", engine: this.status.engine };
		const chunks = loadChunks(this.root);
		try {
			if (this.forceEngine === "typescript-fallback") {
				this.writeFallbackIndex(chunks);
				this.status = { readiness: "degraded_fallback", engine: "typescript-fallback" };
				return this.syncResult(chunks.length);
			}
			await this.writeTantivyIndex(chunks);
			this.status = { readiness: "ready", engine: "tantivy" };
			return this.syncResult(chunks.length);
		} catch (error) {
			if (this.fallback === "typescript") {
				this.writeFallbackIndex(chunks);
				this.status = {
					readiness: "degraded_fallback",
					engine: "typescript-fallback",
					message: error instanceof Error ? error.message : String(error),
				};
				return this.syncResult(chunks.length);
			}
			this.status = {
				readiness: error instanceof BM25UnavailableError ? error.readiness : "error",
				engine: "none",
				message: error instanceof Error ? error.message : String(error),
			};
			return this.syncResult(0);
		}
	}

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
		const index = binding.Index.open(indexDir);
		const searcher = index.searcher();
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

	private writeFallbackIndex(chunks: readonly IndexedChunk[]): void {
		const index: FallbackIndex = { version: 1, chunks };
		const path = this.fallbackIndexPath();
		mkdirSync(dirname(path), { recursive: true });
		writeFileSync(path, `${JSON.stringify(index, null, 2)}\n`);
	}

	private retrieveFallback(query: string, topK: number, scope: string | undefined): RetrievalResult[] {
		const index = this.readFallbackIndex();
		const queryTerms = tokenize(query);
		if (queryTerms.length === 0) return [];
		const scopedChunks = index.chunks.filter((chunk) => matchesVirtualPathScope(chunk.virtualPath, scope));
		const documentFrequencies = new Map<string, number>();
		const tokenized = scopedChunks.map((chunk) => ({ chunk, terms: tokenize(chunk.content) }));
		for (const term of new Set(queryTerms)) {
			documentFrequencies.set(term, tokenized.filter(({ terms }) => terms.includes(term)).length);
		}
		const avgLength = tokenized.reduce((sum, entry) => sum + entry.terms.length, 0) / Math.max(tokenized.length, 1);
		return tokenized
			.map(({ chunk, terms }) => ({
				chunk,
				score: bm25Score(queryTerms, terms, documentFrequencies, tokenized.length, avgLength),
			}))
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

	private readFallbackIndex(): FallbackIndex {
		const path = this.fallbackIndexPath();
		if (!existsSync(path)) throw new BM25UnavailableError("index_missing", "BM25 fallback index is missing");
		const parsed = JSON.parse(readFileSync(path, "utf8")) as FallbackIndex;
		return parsed.version === 1 && Array.isArray(parsed.chunks) ? parsed : { version: 1, chunks: [] };
	}

	private hasFallbackIndex(): boolean {
		return existsSync(this.fallbackIndexPath());
	}

	private tantivyIndexPath(): string {
		return join(this.indexPath, TANTIVY_SUBDIR);
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
	const k1 = 1.2;
	const b = 0.75;
	const termCounts = new Map<string, number>();
	for (const term of documentTerms) termCounts.set(term, (termCounts.get(term) ?? 0) + 1);
	let score = 0;
	for (const term of queryTerms) {
		const frequency = termCounts.get(term) ?? 0;
		if (frequency === 0) continue;
		const docsWithTerm = documentFrequencies.get(term) ?? 0;
		const idf = Math.log(1 + (documentCount - docsWithTerm + 0.5) / (docsWithTerm + 0.5));
		const denominator = frequency + k1 * (1 - b + b * (documentTerms.length / Math.max(avgDocumentLength, 1)));
		score += idf * ((frequency * (k1 + 1)) / denominator);
	}
	return score;
}

function firstString(values: unknown[] | undefined): string | undefined {
	const first = values?.[0];
	return typeof first === "string" ? first : undefined;
}

function hash(value: string): string {
	return createHash("sha256").update(value).digest("hex").slice(0, 12);
}
