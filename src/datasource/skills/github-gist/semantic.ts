/**
 * Semantic retrieval for the github-gist datasource (issue #1588).
 *
 * A small persisted vector sidecar (`vectors.json`) mirrors the chunk store:
 * entries are keyed by chunk id with a content hash so unchanged chunks are
 * never re-embedded, removed chunks are pruned, and an embedding-identity
 * change (provider/model/dimension) forces a full re-embed. The default
 * embedder talks to the loopback autorag-gateway (`ensureRuntime` with
 * `cachedOnly`, then `POST /v1/embeddings`); tests inject a stub. Every
 * failure degrades — indexing stays lexical-only and retrieval returns [].
 */

import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { Embedder, EmbeddingIdentity } from "../../../embedding-runtime/gateway-embedder.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import type { DatasourceChunkStore, StoredChunk } from "../../chunk-store.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";

export interface GistSemanticSyncOk {
	readonly ok: true;
	/** Number of chunks (re-)embedded in this sync. */
	readonly embedded: number;
}

export interface GistSemanticSyncFail {
	readonly ok: false;
	readonly message: string;
}

interface VectorEntry {
	readonly hash: string;
	readonly vector: readonly number[];
}

interface PersistedVectors {
	readonly version: number;
	readonly identity: EmbeddingIdentity;
	readonly entries: Record<string, VectorEntry>;
}

const VECTORS_VERSION = 1;

function contentHash(chunk: StoredChunk): string {
	return createHash("sha256")
		.update(`${chunk.title ?? ""}\n${chunk.content}`)
		.digest("hex");
}

function sameIdentity(a: EmbeddingIdentity, b: EmbeddingIdentity): boolean {
	return a.provider === b.provider && a.model === b.model && a.dimension === b.dimension;
}

function cosine(a: readonly number[], b: readonly number[]): number {
	let dot = 0;
	let normA = 0;
	let normB = 0;
	const length = Math.min(a.length, b.length);
	for (let i = 0; i < length; i += 1) {
		const x = a[i] ?? 0;
		const y = b[i] ?? 0;
		dot += x * y;
		normA += x * x;
		normB += y * y;
	}
	if (normA === 0 || normB === 0) return 0;
	return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/** Persisted per-instance vector sidecar mirroring the chunk store. */
export class GistSemanticIndex {
	private readonly statePath: string | undefined;
	private identity: EmbeddingIdentity | undefined;
	private entries: Record<string, VectorEntry> = {};
	private loaded = false;

	constructor(options: { readonly statePath?: string } = {}) {
		this.statePath = options.statePath;
	}

	/**
	 * Embed new/changed chunks, prune removed ones, and persist. Never throws;
	 * an embedder failure leaves prior entries intact and reports ok:false.
	 */
	async sync(chunks: readonly StoredChunk[], embedder: Embedder): Promise<GistSemanticSyncOk | GistSemanticSyncFail> {
		this.ensureLoaded();
		let identity: EmbeddingIdentity;
		try {
			identity = await embedder.identity();
		} catch (error) {
			return { ok: false, message: error instanceof Error ? error.message : String(error) };
		}
		if (this.identity === undefined || !sameIdentity(this.identity, identity)) {
			this.entries = {};
			this.identity = identity;
		}
		const currentIds = new Set(chunks.map((chunk) => chunk.chunkId));
		for (const chunkId of Object.keys(this.entries)) {
			if (!currentIds.has(chunkId)) delete this.entries[chunkId];
		}
		const pending = chunks.filter((chunk) => {
			const entry = this.entries[chunk.chunkId];
			return entry === undefined || entry.hash !== contentHash(chunk);
		});
		if (pending.length === 0) {
			this.persist();
			return { ok: true, embedded: 0 };
		}
		let vectors: readonly (readonly number[])[];
		try {
			vectors = await embedder.embed(pending.map((chunk) => `${chunk.title ?? ""}\n${chunk.content}`));
		} catch (error) {
			return { ok: false, message: error instanceof Error ? error.message : String(error) };
		}
		if (vectors.length !== pending.length) {
			return { ok: false, message: `embedder returned ${vectors.length} vectors for ${pending.length} chunks` };
		}
		for (const [index, chunk] of pending.entries()) {
			const vector = vectors[index] ?? [];
			this.entries[chunk.chunkId] = { hash: contentHash(chunk), vector };
		}
		this.persist();
		return { ok: true, embedded: pending.length };
	}

	/** Cosine-ranked chunk ids for a query vector. Never throws. */
	searchByVector(queryVector: readonly number[], topK: number): readonly { chunkId: string; score: number }[] {
		this.ensureLoaded();
		const scored: { chunkId: string; score: number }[] = [];
		for (const [chunkId, entry] of Object.entries(this.entries)) {
			const score = cosine(queryVector, entry.vector);
			if (score > 0) scored.push({ chunkId, score });
		}
		scored.sort((a, b) => b.score - a.score || a.chunkId.localeCompare(b.chunkId));
		return scored.slice(0, topK);
	}

	private ensureLoaded(): void {
		if (this.loaded) return;
		this.loaded = true;
		if (this.statePath === undefined || !existsSync(this.statePath)) return;
		try {
			const parsed = JSON.parse(readFileSync(this.statePath, "utf8")) as PersistedVectors;
			if (parsed.version !== VECTORS_VERSION || typeof parsed.entries !== "object" || parsed.entries === null)
				return;
			if (typeof parsed.identity?.dimension !== "number") return;
			this.identity = parsed.identity;
			this.entries = { ...parsed.entries };
		} catch {
			// Corrupt sidecar: start empty; the next sync re-embeds.
		}
	}

	private persist(): void {
		if (this.statePath === undefined) return;
		if (this.identity === undefined) return;
		try {
			mkdirSync(dirname(this.statePath), { recursive: true });
			const payload: PersistedVectors = {
				version: VECTORS_VERSION,
				identity: this.identity,
				entries: this.entries,
			};
			writeFileSync(this.statePath, `${JSON.stringify(payload)}\n`, "utf8");
		} catch {
			// Persistence is best-effort; in-memory entries stay authoritative.
		}
	}
}

export interface GitHubGistSemanticMethodOptions {
	readonly skillName: string;
	readonly skillType: string;
	readonly instanceId: string;
	readonly tags: readonly string[];
	readonly store: DatasourceChunkStore;
	readonly index: GistSemanticIndex;
	readonly embedder: Embedder;
}

const DEFAULT_TOP_K = 20;

/** Cosine similarity retrieval over the gist vector sidecar. Never throws. */
export class GitHubGistSemanticMethod implements RetrievalMethod {
	private readonly options: GitHubGistSemanticMethodOptions;

	constructor(options: GitHubGistSemanticMethodOptions) {
		this.options = options;
	}

	describe(): RetrievalMethodDescriptor {
		const { skillName, skillType, tags } = this.options;
		return {
			name: `${skillName}-semantic`,
			type: "vector",
			description: `Semantic retrieval over embedded ${skillType} chunks`,
			status: "active",
			capabilities: ["semantic", "scoped", "path-opaque-sources"],
			datasourceId: skillName,
			tags: [...tags],
		};
	}

	async retrieve(query: string, options: RetrievalOptions): Promise<RetrievalResult[]> {
		const trimmed = query.trim();
		if (trimmed.length === 0) return [];
		const topK = options.topK ?? DEFAULT_TOP_K;
		let queryVector: readonly number[] | undefined;
		try {
			queryVector = (await this.options.embedder.embed([trimmed]))[0];
		} catch {
			return [];
		}
		if (queryVector === undefined) return [];
		const hits = this.options.index.searchByVector(queryVector, Math.max(topK * 3, topK));
		if (hits.length === 0) return [];
		const byChunkId = new Map(this.options.store.chunks().map((chunk) => [chunk.chunkId, chunk]));
		const { skillName, instanceId } = this.options;
		const mapped: RetrievalResult[] = [];
		for (const { chunkId, score } of hits) {
			const chunk = byChunkId.get(chunkId);
			if (chunk === undefined) continue;
			const source = datasourceSourcePath(skillName, instanceId, chunk.chunkId);
			if (!matchesDatasourceScope(source, options.scope)) continue;
			mapped.push({
				id: `${skillName}:${instanceId}:${chunk.chunkId}`,
				content: chunk.content,
				source,
				score,
				metadata: {
					...chunk.metadata,
					method: `${skillName}-semantic`,
					datasourceId: skillName,
					instanceId,
					chunkId: chunk.chunkId,
					...(chunk.title !== undefined ? { title: chunk.title } : {}),
					...(chunk.hierarchy.length > 0 ? { hierarchy: chunk.hierarchy.join("/") } : {}),
					...(chunk.publishedAt !== undefined ? { publishedAt: chunk.publishedAt } : {}),
				},
			});
			if (mapped.length >= topK) break;
		}
		return mapped;
	}
}
