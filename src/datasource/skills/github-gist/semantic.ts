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
import type { ProfileId } from "../../../embedding-runtime/types.ts";
import type {
	RetrievalMethod,
	RetrievalMethodDescriptor,
	RetrievalOptions,
	RetrievalResult,
} from "../../../retrieval/types.ts";
import type { DatasourceChunkStore, StoredChunk } from "../../chunk-store.ts";
import { datasourceSourcePath, matchesDatasourceScope } from "../../scope.ts";

export interface GistEmbeddingIdentity {
	readonly provider: string;
	readonly model: string;
	readonly dimension: number;
}

/** Embedder contract: identity is async because the gateway resolves it lazily. */
export interface GistEmbedder {
	identity(): Promise<GistEmbeddingIdentity>;
	embed(texts: readonly string[]): Promise<readonly (readonly number[])[]>;
}

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
	readonly identity: GistEmbeddingIdentity;
	readonly entries: Record<string, VectorEntry>;
}

const VECTORS_VERSION = 1;

function contentHash(chunk: StoredChunk): string {
	return createHash("sha256").update(`${chunk.title ?? ""}\n${chunk.content}`).digest("hex");
}

function sameIdentity(a: GistEmbeddingIdentity, b: GistEmbeddingIdentity): boolean {
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
	private identity: GistEmbeddingIdentity | undefined;
	private entries: Record<string, VectorEntry> = {};
	private loaded = false;

	constructor(options: { readonly statePath?: string } = {}) {
		this.statePath = options.statePath;
	}

	/**
	 * Embed new/changed chunks, prune removed ones, and persist. Never throws;
	 * an embedder failure leaves prior entries intact and reports ok:false.
	 */
	async sync(chunks: readonly StoredChunk[], embedder: GistEmbedder): Promise<GistSemanticSyncOk | GistSemanticSyncFail> {
		this.ensureLoaded();
		let identity: GistEmbeddingIdentity;
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
			if (parsed.version !== VECTORS_VERSION || typeof parsed.entries !== "object" || parsed.entries === null) return;
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
	readonly embedder: GistEmbedder;
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
			if (!this.matchesScope(source, options.scope, options.allowedScopes)) continue;
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

	private matchesScope(
		source: string,
		scope: string | undefined,
		allowedScopes: readonly string[] | undefined,
	): boolean {
		if (!matchesDatasourceScope(source, scope)) return false;
		if (allowedScopes === undefined || allowedScopes.length === 0) return true;
		return allowedScopes.some((entry) => matchesDatasourceScope(source, entry));
	}
}

export interface GatewayGistEmbedderOptions {
	/** Runtime resolver; defaults to the shared embedding-runtime ensureRuntime. */
	readonly runtime?: {
		ensureRuntime(input?: { profileId?: ProfileId; cachedOnly?: boolean }): Promise<{
			baseUrl: string;
			identity: { provider: string; model: string; dimension: number };
		}>;
	};
	readonly profileId?: ProfileId;
	readonly fetchImpl?: typeof fetch;
}

/**
 * Production embedder backed by the loopback autorag-gateway. The runtime is
 * ensured lazily with `cachedOnly` so semantic retrieval never triggers a
 * model download; an unprimed cache surfaces as a sync/retrieve degrade.
 */
export function createGatewayGistEmbedder(options: GatewayGistEmbedderOptions = {}): GistEmbedder {
	let ensured: { baseUrl: string; identity: GistEmbeddingIdentity } | undefined;
	async function ensure(): Promise<{ baseUrl: string; identity: GistEmbeddingIdentity }> {
		if (ensured !== undefined) return ensured;
		let runtime = options.runtime;
		if (runtime === undefined) {
			const module = await import("../../../embedding-runtime/index.ts");
			runtime = { ensureRuntime: module.ensureRuntime };
		}
		const result = await runtime.ensureRuntime({ profileId: options.profileId, cachedOnly: true });
		ensured = {
			baseUrl: result.baseUrl,
			identity: {
				provider: result.identity.provider,
				model: result.identity.model,
				dimension: result.identity.dimension,
			},
		};
		return ensured;
	}
	return {
		identity: async () => (await ensure()).identity,
		async embed(texts) {
			if (texts.length === 0) return [];
			const { baseUrl, identity } = await ensure();
			const fetchImpl = options.fetchImpl ?? fetch;
			const response = await fetchImpl(`${baseUrl}/v1/embeddings`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ model: identity.model, input: [...texts] }),
			});
			if (!response.ok) throw new Error(`gateway /v1/embeddings returned HTTP ${response.status}`);
			const json = (await response.json()) as { data?: { index?: number; embedding?: number[] }[] };
			const rows = [...(json.data ?? [])].sort((a, b) => (a.index ?? 0) - (b.index ?? 0));
			const embeddings = rows.map((row) => row.embedding ?? []);
			if (embeddings.length !== texts.length || embeddings.some((row) => row.length !== identity.dimension)) {
				throw new Error("gateway /v1/embeddings returned an invalid embedding batch");
			}
			return embeddings;
		},
	};
}
