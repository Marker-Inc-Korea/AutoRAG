/**
 * Persistent chunk store shared by the connector-backed datasource skills.
 *
 * Documents fetched by a {@link DatasourceConnector} are split into bounded
 * chunks, persisted as JSON under
 * `<workspaceRoot>/.autorag/datasources/<skill>/<instance>/chunks.json`, and
 * searched with a lightweight BM25-style lexical scorer. The store never
 * throws from load/search; persistence failures degrade to in-memory-only
 * operation.
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { bm25Scores, tokenize } from "../retrieval/bm25.ts";
import type { ConnectorDocument } from "./connector.ts";
import { sanitizeIdSegment } from "./connector.ts";

export interface StoredChunk {
	readonly chunkId: string;
	readonly docId: string;
	/** Sanitized hierarchy segments under the instance root. */
	readonly hierarchy: readonly string[];
	readonly title?: string;
	readonly content: string;
	readonly publishedAt?: number;
	readonly metadata: Readonly<Record<string, unknown>>;
}

export interface ScoredChunk {
	readonly chunk: StoredChunk;
	readonly score: number;
}

export interface DatasourceChunkStoreOptions {
	readonly skillName: string;
	readonly instanceId: string;
	/** When set, chunks are persisted under `.autorag/datasources/…`. */
	readonly workspaceRoot?: string;
	/** Maximum characters per chunk. Default 2000. */
	readonly maxChunkChars?: number;
}

const STORE_VERSION = 1;
const DEFAULT_MAX_CHUNK_CHARS = 2000;

interface PersistedStore {
	readonly version: number;
	readonly skill: string;
	readonly instance: string;
	readonly updatedAt: number;
	readonly chunks: readonly StoredChunk[];
}

export function datasourceChunkStorePath(workspaceRoot: string, skillName: string, instanceId: string): string {
	return join(
		workspaceRoot,
		".autorag",
		"datasources",
		sanitizeIdSegment(skillName),
		sanitizeIdSegment(instanceId),
		"chunks.json",
	);
}

export class DatasourceChunkStore {
	private readonly skillName: string;
	private readonly instanceId: string;
	private readonly storePath: string | undefined;
	private readonly maxChunkChars: number;
	private chunkList: StoredChunk[] = [];
	private loaded = false;

	constructor(options: DatasourceChunkStoreOptions) {
		this.skillName = options.skillName;
		this.instanceId = options.instanceId;
		this.maxChunkChars = options.maxChunkChars ?? DEFAULT_MAX_CHUNK_CHARS;
		this.storePath =
			options.workspaceRoot === undefined
				? undefined
				: datasourceChunkStorePath(options.workspaceRoot, options.skillName, options.instanceId);
	}

	/**
	 * Replace the store contents with chunks derived from `documents`.
	 * Returns the resulting chunk count. Persistence failures are swallowed
	 * (in-memory contents stay authoritative for this process).
	 */
	replaceDocuments(documents: readonly ConnectorDocument[]): number {
		this.chunkList = buildChunks(documents, this.maxChunkChars);
		this.loaded = true;
		this.persist();
		return this.chunkList.length;
	}

	/** Apply file-level additions/updates/deletions without rebuilding unchanged chunks. */
	updateDocuments(documents: readonly ConnectorDocument[], deletedDocIds: readonly string[] = []): number {
		this.ensureLoaded();
		const replaced = new Set(documents.map((document) => document.docId));
		const deleted = new Set(deletedDocIds);
		const unchanged = this.chunkList.filter((chunk) => !replaced.has(chunk.docId) && !deleted.has(chunk.docId));
		const additions = buildChunks(documents, this.maxChunkChars, new Set(unchanged.map((chunk) => chunk.chunkId)));
		this.chunkList = [...unchanged, ...additions];
		this.loaded = true;
		this.persist();
		return this.chunkList.length;
	}

	/** Load persisted chunks when present. Returns true when data was loaded. */
	load(): boolean {
		if (this.storePath === undefined) return false;
		try {
			if (!existsSync(this.storePath)) return false;
			const parsed = JSON.parse(readFileSync(this.storePath, "utf8")) as PersistedStore;
			if (parsed.version !== STORE_VERSION || !Array.isArray(parsed.chunks)) return false;
			this.chunkList = parsed.chunks.filter(
				(chunk) => typeof chunk?.chunkId === "string" && typeof chunk?.content === "string",
			);
			this.loaded = true;
			return true;
		} catch {
			return false;
		}
	}

	/** Ensure persisted chunks are loaded once for read paths. */
	ensureLoaded(): void {
		if (!this.loaded) {
			this.load();
			this.loaded = true;
		}
	}

	chunks(): readonly StoredChunk[] {
		this.ensureLoaded();
		return this.chunkList;
	}

	/** Distinct hierarchy prefixes present in the store (for source listing). */
	hierarchies(): readonly (readonly string[])[] {
		this.ensureLoaded();
		const seen = new Map<string, readonly string[]>();
		for (const chunk of this.chunkList) {
			seen.set(chunk.hierarchy.join("/"), chunk.hierarchy);
		}
		return [...seen.values()];
	}

	/**
	 * BM25-style lexical search over the stored chunks. Empty queries and
	 * empty stores return `[]`; this method never throws. Scoring is delegated
	 * to {@link bm25Scores}.
	 */
	search(query: string, topK: number): readonly ScoredChunk[] {
		this.ensureLoaded();
		if (tokenize(query).length === 0 || this.chunkList.length === 0 || topK <= 0) return [];

		const scores = bm25Scores(
			query,
			this.chunkList.map((chunk) => `${chunk.title ?? ""} ${chunk.content}`),
		);
		const scored: ScoredChunk[] = [];
		for (const [index, chunk] of this.chunkList.entries()) {
			const score = scores[index] ?? 0;
			if (score > 0) scored.push({ chunk, score });
		}
		scored.sort((a, b) => b.score - a.score || a.chunk.chunkId.localeCompare(b.chunk.chunkId));
		return scored.slice(0, topK);
	}

	private persist(): void {
		if (this.storePath === undefined) return;
		try {
			mkdirSync(dirname(this.storePath), { recursive: true });
			const payload: PersistedStore = {
				version: STORE_VERSION,
				skill: this.skillName,
				instance: this.instanceId,
				updatedAt: Date.now(),
				chunks: this.chunkList,
			};
			writeFileSync(this.storePath, `${JSON.stringify(payload)}\n`, "utf8");
		} catch {
			// Persistence is best-effort; in-memory contents stay authoritative.
		}
	}
}

function buildChunks(
	documents: readonly ConnectorDocument[],
	maxChunkChars: number,
	seenIds = new Set<string>(),
): StoredChunk[] {
	const chunks: StoredChunk[] = [];
	for (const document of documents) {
		const docSegment = sanitizeIdSegment(document.docId);
		const hierarchy = (document.hierarchy ?? []).map(sanitizeIdSegment).filter((segment) => segment.length > 0);
		const parts = splitContent(document.content, maxChunkChars);
		for (const [index, part] of parts.entries()) {
			let chunkId = parts.length === 1 ? docSegment : `${docSegment}-p${index + 1}`;
			while (seenIds.has(chunkId)) chunkId = `${chunkId}x`;
			seenIds.add(chunkId);
			chunks.push({
				chunkId,
				docId: document.docId,
				hierarchy,
				...(document.title !== undefined ? { title: document.title } : {}),
				content: part,
				...(document.publishedAt !== undefined ? { publishedAt: document.publishedAt } : {}),
				metadata: { ...(document.metadata ?? {}) },
			});
		}
	}
	return chunks;
}

function splitContent(content: string, maxChars: number): readonly string[] {
	const trimmed = content.trim();
	if (trimmed.length === 0) return [];
	if (trimmed.length <= maxChars) return [trimmed];
	const parts: string[] = [];
	const paragraphs = trimmed.split(/\n{2,}/u);
	let current = "";
	for (const paragraph of paragraphs) {
		if (current.length > 0 && current.length + paragraph.length + 2 > maxChars) {
			parts.push(current);
			current = "";
		}
		if (paragraph.length > maxChars) {
			if (current.length > 0) {
				parts.push(current);
				current = "";
			}
			for (let offset = 0; offset < paragraph.length; offset += maxChars) {
				parts.push(paragraph.slice(offset, offset + maxChars));
			}
			continue;
		}
		current = current.length === 0 ? paragraph : `${current}\n\n${paragraph}`;
	}
	if (current.length > 0) parts.push(current);
	return parts.filter((part) => part.trim().length > 0);
}
