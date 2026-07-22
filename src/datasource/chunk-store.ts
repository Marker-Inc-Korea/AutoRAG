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
		const chunks: StoredChunk[] = [];
		const seenIds = new Set<string>();
		for (const document of documents) {
			const docSegment = sanitizeIdSegment(document.docId);
			const hierarchy = (document.hierarchy ?? []).map(sanitizeIdSegment).filter((segment) => segment.length > 0);
			const parts = splitContent(document.content, this.maxChunkChars);
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
		this.chunkList = chunks;
		this.loaded = true;
		this.persist();
		return chunks.length;
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
	 * empty stores return `[]`; this method never throws.
	 *
	 * Terms match document tokens exactly or as a prefix (min 2 chars), so
	 * agglutinative suffixes — Korean particles/endings (인증서 → 인증서를) and
	 * English inflections (index → indexing) — still rank. Prefix hits are
	 * slightly discounted against exact hits.
	 */
	search(query: string, topK: number): readonly ScoredChunk[] {
		this.ensureLoaded();
		const queryTerms = [...new Set(tokenize(query))];
		if (queryTerms.length === 0 || this.chunkList.length === 0 || topK <= 0) return [];

		const docCount = this.chunkList.length;
		const tokenized = this.chunkList.map((chunk) => tokenize(`${chunk.title ?? ""} ${chunk.content}`));
		const avgLength = tokenized.reduce((sum, tokens) => sum + tokens.length, 0) / docCount || 1;

		// Weighted term frequency per (term, doc): exact = 1, prefix = 0.75.
		const termFrequency = (term: string, tokens: readonly string[]): number => {
			let tf = 0;
			for (const token of tokens) {
				if (token === term) tf += 1;
				else if (term.length >= 2 && token.startsWith(term)) tf += 0.75;
			}
			return tf;
		};
		const frequencies = queryTerms.map((term) => tokenized.map((tokens) => termFrequency(term, tokens)));
		const documentFrequency = frequencies.map((perDoc) => perDoc.reduce((count, tf) => count + (tf > 0 ? 1 : 0), 0));

		const k1 = 1.2;
		const b = 0.75;
		const scored: ScoredChunk[] = [];
		for (const [index, chunk] of this.chunkList.entries()) {
			const tokens = tokenized[index] ?? [];
			let score = 0;
			for (const [termIndex] of queryTerms.entries()) {
				const df = documentFrequency[termIndex] ?? 0;
				if (df === 0) continue;
				const tf = frequencies[termIndex]?.[index] ?? 0;
				if (tf === 0) continue;
				const idf = Math.log(1 + (docCount - df + 0.5) / (df + 0.5));
				score += (idf * tf * (k1 + 1)) / (tf + k1 * (1 - b + (b * tokens.length) / avgLength));
			}
			if (score > 0) scored.push({ chunk, score });
		}
		scored.sort((a, b2) => b2.score - a.score || a.chunk.chunkId.localeCompare(b2.chunk.chunkId));
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

function tokenize(text: string): string[] {
	return (
		text
			.toLowerCase()
			// Split on anything that is not a letter, digit, or Hangul syllable.
			.split(/[^\p{L}\p{N}]+/u)
			.filter((token) => token.length > 0)
	);
}
