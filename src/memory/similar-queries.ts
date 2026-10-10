import { createHash } from "node:crypto";
import { mkdirSync, rmSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { Embedder, EmbeddingIdentity } from "../embedding-runtime/gateway-embedder.ts";
import { bm25Scores } from "../retrieval/bm25.ts";
import type { JudgedEvidenceRecord } from "./judged-evidence.ts";

/** How many similar past questions memory brings back for one question. */
export const SIMILAR_QUESTION_LIMIT = 25;

/**
 * Cosine similarity below this is not a semantic match. Embedding models give
 * unrelated short texts a clearly positive cosine, so a bare "top N by
 * cosine" would always return something.
 */
export const VECTOR_MIN_SIMILARITY = 0.5;

/** Reciprocal-rank-fusion constant: how fast a lower rank stops mattering. */
const RRF_K = 60;
const VECTOR_STORE_VERSION = 1;

/** A past question and the evidence Jev kept for it, best evidence first. */
export interface SimilarQuestion {
	readonly question: string;
	/** Fused BM25 + vector rank score; only meaningful for ordering. */
	readonly score: number;
	readonly records: readonly JudgedEvidenceRecord[];
}

export interface SimilarQuestionsOptions {
	readonly limit?: number;
	/** Adds semantic matching; without it only BM25 ranks. */
	readonly embedder?: Embedder;
	/** Where question vectors are cached; unset keeps them in memory for this call only. */
	readonly vectorStorePath?: string;
}

export interface SimilarQuestionsResult {
	readonly questions: readonly SimilarQuestion[];
	/** Why semantic matching was skipped (verbatim error); absent when it ran or no embedder was given. */
	readonly fallbackReason?: string;
}

/** The vector cache lives beside the memory file it indexes. */
export function memoryVectorStorePath(memoryPath: string): string {
	return `${memoryPath}.vectors.db`;
}

function questionKey(question: string): string {
	return createHash("sha256").update(question).digest("hex");
}

function cosine(a: ArrayLike<number>, b: ArrayLike<number>): number {
	let dot = 0;
	let normA = 0;
	let normB = 0;
	for (let index = 0; index < Math.min(a.length, b.length); index++) {
		const x = a[index] ?? 0;
		const y = b[index] ?? 0;
		dot += x * y;
		normA += x * x;
		normB += y * y;
	}
	return normA === 0 || normB === 0 ? 0 : dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

/**
 * Question vectors persisted in a SQLite file: one little-endian Float32 BLOB
 * per question, keyed by the question's hash. A row is written the moment it
 * is embedded, so concurrent processes never overwrite each other's vectors
 * and a question is embedded at most once per embedding model.
 */
class VectorStore {
	private readonly db: DatabaseSync;

	private constructor(db: DatabaseSync) {
		this.db = db;
	}

	/**
	 * Opens (creating if needed) the store and empties it when it was built by a
	 * different embedding model. A file that is not a SQLite database is
	 * replaced: the cache is rebuilt from the questions on the next lookup.
	 */
	static open(path: string, identity: EmbeddingIdentity): VectorStore {
		mkdirSync(dirname(path), { recursive: true });
		try {
			return VectorStore.init(new DatabaseSync(path), identity);
		} catch {
			for (const suffix of ["", "-wal", "-shm"]) rmSync(`${path}${suffix}`, { force: true });
			return VectorStore.init(new DatabaseSync(path), identity);
		}
	}

	private static init(db: DatabaseSync, identity: EmbeddingIdentity): VectorStore {
		try {
			db.exec(`PRAGMA busy_timeout = 5000;
PRAGMA journal_mode = WAL;
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, vector BLOB NOT NULL) WITHOUT ROWID;`);
			const wanted = JSON.stringify([VECTOR_STORE_VERSION, identity.provider, identity.model, identity.dimension]);
			const stored = db.prepare("SELECT value FROM meta WHERE key = 'identity'").get() as
				| { value: string }
				| undefined;
			if (stored?.value !== wanted) {
				db.exec("BEGIN IMMEDIATE");
				db.exec("DELETE FROM vectors");
				db.prepare("INSERT OR REPLACE INTO meta (key, value) VALUES ('identity', ?)").run(wanted);
				db.exec("COMMIT");
			}
			return new VectorStore(db);
		} catch (error) {
			db.close();
			throw error;
		}
	}

	/** Every stored vector whose size matches the embedding dimension. */
	load(dimension: number): Map<string, Float32Array> {
		const vectors = new Map<string, Float32Array>();
		const rows = this.db.prepare("SELECT key, vector FROM vectors").all() as {
			key: string;
			vector: Uint8Array;
		}[];
		for (const row of rows) {
			if (row.vector.byteLength !== dimension * Float32Array.BYTES_PER_ELEMENT) continue;
			// SQLite hands out unaligned views, so copy into a fresh buffer.
			vectors.set(row.key, new Float32Array(row.vector.slice().buffer));
		}
		return vectors;
	}

	save(entries: readonly (readonly [string, Float32Array])[]): void {
		const insert = this.db.prepare("INSERT OR REPLACE INTO vectors (key, vector) VALUES (?, ?)");
		this.db.exec("BEGIN IMMEDIATE");
		try {
			for (const [key, vector] of entries) {
				insert.run(key, new Uint8Array(vector.buffer, vector.byteOffset, vector.byteLength));
			}
			this.db.exec("COMMIT");
		} catch (error) {
			this.db.exec("ROLLBACK");
			throw error;
		}
	}

	close(): void {
		this.db.close();
	}
}

/** Question indexes whose embedding is close enough to the query, closest first. */
async function rankByVector(
	questions: readonly string[],
	query: string,
	embedder: Embedder,
	vectorStorePath: string | undefined,
): Promise<number[]> {
	const identity = await embedder.identity();
	// Without a path the vectors live for this call only.
	let store: VectorStore | undefined;
	if (vectorStorePath !== undefined) {
		try {
			store = VectorStore.open(vectorStorePath, identity);
		} catch {
			// An unusable cache only costs re-embedding; the lookup itself carries on.
		}
	}
	try {
		const vectors = store?.load(identity.dimension) ?? new Map<string, Float32Array>();
		const missing = questions.filter((question) => !vectors.has(questionKey(question)));
		const embedded = await embedder.embed([query, ...missing]);
		const queryVector = embedded[0];
		if (queryVector === undefined || embedded.length !== missing.length + 1) {
			throw new Error("embedder returned a different number of vectors than texts");
		}
		const fresh: [string, Float32Array][] = [];
		for (const [index, question] of missing.entries()) {
			const vector = embedded[index + 1];
			if (vector === undefined || vector.length !== identity.dimension) continue;
			const entry: [string, Float32Array] = [questionKey(question), Float32Array.from(vector)];
			vectors.set(entry[0], entry[1]);
			fresh.push(entry);
		}
		try {
			if (fresh.length > 0) store?.save(fresh);
		} catch {
			// The vector cache is best-effort: a failed write only means re-embedding next time.
		}
		return questions
			.map((question, index) => ({
				index,
				similarity: cosine(queryVector, vectors.get(questionKey(question)) ?? []),
			}))
			.filter((entry) => entry.similarity >= VECTOR_MIN_SIMILARITY)
			.sort((a, b) => b.similarity - a.similarity || a.index - b.index)
			.map((entry) => entry.index);
	} finally {
		store?.close();
	}
}

/**
 * Past questions that resemble `query`, with the evidence Jev judged to
 * support each. Keyword (BM25) and semantic (vector) rankings are fused by
 * reciprocal rank, so a question found by both outranks one found by either
 * alone, and a rephrasing that shares no word with the query is still found
 * semantically. Never throws: a failed embedder degrades to keywords only and
 * says why.
 */
export async function findSimilarQuestions(
	records: readonly JudgedEvidenceRecord[],
	query: string,
	options: SimilarQuestionsOptions = {},
): Promise<SimilarQuestionsResult> {
	const groups = new Map<string, JudgedEvidenceRecord[]>();
	for (const record of records) {
		const group = groups.get(record.question);
		if (group === undefined) groups.set(record.question, [record]);
		else group.push(record);
	}
	const questions = [...groups.keys()];
	if (questions.length === 0 || query.trim().length === 0) return { questions: [] };

	let vectorRanking: number[] = [];
	let fallbackReason: string | undefined;
	if (options.embedder !== undefined) {
		try {
			vectorRanking = await rankByVector(questions, query, options.embedder, options.vectorStorePath);
		} catch (error) {
			fallbackReason = `Semantic matching of similar questions was unavailable; ranked by keywords only. ${error instanceof Error ? error.message : String(error)}`;
		}
	}

	const scores = new Map<number, number>();
	const keywordRanking = bm25Scores(query, questions)
		.map((score, index) => ({ index, score }))
		.filter((entry) => entry.score > 0)
		.sort((a, b) => b.score - a.score || a.index - b.index)
		.map((entry) => entry.index);
	for (const ranking of [keywordRanking, vectorRanking]) {
		for (const [rank, index] of ranking.entries()) {
			scores.set(index, (scores.get(index) ?? 0) + 1 / (RRF_K + rank + 1));
		}
	}
	const ranked: { entry: SimilarQuestion; newestAt: number }[] = [];
	for (const [index, score] of scores) {
		const question = questions[index];
		const group = question === undefined ? undefined : groups.get(question);
		if (question === undefined || group === undefined) continue;
		const records = [...group].sort((a, b) => b.probability - a.probability || b.createdAt - a.createdAt);
		ranked.push({
			entry: { question, score, records },
			newestAt: Math.max(...records.map((record) => record.createdAt)),
		});
	}
	ranked.sort((a, b) => b.entry.score - a.entry.score || b.newestAt - a.newestAt);
	return {
		questions: ranked.slice(0, options.limit ?? SIMILAR_QUESTION_LIMIT).map(({ entry }) => entry),
		...(fallbackReason !== undefined ? { fallbackReason } : {}),
	};
}
