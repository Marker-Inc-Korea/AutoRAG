/**
 * Crawler sidecar: BM25 + semantic + hybrid retrieval over a crawler's own
 * SQLite archive without touching it.
 *
 * The sidecar owns a separate database file (MAIN). The crawler's database is
 * ATTACHed read-only, so a crawler that migrates, purges, vacuums or rebuilds
 * its file cannot lose sidecar state, and the sidecar cannot corrupt the
 * crawler. Only vectors keyed by the crawler's message key are stored; message
 * bodies are always read from the crawler database at query time.
 *
 * Everything crawler-specific lives in a {@link CrawlerDescriptor}; this file
 * is shared by every crawler.
 */

import { createHash } from "node:crypto";
import { mkdirSync, statSync } from "node:fs";
import { dirname } from "node:path";
import { DatabaseSync, type SQLInputValue } from "node:sqlite";
import { pathToFileURL } from "node:url";
import type { Embedder, EmbeddingIdentity } from "../../embedding-runtime/gateway-embedder.ts";
import type { CrawlerDescriptor } from "./descriptors.ts";
import {
	decodeVector,
	encodeVector,
	type KeyedVector,
	reciprocalRankFusion,
	type ScoredKey,
	topKByCosine,
} from "./fusion.ts";

export interface SidecarHit {
	readonly key: string;
	readonly content: string;
	readonly score: number;
	readonly title?: string;
	/** Milliseconds since epoch. */
	readonly publishedAt?: number;
	readonly metadata: Readonly<Record<string, string>>;
}

export type SidecarSyncResult =
	| {
			readonly ok: true;
			readonly embedded: number;
			/** Messages the embedder rejected on their own; retried only after their text changes. */
			readonly skipped: number;
			readonly pruned: number;
			/** Messages still waiting for a vector after this sync (0 when fully caught up). */
			readonly remaining: number;
	  }
	| { readonly ok: false; readonly message: string };

export interface CrawlerSidecarOptions {
	readonly descriptor: CrawlerDescriptor;
	readonly crawlerDbPath: string;
	readonly sidecarDbPath: string;
	/** Restricts every query to one workspace/account when the descriptor supports scoping. */
	readonly workspaceId?: string;
	/**
	 * Chat/channel allowlist (`channels.ids/names`). Empty means every chat. Applied in SQL, so excluded chats are
	 * never searched, embedded, or kept as vectors.
	 */
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
	/** Required for semantic and hybrid search; lexical search works without it. */
	readonly embedder?: Embedder;
	/** Messages per embedding request and per write transaction. Default {@link DEFAULT_BATCH_SIZE}. */
	readonly batchSize?: number;
	/**
	 * Most messages one `sync()` will embed. A first sync over a large archive otherwise holds the whole
	 * refresh for minutes; the remainder is picked up by the next sync, and BM25 never waits for vectors.
	 * Default {@link DEFAULT_MAX_EMBED_PER_SYNC}.
	 */
	readonly maxEmbedPerSync?: number;
}

/** Messages embedded per request and per write transaction; bounds memory and keeps a failed sync resumable. */
const DEFAULT_BATCH_SIZE = 256;
/** About 9 s on the local gateway at the measured ~224 msg/s; 16.6k messages catch up in 9 refreshes. */
const DEFAULT_MAX_EMBED_PER_SYNC = 2048;
/**
 * Characters of one message sent to the embedder. The gateway rejects a request over 262,144 characters
 * (HTTP 413) and the embedder sends 32 inputs per request, so 8192 keeps any full request under the cap.
 * A longer message is embedded by its head; BM25 still indexes all of it.
 */
const MAX_EMBED_INPUT_CHARS = 8192;
/** How long a write waits for another process (refresh vs. TUI/serve poller) holding the sidecar's write lock. */
const BUSY_TIMEOUT_MS = 5000;
/** Candidate pool multiplier: over-fetch so scope/deleted filtering and fusion still fill `topK`. */
const POOL_FACTOR = 3;
/** How long one query's embedding and ranking are reused (the methods of one search run within it). */
const SEMANTIC_REUSE_MS = 10_000;
const IDENTITY_KEY = "identity";

type Row = Record<string, unknown>;

/** One query's embedding (shared, in flight or settled) and its best-`pool` ranking once computed. */
interface SemanticRanking {
	readonly query: string;
	readonly at: number;
	readonly vector: Promise<readonly number[] | undefined>;
	ranked: { readonly pool: number; readonly keys: ScoredKey[] } | undefined;
}

/** A crawler message normalized for search: display fields plus the embedding input and its hash. */
interface SidecarRow {
	readonly key: string;
	readonly content: string;
	readonly title: string | undefined;
	/** Seconds since epoch, when the crawler records one. */
	readonly epoch: number | undefined;
	readonly metadata: Record<string, string>;
	/** Exact text sent to the embedder; its hash decides whether a stored vector is stale. */
	readonly input: string;
	readonly hash: string;
}

export class CrawlerSidecar {
	readonly descriptor: CrawlerDescriptor;
	private readonly options: CrawlerSidecarOptions;
	private readonly select: string;
	private readonly where: string;
	/** Opened on first use: building a skill must not create files, directories, or file descriptors. */
	private opened: DatabaseSync | undefined;
	/** Identity of the attached crawler file; a crawler that replaces or deletes its archive forces a re-attach. */
	private attached: { readonly dev: number; readonly ino: number } | undefined;
	/**
	 * The last query's embedding and ranking. The semantic and hybrid methods of one search ask for the same
	 * query, so they share one embedding call and one vector scan. Short-lived, and dropped on any write.
	 */
	private lastSemantic: SemanticRanking | undefined;

	constructor(options: CrawlerSidecarOptions) {
		this.options = options;
		const { descriptor } = options;
		this.descriptor = descriptor;
		const metadataColumns = descriptor.metadata.map((column, index) => `${column.sql} as m${index}`);
		this.select = [
			`${descriptor.keyExpression} as k`,
			`${descriptor.textExpression} as content`,
			`${descriptor.titleExpression ?? "null"} as title`,
			`${descriptor.epochSecondsExpression ?? "null"} as epoch`,
			...metadataColumns,
		].join(", ");
		const scope = descriptor.scopeExpression !== undefined && options.workspaceId !== undefined;
		const allowlist = [
			...((options.channelIds?.length ?? 0) > 0
				? [`${descriptor.channelIdExpression} in (${options.channelIds?.map(() => "?").join(", ")})`]
				: []),
			...((options.channelNames?.length ?? 0) > 0
				? [`${descriptor.channelNameExpression} in (${options.channelNames?.map(() => "?").join(", ")})`]
				: []),
		];
		this.where = [
			descriptor.liveWhere,
			...(scope ? [`${descriptor.scopeExpression} = ?`] : []),
			...(allowlist.length > 0 ? [`(${allowlist.join(" or ")})`] : []),
		].join(" and ");
	}

	/** The sidecar database, created on first call. Only reached once the crawler archive exists. */
	private get db(): DatabaseSync {
		if (this.opened !== undefined) return this.opened;
		mkdirSync(dirname(this.options.sidecarDbPath), { recursive: true });
		const db = new DatabaseSync(this.options.sidecarDbPath);
		db.exec(`pragma busy_timeout = ${BUSY_TIMEOUT_MS}`);
		db.exec("pragma journal_mode = wal");
		db.exec(`
			create table if not exists sidecar_meta (k text primary key, v text not null);
			create table if not exists sidecar_vec (message_key text primary key, hash text not null, vec blob not null);
		`);
		this.opened = db;
		return db;
	}

	/**
	 * Attach the crawler archive read-only, once it exists and has the expected
	 * shape. The crawler may not have synced yet, or may have replaced or deleted
	 * its file, so this is re-checked on every call: a changed file is re-attached,
	 * a missing one detached. Nothing is created on disk until the archive exists.
	 */
	private attachCrawler(): boolean {
		let file: { dev: number; ino: number };
		try {
			const { dev, ino } = statSync(this.options.crawlerDbPath);
			file = { dev, ino };
		} catch (error) {
			this.detachCrawler();
			if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
			throw error;
		}
		if (this.attached !== undefined) {
			if (this.attached.dev === file.dev && this.attached.ino === file.ino) return true;
			this.detachCrawler();
		}
		const crawlerUri = `${pathToFileURL(this.options.crawlerDbPath).href}?mode=ro`.replaceAll("'", "''");
		// A present but unreadable archive is a retrieval failure, not an empty index.
		this.db.exec(`attach database '${crawlerUri}' as crawler`);
		const { descriptor } = this.options;
		try {
			this.db
				.prepare(`select ${this.select} from ${descriptor.fromClause} where ${this.where} limit 0`)
				.all(...this.whereParams());
			this.db.prepare(`select 1 from ${descriptor.fts.table} limit 0`).all();
		} catch (error) {
			this.db.exec("detach database crawler");
			throw error;
		}
		this.attached = file;
		this.lastSemantic = undefined;
		return true;
	}

	private detachCrawler(): void {
		if (this.attached === undefined) return;
		this.attached = undefined;
		this.lastSemantic = undefined;
		try {
			this.opened?.exec("detach database crawler");
		} catch {
			// Already detached; nothing to release.
		}
	}

	/** True when a semantic index can be built: an embedder is configured. */
	get semanticEnabled(): boolean {
		return this.options.embedder !== undefined;
	}

	/** Release the database handle. Safe to call repeatedly; a later search reopens and re-attaches. */
	close(): void {
		this.opened?.close();
		this.opened = undefined;
		this.attached = undefined;
		this.lastSemantic = undefined;
	}

	/** Bound parameters for {@link where}, in the order its predicates appear. */
	private whereParams(): SQLInputValue[] {
		const { descriptor, workspaceId, channelIds = [], channelNames = [] } = this.options;
		return [
			...(descriptor.scopeExpression !== undefined && workspaceId !== undefined ? [workspaceId] : []),
			...channelIds,
			...channelNames,
		];
	}

	/** BM25-ranked lexical search. Natural-language queries are OR-ed term by term. */
	searchLexical(query: string, topK: number): SidecarHit[] {
		const match = toMatchQuery(query);
		if (match === undefined || topK <= 0 || !this.attachCrawler()) return [];
		const { descriptor } = this.options;
		const rows = this.db
			.prepare(
				`select ${this.select}, bm25(${descriptor.fts.table}) as rank
				from ${descriptor.fromClause} join ${descriptor.fts.table} on ${descriptor.fts.joinOn}
				where ${descriptor.fts.table} match ? and ${this.where}
				order by rank limit ?`,
			)
			.all(match, ...this.whereParams(), topK) as Row[];
		return rows.map((row) => this.toHit(this.toRow(row), -Number(row.rank)));
	}

	/** Cosine search over embedded messages. Empty when there is no embedder or the query embedding fails. */
	async searchSemantic(query: string, topK: number): Promise<SidecarHit[]> {
		const trimmed = query.trim();
		if (this.options.embedder === undefined || trimmed.length === 0 || topK <= 0 || !this.attachCrawler()) return [];
		const ranked = await this.rankSemantic(trimmed, topK * POOL_FACTOR);
		if (ranked.length === 0) return [];
		const live = this.hydrate(ranked.map((entry) => entry.key));
		const hits: SidecarHit[] = [];
		for (const entry of ranked) {
			const row = live.get(entry.key);
			if (row !== undefined) hits.push(this.toHit(row, entry.score));
			if (hits.length >= topK) break;
		}
		return hits;
	}

	/** Best `pool` stored vectors for `query`, reusing the previous ranking when it already covers `pool`. */
	private async rankSemantic(query: string, pool: number): Promise<ScoredKey[]> {
		const now = Date.now();
		let cached = this.lastSemantic;
		if (cached === undefined || cached.query !== query || now - cached.at > SEMANTIC_REUSE_MS) {
			const embedder = this.options.embedder;
			if (embedder === undefined) return [];
			cached = {
				query,
				at: now,
				vector: embedder.embed([query]).then(
					(vectors) => vectors[0],
					() => undefined,
				),
				ranked: undefined,
			};
			this.lastSemantic = cached;
		}
		const vector = await cached.vector;
		if (vector === undefined) {
			// Never keep a failed query embedding: the next search retries the gateway.
			if (this.lastSemantic === cached) this.lastSemantic = undefined;
			return [];
		}
		if (cached.ranked === undefined || cached.ranked.pool < pool) {
			// The scan scores and sorts every positive row whatever the pool, so ranking a wider pool costs nothing
			// extra and lets semantic (topK × 3) and hybrid (topK × 9) share one scan in either call order.
			const wide = pool * POOL_FACTOR;
			cached.ranked = { pool: wide, keys: topKByCosine(vector, this.storedVectors(), wide) };
		}
		return cached.ranked.keys.slice(0, pool);
	}

	/** Reciprocal-rank fusion of lexical and semantic rankings; lexical-only when semantic is unavailable. */
	async searchHybrid(query: string, topK: number): Promise<SidecarHit[]> {
		if (topK <= 0) return [];
		const pool = topK * POOL_FACTOR;
		const lexical = this.searchLexical(query, pool);
		const semantic = await this.searchSemantic(query, pool);
		if (semantic.length === 0) return lexical.slice(0, topK);
		const byKey = new Map<string, SidecarHit>();
		for (const hit of [...semantic, ...lexical]) byKey.set(hit.key, hit);
		const fused: SidecarHit[] = [];
		for (const entry of reciprocalRankFusion([lexical.map((hit) => hit.key), semantic.map((hit) => hit.key)])) {
			const hit = byKey.get(entry.key);
			if (hit !== undefined) fused.push({ ...hit, score: entry.score });
			if (fused.length >= topK) break;
		}
		return fused;
	}

	/**
	 * Bring the vector table in line with the crawler archive: drop vectors of
	 * purged/deleted messages, embed new or edited ones, re-embed all on an
	 * embedding identity change. An embedder failure keeps what was already
	 * stored. A message the embedder keeps rejecting while others succeed is
	 * skipped until its text changes, so it cannot stall the backlog behind it.
	 * Database errors (for example a lock held past the busy timeout) throw.
	 */
	async sync(): Promise<SidecarSyncResult> {
		const { embedder } = this.options;
		if (embedder === undefined) return { ok: false, message: "no embedder configured" };
		if (!this.attachCrawler()) return { ok: false, message: "crawler archive not available yet" };
		let identity: EmbeddingIdentity;
		try {
			identity = await embedder.identity();
		} catch (error) {
			return { ok: false, message: errorMessage(error) };
		}
		this.adoptIdentity(identity);
		const pruned = this.prune();
		const pending = this.pendingKeys();
		const batchSize = Math.max(1, this.options.batchSize ?? DEFAULT_BATCH_SIZE);
		const budget = Math.max(1, this.options.maxEmbedPerSync ?? DEFAULT_MAX_EMBED_PER_SYNC);
		const target = Math.min(pending.length, budget);
		const embed = async (rows: readonly SidecarRow[]): Promise<readonly (readonly number[])[]> => {
			const vectors = await embedder.embed(rows.map((row) => row.input));
			if (vectors.length !== rows.length) {
				throw new Error(`embedder returned ${vectors.length} vectors for ${rows.length} messages`);
			}
			const wrong = vectors.find((vector) => vector.length !== identity.dimension);
			if (wrong !== undefined) {
				throw new Error(`embedder returned a ${wrong.length}-dimension vector, expected ${identity.dimension}`);
			}
			return vectors;
		};
		let embedded = 0;
		let skipped = 0;
		for (let offset = 0; offset < target; offset += batchSize) {
			const rows = [...this.hydrate(pending.slice(offset, Math.min(offset + batchSize, target))).values()];
			if (rows.length === 0) continue;
			try {
				this.store(rows, await embed(rows));
				embedded += rows.length;
			} catch (batchError) {
				// Find the rows that fail on their own. Skip them only once another row has embedded in this
				// sync, which proves the embedder is up; otherwise stop, keeping everything pending.
				const rejected: SidecarRow[] = [];
				let lastError: unknown = batchError;
				for (const row of rows) {
					try {
						this.store([row], await embed([row]));
						embedded += 1;
					} catch (error) {
						rejected.push(row);
						lastError = error;
					}
				}
				if (rejected.length === 0) continue;
				if (embedded === 0) return { ok: false, message: errorMessage(lastError) };
				this.store(rejected, undefined);
				skipped += rejected.length;
			}
		}
		return { ok: true, embedded, skipped, pruned, remaining: pending.length - embedded - skipped };
	}

	private adoptIdentity(identity: EmbeddingIdentity): void {
		const serialized = JSON.stringify(identity);
		const current = this.db.prepare("select v from sidecar_meta where k = ?").get(IDENTITY_KEY) as Row | undefined;
		if (current?.v === serialized) return;
		this.transaction(() => {
			this.db.exec("delete from sidecar_vec");
			this.db.prepare("insert or replace into sidecar_meta (k, v) values (?, ?)").run(IDENTITY_KEY, serialized);
		});
		this.lastSemantic = undefined;
	}

	/** Delete vectors whose message no longer exists, is deleted, or fell out of scope. */
	private prune(): number {
		const { descriptor } = this.options;
		const result = this.db
			.prepare(
				`delete from sidecar_vec where message_key not in (
					select ${descriptor.keyExpression} from ${descriptor.fromClause} where ${this.where}
				)`,
			)
			.run(...this.whereParams());
		if (Number(result.changes) > 0) this.lastSemantic = undefined;
		return Number(result.changes);
	}

	/** Keys with no vector yet, or whose embedded text changed since. */
	private pendingKeys(): string[] {
		const { descriptor } = this.options;
		const rows = this.db
			.prepare(
				`select ${this.select}, v.hash as stored_hash
				from ${descriptor.fromClause}
				left join sidecar_vec v on v.message_key = ${descriptor.keyExpression}
				where ${this.where}`,
			)
			.iterate(...this.whereParams()) as Iterable<Row>;
		const pending: string[] = [];
		for (const row of rows) {
			if (row.stored_hash !== this.toRow(row).hash) pending.push(String(row.k));
		}
		return pending;
	}

	/** Live rows for the given keys, in the order given. */
	private hydrate(keys: readonly string[]): Map<string, SidecarRow> {
		const { descriptor } = this.options;
		const out = new Map<string, SidecarRow>();
		if (keys.length === 0) return out;
		const rows = this.db
			.prepare(
				`select ${this.select} from ${descriptor.fromClause}
				where ${descriptor.keyExpression} in (${keys.map(() => "?").join(", ")}) and ${this.where}`,
			)
			.all(...keys, ...this.whereParams()) as Row[];
		const byKey = new Map(rows.map((row) => [String(row.k), this.toRow(row)]));
		for (const key of keys) {
			const row = byKey.get(key);
			if (row !== undefined) out.set(key, row);
		}
		return out;
	}

	/**
	 * Upsert vectors keyed by message. `undefined` stores an empty vector: the message is recorded as tried at
	 * this text hash (so it is not retried every sync) but never ranked.
	 */
	private store(rows: readonly SidecarRow[], vectors: readonly (readonly number[])[] | undefined): void {
		const upsert = this.db.prepare("insert or replace into sidecar_vec (message_key, hash, vec) values (?, ?, ?)");
		this.transaction(() => {
			for (const [index, row] of rows.entries()) {
				upsert.run(row.key, row.hash, encodeVector(vectors?.[index] ?? []));
			}
		});
		this.lastSemantic = undefined;
	}

	private *storedVectors(): Generator<KeyedVector> {
		const rows = this.db
			.prepare("select message_key, vec from sidecar_vec where length(vec) > 0")
			.iterate() as Iterable<Row>;
		for (const row of rows) {
			const vector = row.vec instanceof Uint8Array ? decodeVector(row.vec) : undefined;
			if (vector !== undefined) yield { key: String(row.message_key), vector };
		}
	}

	private transaction(work: () => void): void {
		this.db.exec("begin immediate");
		try {
			work();
			this.db.exec("commit");
		} catch (error) {
			this.db.exec("rollback");
			throw error;
		}
	}

	private toRow(row: Row): SidecarRow {
		const metadata: Record<string, string> = {};
		for (const [index, column] of this.options.descriptor.metadata.entries()) {
			const value = row[`m${index}`];
			if ((typeof value === "string" && value.length > 0) || typeof value === "number")
				metadata[column.key] = String(value);
		}
		const title = typeof row.title === "string" && row.title.length > 0 ? row.title : undefined;
		const epoch = typeof row.epoch === "number" && Number.isFinite(row.epoch) ? row.epoch : undefined;
		const content = String(row.content ?? "");
		const input = `${title ?? ""}\n${content}`.slice(0, MAX_EMBED_INPUT_CHARS);
		return {
			key: String(row.k),
			content,
			title,
			epoch,
			metadata,
			input,
			hash: createHash("sha1").update(input).digest("hex"),
		};
	}

	private toHit(row: SidecarRow, score: number): SidecarHit {
		return {
			key: row.key,
			content: row.content,
			score,
			...(row.title !== undefined ? { title: row.title } : {}),
			...(row.epoch !== undefined ? { publishedAt: Math.round(row.epoch * 1000) } : {}),
			metadata: row.metadata,
		};
	}
}

/**
 * FTS5 MATCH expression for free text: every alphanumeric term quoted and
 * OR-ed. Quoting neutralizes FTS operators (AND, NOT, NEAR, *, parentheses),
 * so user text can never change the query's structure. Undefined when the
 * text contains no searchable term.
 */
export function toMatchQuery(text: string): string | undefined {
	const terms = [...new Set(text.match(/[\p{L}\p{N}_]+/gu) ?? [])];
	return terms.length === 0 ? undefined : terms.map((term) => `"${term}"`).join(" OR ");
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}
