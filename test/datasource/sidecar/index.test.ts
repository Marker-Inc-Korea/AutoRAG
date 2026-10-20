import { existsSync, mkdtempSync, renameSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { CRAWLER_DESCRIPTORS } from "../../../src/datasource/sidecar/descriptors.ts";
import { CrawlerSidecar } from "../../../src/datasource/sidecar/index.ts";
import type { Embedder } from "../../../src/embedding/embedder.ts";

let root: string;
let crawlerDb: string;
let sidecarDb: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-sidecar-"));
	crawlerDb = join(root, "slacrawl.db");
	sidecarDb = join(root, "sidecar.sqlite");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** The slice of slacrawl's schema the descriptor reads: messages joined to message_fts by "channel|ts". */
function seedSlack(rows: readonly { channel: string; ts: string; text: string; user?: string }[]): void {
	const db = new DatabaseSync(crawlerDb);
	db.exec(`
		create table workspaces (id text primary key, name text);
		create table channels (id text primary key, workspace_id text, name text);
		create table users (id text primary key, name text, real_name text);
		create table messages (
			channel_id text not null, ts text not null, workspace_id text not null, user_id text,
			text text not null, normalized_text text not null, thread_ts text, deleted_ts text,
			primary key (channel_id, ts)
		);
		create virtual table message_fts using fts5(message_key unindexed, content);
		insert into workspaces values ('T1', 'acme');
		insert into channels values ('C1', 'T1', 'eng'), ('C2', 'T1', 'random');
		insert into users values ('U1', 'ada', 'Ada Lovelace');
	`);
	const insertMessage = db.prepare(
		"insert into messages (channel_id, ts, workspace_id, user_id, text, normalized_text) values (?, ?, 'T1', ?, ?, ?)",
	);
	const insertFts = db.prepare("insert into message_fts (message_key, content) values (?, ?)");
	for (const row of rows) {
		insertMessage.run(row.channel, row.ts, row.user ?? "U1", row.text, row.text);
		insertFts.run(`${row.channel}|${row.ts}`, row.text);
	}
	db.close();
}

/** Deterministic bag-of-words embedder over a fixed vocabulary: semantic closeness == shared words. */
function vocabEmbedder(
	vocab: readonly string[],
	identity = { provider: "test", model: "bow", dimension: vocab.length },
): Embedder {
	return {
		identity: async () => identity,
		embed: async (texts) =>
			texts.map((text) => {
				const words = new Set(text.toLowerCase().match(/[a-z]+/g) ?? []);
				return vocab.map((word) => (words.has(word) ? 1 : 0));
			}),
	};
}

const VOCAB = ["deploy", "release", "ship", "lunch", "pizza", "bug", "crash"] as const;

function open(embedder?: Embedder): CrawlerSidecar {
	return new CrawlerSidecar({
		descriptor: CRAWLER_DESCRIPTORS.slack,
		crawlerDbPath: crawlerDb,
		sidecarDbPath: sidecarDb,
		workspaceId: "T1",
		...(embedder === undefined ? {} : { embedder }),
	});
}

describe("CrawlerSidecar lexical search", () => {
	it("ranks by BM25 relevance instead of recency", () => {
		seedSlack([
			{ channel: "C1", ts: "100.000001", text: "bug bug bug in the checkout crash path" },
			{ channel: "C1", ts: "900.000001", text: "lunch is pizza today and also a bug" },
			{ channel: "C2", ts: "950.000001", text: "unrelated chatter" },
		]);
		const sidecar = open();
		const hits = sidecar.searchLexical("bug crash", 5);
		expect(hits.map((hit) => hit.key)).toEqual(["C1|100.000001", "C1|900.000001"]);
		sidecar.close();
	});

	it("matches natural-language queries by OR-ing terms and ignoring FTS operators and punctuation", () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "we will deploy the release tonight" }]);
		const sidecar = open();
		// Default slacrawl search ANDs every term and returns nothing for this query.
		expect(sidecar.searchLexical("when do we ship the release? AND NOT (deploy)", 5)).toHaveLength(1);
		expect(sidecar.searchLexical('"', 5)).toEqual([]);
		expect(sidecar.searchLexical("   ", 5)).toEqual([]);
		sidecar.close();
	});

	it("returns the message text, channel and author needed to build a result", () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes are ready" }]);
		const sidecar = open();
		const [hit] = sidecar.searchLexical("release", 5);
		expect(hit).toMatchObject({ key: "C1|1.000001", content: "release notes are ready" });
		expect(hit?.metadata).toMatchObject({ channelId: "C1", channelName: "eng", senderName: "Ada Lovelace" });
		expect(hit?.publishedAt).toBe(1_000);
		sidecar.close();
	});

	it("never writes to the crawler database", () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "hello world" }]);
		const before = statSync(crawlerDb).mtimeMs;
		const sidecar = open();
		sidecar.searchLexical("hello", 5);
		sidecar.close();
		expect(statSync(crawlerDb).mtimeMs).toBe(before);
		const check = new DatabaseSync(crawlerDb, { readOnly: true });
		const tables = check
			.prepare("select name from sqlite_master where type = 'table' and name not like 'message_fts%'")
			.all();
		check.close();
		expect(tables.map((row) => String(row.name)).sort()).toEqual(["channels", "messages", "users", "workspaces"]);
	});

	it("excludes soft-deleted messages and respects the workspace scope", () => {
		seedSlack([
			{ channel: "C1", ts: "1.000001", text: "alive release" },
			{ channel: "C1", ts: "2.000001", text: "deleted release" },
		]);
		const db = new DatabaseSync(crawlerDb);
		db.exec("update messages set deleted_ts = '3.0' where ts = '2.000001'");
		db.close();
		const sidecar = open();
		expect(sidecar.searchLexical("release", 5).map((hit) => hit.key)).toEqual(["C1|1.000001"]);
		sidecar.close();
	});
});

describe("CrawlerSidecar is lazy", () => {
	it("creates no file or directory until it is first used", () => {
		const nested = join(root, "deep", "dir", "sidecar.sqlite");
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.slack,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: nested,
		});
		expect(existsSync(join(root, "deep"))).toBe(false);
		sidecar.close();
		expect(existsSync(join(root, "deep"))).toBe(false);
	});

	it("does not create its database for a search that cannot succeed (no crawler archive yet)", async () => {
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(sidecar.searchLexical("release", 5)).toEqual([]);
		expect(await sidecar.searchSemantic("release", 5)).toEqual([]);
		expect(existsSync(sidecarDb)).toBe(false);
		sidecar.close();
	});

	it("opens, then releases, its handles: close() is idempotent and a closed sidecar reopens on next use", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes" }]);
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(sidecar.searchLexical("release", 5)).toHaveLength(1);
		sidecar.close();
		sidecar.close();
		expect(sidecar.searchLexical("release", 5)).toHaveLength(1);
		sidecar.close();
	});
});

describe("CrawlerSidecar before the crawler has an archive", () => {
	it("constructs without a crawler database and answers empty instead of throwing", async () => {
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(sidecar.searchLexical("release", 5)).toEqual([]);
		expect(await sidecar.searchSemantic("release", 5)).toEqual([]);
		expect(await sidecar.searchHybrid("release", 5)).toEqual([]);
		expect(await sidecar.sync()).toMatchObject({ ok: false, message: expect.stringContaining("not available") });
		sidecar.close();
	});

	it("starts working as soon as the crawler creates its database, without being reopened", async () => {
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(sidecar.searchLexical("release", 5)).toEqual([]);
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes" }]);
		expect(sidecar.searchLexical("release", 5)).toHaveLength(1);
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 1 });
		sidecar.close();
	});

	it("reports a crawler database that exists but lacks the expected tables as unavailable, not as a crash", async () => {
		const db = new DatabaseSync(crawlerDb);
		db.exec("create table unrelated (x)");
		db.close();
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(sidecar.searchLexical("release", 5)).toEqual([]);
		expect(await sidecar.sync()).toMatchObject({ ok: false });
		sidecar.close();
	});
});

describe("CrawlerSidecar semantic index", () => {
	it("embeds only messages it has not embedded yet and finds paraphrases by meaning", async () => {
		seedSlack([
			{ channel: "C1", ts: "1.000001", text: "we ship the release friday" },
			{ channel: "C1", ts: "2.000001", text: "pizza for lunch" },
		]);
		const sidecar = open(vocabEmbedder(VOCAB));
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 2 });
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 0 });
		// Lexically disjoint from the message ("deploy" never appears) but sharing "release" semantics via the bag.
		const hits = await sidecar.searchSemantic("release deploy", 5);
		expect(hits[0]?.key).toBe("C1|1.000001");
		sidecar.close();
	});

	it("embeds messages as documents and search queries as queries", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "we ship the release friday" }]);
		const base = vocabEmbedder(VOCAB);
		const roles: { texts: readonly string[]; role: string | undefined }[] = [];
		const recording: Embedder = {
			identity: base.identity,
			embed: async (texts, options) => {
				roles.push({ texts: [...texts], role: options?.role });
				return base.embed(texts, options);
			},
		};
		const sidecar = open(recording);
		await sidecar.sync();
		await sidecar.searchSemantic("release deploy", 5);
		await sidecar.searchHybrid("ship it", 5);
		expect(roles).toEqual([
			{ texts: ["eng\nwe ship the release friday"], role: "document" },
			{ texts: ["release deploy"], role: "query" },
			{ texts: ["ship it"], role: "query" },
		]);
		sidecar.close();
	});

	it("picks up new messages incrementally without re-embedding old ones", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release" }]);
		const calls: number[] = [];
		const base = vocabEmbedder(VOCAB);
		const counting: Embedder = {
			identity: base.identity,
			embed: async (texts) => {
				calls.push(texts.length);
				return base.embed(texts);
			},
		};
		const sidecar = open(counting);
		await sidecar.sync();
		const db = new DatabaseSync(crawlerDb);
		db.exec(
			"insert into messages (channel_id, ts, workspace_id, user_id, text, normalized_text) values ('C1','5.000001','T1','U1','deploy','deploy'); insert into message_fts values ('C1|5.000001','deploy')",
		);
		db.close();
		await sidecar.sync();
		expect(calls).toEqual([1, 1]);
		sidecar.close();
	});

	it("re-embeds everything when the embedding identity changes", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release" }]);
		const first = open(vocabEmbedder(VOCAB));
		await first.sync();
		first.close();
		const second = open(vocabEmbedder(VOCAB, { provider: "test", model: "bow-v2", dimension: VOCAB.length }));
		expect(await second.sync()).toMatchObject({ ok: true, embedded: 1 });
		second.close();
	});

	it("removes vectors whose message the crawler purged (orphan cleanup)", async () => {
		seedSlack([
			{ channel: "C1", ts: "1.000001", text: "release one" },
			{ channel: "C1", ts: "2.000001", text: "release two" },
		]);
		const sidecar = open(vocabEmbedder(VOCAB));
		await sidecar.sync();
		const db = new DatabaseSync(crawlerDb);
		db.exec("delete from message_fts where message_key = 'C1|1.000001'; delete from messages where ts = '1.000001'");
		db.close();
		const result = await sidecar.sync();
		expect(result).toMatchObject({ ok: true, pruned: 1 });
		expect((await sidecar.searchSemantic("release", 5)).map((hit) => hit.key)).toEqual(["C1|2.000001"]);
		sidecar.close();
	});

	it("keeps the sidecar usable and reports the failure when the embedder throws", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release" }]);
		const failing: Embedder = {
			identity: async () => ({ provider: "t", model: "m", dimension: 3 }),
			embed: async () => {
				throw new Error("gateway down");
			},
		};
		const sidecar = open(failing);
		expect(await sidecar.sync()).toEqual({ ok: false, message: "gateway down" });
		expect(sidecar.searchLexical("release", 5)).toHaveLength(1);
		expect(await sidecar.searchSemantic("release", 5)).toEqual([]);
		sidecar.close();
	});

	it("reports a mismatched vector count instead of storing garbage", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release" }]);
		const broken: Embedder = {
			identity: async () => ({ provider: "t", model: "m", dimension: 3 }),
			embed: async () => [],
		};
		const sidecar = open(broken);
		expect(await sidecar.sync()).toMatchObject({ ok: false });
		sidecar.close();
	});
});

describe("CrawlerSidecar sync budget", () => {
	it("stops embedding when the budget is spent, keeps what it embedded, and finishes the rest on the next sync", async () => {
		seedSlack(Array.from({ length: 6 }, (_, i) => ({ channel: "C1", ts: `${i + 1}.000001`, text: `release ${i}` })));
		const base = vocabEmbedder(VOCAB);
		const batches: number[] = [];
		const slow: Embedder = {
			identity: base.identity,
			embed: async (texts) => {
				batches.push(texts.length);
				return base.embed(texts);
			},
		};
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.slack,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			workspaceId: "T1",
			embedder: slow,
			batchSize: 2,
			maxEmbedPerSync: 4,
		});
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 4, remaining: 2 });
		expect(batches).toEqual([2, 2]);
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 2, remaining: 0 });
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 0, remaining: 0 });
		sidecar.close();
	});

	it("keeps partially embedded messages searchable while the backlog drains (BM25 never waits for vectors)", async () => {
		seedSlack(Array.from({ length: 4 }, (_, i) => ({ channel: "C1", ts: `${i + 1}.000001`, text: `release ${i}` })));
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.slack,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: vocabEmbedder(VOCAB),
			batchSize: 2,
			maxEmbedPerSync: 2,
		});
		await sidecar.sync();
		expect(sidecar.searchLexical("release", 10)).toHaveLength(4);
		expect(await sidecar.searchSemantic("release", 10)).toHaveLength(2);
		expect(await sidecar.searchHybrid("release", 10)).toHaveLength(4);
		sidecar.close();
	});
});

describe("CrawlerSidecar hybrid search", () => {
	it("fuses lexical and semantic rankings so a message found by both outranks one found by either alone", async () => {
		seedSlack([
			{ channel: "C1", ts: "1.000001", text: "release ship deploy plan" },
			{ channel: "C1", ts: "2.000001", text: "release" },
			{ channel: "C1", ts: "3.000001", text: "ship deploy" },
		]);
		const sidecar = open(vocabEmbedder(VOCAB));
		await sidecar.sync();
		const hits = await sidecar.searchHybrid("release ship deploy", 5);
		expect(hits[0]?.key).toBe("C1|1.000001");
		expect(new Set(hits.map((hit) => hit.key)).size).toBe(hits.length);
		sidecar.close();
	});

	it("degrades to lexical results when no embedder is configured", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes" }]);
		const sidecar = open();
		expect((await sidecar.searchHybrid("release", 5)).map((hit) => hit.key)).toEqual(["C1|1.000001"]);
		sidecar.close();
	});

	it("degrades to lexical results when the query embedding fails", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes" }]);
		const base = vocabEmbedder(VOCAB);
		let first = true;
		const flaky: Embedder = {
			identity: base.identity,
			embed: async (texts) => {
				if (first) {
					first = false;
					return base.embed(texts);
				}
				throw new Error("gateway down");
			},
		};
		const sidecar = open(flaky);
		await sidecar.sync();
		expect((await sidecar.searchHybrid("release", 5)).map((hit) => hit.key)).toEqual(["C1|1.000001"]);
		sidecar.close();
	});
});

/** telecrawl/wacrawl shape: one messages table, external-content messages_fts keyed by rowid, integer-second ts. */
function seedChat(rows: readonly { event: string; chat: string; text: string; ts: number; deleted?: boolean }[]): void {
	const db = new DatabaseSync(crawlerDb);
	db.exec(`
		create table messages (
			rowid integer primary key autoincrement, event_id text not null unique, chat_jid text not null,
			chat_name text, sender_name text, ts integer not null, text text, deleted_at integer
		);
		create virtual table messages_fts using fts5(text, chat, sender, media);
	`);
	const insert = db.prepare(
		"insert into messages (event_id, chat_jid, chat_name, sender_name, ts, text, deleted_at) values (?, ?, ?, 'Bo', ?, ?, ?)",
	);
	const fts = db.prepare(
		"insert into messages_fts (rowid, text, chat, sender, media) values (last_insert_rowid(), ?, ?, 'Bo', '')",
	);
	for (const row of rows) {
		insert.run(row.event, `jid-${row.chat}`, row.chat, row.ts, row.text, row.deleted ? 1 : null);
		fts.run(row.text, row.chat);
	}
	db.close();
}

describe("CrawlerSidecar over the telecrawl/wacrawl schema", () => {
	for (const id of ["telegram", "whatsapp"] as const) {
		it(`${id}: BM25 search maps chat metadata, epoch seconds, and hides soft-deleted rows`, async () => {
			seedChat([
				{ event: "e1", chat: "family", text: "dinner at seven, bring wine", ts: 1_700_000_000 },
				{ event: "e2", chat: "family", text: "dinner cancelled", ts: 1_700_000_100, deleted: true },
				{ event: "e3", chat: "work", text: "standup moved", ts: 1_700_000_200 },
			]);
			const sidecar = new CrawlerSidecar({
				descriptor: CRAWLER_DESCRIPTORS[id],
				crawlerDbPath: crawlerDb,
				sidecarDbPath: sidecarDb,
			});
			const hits = sidecar.searchLexical("dinner wine", 5);
			expect(hits.map((hit) => hit.key)).toEqual(["e1"]);
			expect(hits[0]).toMatchObject({ title: "family", publishedAt: 1_700_000_000_000 });
			expect(hits[0]?.metadata).toMatchObject({ chatId: "jid-family", chatName: "family", senderName: "Bo" });
			sidecar.close();
		});
	}

	it("prunes the vector of a message the crawler later soft-deletes", async () => {
		seedChat([
			{ event: "e1", chat: "family", text: "release", ts: 1 },
			{ event: "e2", chat: "family", text: "release again", ts: 2 },
		]);
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.telegram,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: vocabEmbedder(VOCAB),
		});
		await sidecar.sync();
		const db = new DatabaseSync(crawlerDb);
		db.exec("update messages set deleted_at = 5 where event_id = 'e1'");
		db.close();
		expect(await sidecar.sync()).toMatchObject({ ok: true, pruned: 1 });
		expect((await sidecar.searchSemantic("release", 5)).map((hit) => hit.key)).toEqual(["e2"]);
		sidecar.close();
	});
});

describe("CrawlerSidecar robustness", () => {
	it("skips a message the embedder keeps rejecting so the backlog behind it still drains", async () => {
		seedChat(
			Array.from({ length: 6 }, (_, i) => ({
				event: `e${i}`,
				chat: "family",
				text: i === 1 ? "poison" : `release ${i}`,
				ts: i,
			})),
		);
		const base = vocabEmbedder(VOCAB);
		const picky: Embedder = {
			identity: base.identity,
			embed: async (texts) => {
				if (texts.some((text) => text.includes("poison")))
					throw new Error("gateway /v1/embeddings returned HTTP 413");
				return base.embed(texts);
			},
		};
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.whatsapp,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: picky,
			batchSize: 2,
		});
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 5, skipped: 1, remaining: 0 });
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 0, skipped: 0, remaining: 0 });
		expect((await sidecar.searchSemantic("release", 10)).map((hit) => hit.key).sort()).toEqual([
			"e0",
			"e2",
			"e3",
			"e4",
			"e5",
		]);
		sidecar.close();
	});

	it("keeps everything pending when the embedder rejects every message (the gateway is down, not one message)", async () => {
		seedChat([
			{ event: "e1", chat: "family", text: "release", ts: 1 },
			{ event: "e2", chat: "family", text: "ship", ts: 2 },
		]);
		const down: Embedder = {
			identity: async () => ({ provider: "t", model: "m", dimension: VOCAB.length }),
			embed: async () => {
				throw new Error("gateway down");
			},
		};
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.whatsapp,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: down,
		});
		expect(await sidecar.sync()).toEqual({ ok: false, message: "gateway down" });
		sidecar.close();
		const recovered = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.whatsapp,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: vocabEmbedder(VOCAB),
		});
		expect(await recovered.sync()).toMatchObject({ ok: true, embedded: 2, skipped: 0 });
		recovered.close();
	});

	// Windows refuses to rename over or delete a file another handle has open, so a crawler cannot swap its
	// archive under an attached sidecar there; the scenario only exists on POSIX.
	it.skipIf(process.platform === "win32")("follows the crawler when it replaces or deletes its archive file", () => {
		seedChat([{ event: "old", chat: "family", text: "deploy old", ts: 1 }]);
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.whatsapp,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
		});
		expect(sidecar.searchLexical("deploy", 5).map((hit) => hit.key)).toEqual(["old"]);
		const live = crawlerDb;
		crawlerDb = join(root, "rebuilt.db");
		seedChat([{ event: "new", chat: "family", text: "deploy new", ts: 2 }]);
		renameSync(crawlerDb, live);
		crawlerDb = live;
		expect(sidecar.searchLexical("deploy", 5).map((hit) => hit.key)).toEqual(["new"]);
		rmSync(live);
		expect(sidecar.searchLexical("deploy", 5)).toEqual([]);
		sidecar.close();
	});

	it("embeds a search query once for the semantic and hybrid methods of the same search", async () => {
		seedSlack([{ channel: "C1", ts: "1.000001", text: "release notes" }]);
		const base = vocabEmbedder(VOCAB);
		const queries: string[] = [];
		const counting: Embedder = {
			identity: base.identity,
			embed: async (texts) => {
				if (texts.length === 1 && texts[0] === "release") queries.push(texts[0]);
				return base.embed(texts);
			},
		};
		const sidecar = open(counting);
		await sidecar.sync();
		await Promise.all([sidecar.searchSemantic("release", 5), sidecar.searchHybrid("release", 5)]);
		expect(queries).toEqual(["release"]);
		sidecar.close();
	});
});

describe("CrawlerSidecar kakao descriptor", () => {
	function seedKakao(
		rows: readonly {
			chunkId: string;
			chatId: string;
			chatName: string;
			sender: string;
			startedAt: string;
			text: string;
		}[],
	): void {
		const db = new DatabaseSync(crawlerDb);
		db.exec(`
			create table chunks (
				chunk_id text primary key,
				account_hash text not null,
				chat_id text not null,
				chat_name text not null,
				sender_nickname text not null,
				started_at text not null,
				ended_at text not null,
				text text not null,
				message_count integer not null
			);
			create virtual table chunks_fts using fts5(chunk_id unindexed, text);
		`);
		const insertChunk = db.prepare(
			"insert into chunks (chunk_id, account_hash, chat_id, chat_name, sender_nickname, started_at, ended_at, text, message_count) values (?, 'acct', ?, ?, ?, ?, ?, ?, 1)",
		);
		const insertFts = db.prepare("insert into chunks_fts (rowid, chunk_id, text) values (?, ?, ?)");
		for (const row of rows) {
			const res = insertChunk.run(
				row.chunkId,
				row.chatId,
				row.chatName,
				row.sender,
				row.startedAt,
				row.startedAt,
				row.text,
			);
			insertFts.run(res.lastInsertRowid, row.chunkId, row.text);
		}
		db.close();
	}

	it("indexes and searches lazykatok chunks with BM25, semantic and hybrid while preserving ISO timestamps and chat metadata", async () => {
		seedKakao([
			{
				chunkId: "chunk_1",
				chatId: "room_eng",
				chatName: "오픈소스 개발과제",
				sender: "정철현",
				startedAt: "2026-04-13T05:57:50+00:00",
				text: "we will ship the release tonight",
			},
			{
				chunkId: "chunk_2",
				chatId: "room_lunch",
				chatName: "점심방",
				sender: "김민수",
				startedAt: "2026-04-14T03:00:00+00:00",
				text: "pizza for lunch today",
			},
			{
				chunkId: "chunk_feed",
				chatId: "room_eng",
				chatName: "오픈소스 개발과제",
				sender: "정철현",
				startedAt: "2026-04-13T06:00:00+00:00",
				text: '{"logId":3809217410566397952,"byHost":false,"hidden":true,"feedType":14}',
			},
			{
				chunkId: "chunk_voip",
				chatId: "room_eng",
				chatName: "오픈소스 개발과제",
				sender: "정철현",
				startedAt: "2026-04-13T06:01:00+00:00",
				text: '{"type":"bye","csIP":"121.53.219.109","csPort":12029}',
			},
		]);
		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.kakao,
			crawlerDbPath: crawlerDb,
			sidecarDbPath: sidecarDb,
			embedder: vocabEmbedder(VOCAB),
		});
		expect(await sidecar.sync()).toMatchObject({ ok: true, embedded: 2, remaining: 0 });
		const lexical = sidecar.searchLexical("release", 5);
		expect(lexical).toHaveLength(1);
		expect(lexical[0]).toMatchObject({
			key: "chunk_1",
			content: "we will ship the release tonight",
			title: "오픈소스 개발과제",
			publishedAt: Date.parse("2026-04-13T05:57:50+00:00"),
			metadata: {
				chatId: "room_eng",
				chatName: "오픈소스 개발과제",
				senderNickname: "정철현",
				startedAt: "2026-04-13T05:57:50+00:00",
				endedAt: "2026-04-13T05:57:50+00:00",
			},
		});
		const hybrid = await sidecar.searchHybrid("deploy ship", 5);
		expect(hybrid[0]?.key).toBe("chunk_1");
		sidecar.close();
	});
});
