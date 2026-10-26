import { createHash } from "node:crypto";
import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { discordDescriptor } from "../../../../src/datasource/sidecar/descriptors.ts";
import { CrawlerSidecar, type CrawlerSidecarOptions } from "../../../../src/datasource/sidecar/index.ts";
import { SidecarRetrievalMethod } from "../../../../src/datasource/sidecar/methods.ts";
import { DiscrawlFtsMethod } from "../../../../src/datasource/skills/discrawl/methods.ts";
import { DiscrawlSidecar, type DiscrawlSidecarOptions } from "../../../../src/datasource/skills/discrawl/sidecar.ts";
import { DiscrawlSkill, type DiscrawlSkillClient } from "../../../../src/datasource/skills/discrawl/skill.ts";
import type { DiscrawlStatusResult } from "../../../../src/datasource/skills/discrawl/types.ts";
import { buildDatasourceSkills } from "../../../../src/datasource/skills/factory.ts";

let root: string;
let crawlerDb: string;
let sidecarDb: string;
const opened: { close(): void }[] = [];
const OLD_MESSAGE = "1513467741523415161";

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-discord-sidecar-"));
	crawlerDb = join(root, "discrawl.db");
	sidecarDb = join(root, "sidecar", "index.sqlite");
});

afterEach(() => {
	for (const resource of opened.splice(0)) resource.close();
	rmSync(root, { recursive: true, force: true });
});

interface Message {
	id: string;
	content: string;
	channel?: string;
	guild?: string;
	normalized?: string;
	deletedAt?: string;
	raw?: string;
}

/** discrawl's native TEXT snowflakes and FTS5 columns, including its own vector table. */
function seed(rows: readonly Message[], path = crawlerDb): void {
	const db = new DatabaseSync(path);
	try {
		db.exec(`
			pragma journal_mode = wal;
			create table guilds (id text primary key, name text not null);
			create table channels (id text primary key, guild_id text not null, name text not null);
			create table messages (
				id text primary key, guild_id text not null, channel_id text not null, author_id text,
				message_type integer not null, created_at text not null, edited_at text, deleted_at text,
				content text not null, normalized_content text not null, reply_to_message_id text,
				pinned integer not null default 0, has_attachments integer not null default 0,
				raw_json text not null, updated_at text not null
			);
			create virtual table message_fts using fts5(
				message_id unindexed, guild_id unindexed, channel_id unindexed, author_id unindexed,
				author_name, channel_name, content
			);
			create table message_embeddings (
				message_id text not null, provider text not null, model text not null, input_version text not null,
				dimensions integer not null, embedding_blob blob not null, embedded_at text not null,
				primary key (message_id, provider, model, input_version)
			);
			insert into guilds values ('G1', 'Engineering'), ('G2', 'Community');
			insert into channels values ('C1', 'G1', 'releases'), ('C2', 'G1', 'general'), ('C3', 'G2', 'releases');
		`);
		const message = db.prepare(`insert into messages
			(id, guild_id, channel_id, author_id, message_type, created_at, deleted_at,
			 content, normalized_content, raw_json, updated_at)
			values (?, ?, ?, 'U1', 0, ?, ?, ?, ?, ?, ?)`);
		const fts = db.prepare("insert into message_fts values (?, ?, ?, 'U1', 'Ada', ?, ?)");
		for (const [index, row] of rows.entries()) {
			const guild = row.guild ?? "G1";
			const channel = row.channel ?? "C1";
			const timestamp = new Date(Date.UTC(2024, 0, index + 1)).toISOString();
			message.run(
				row.id,
				guild,
				channel,
				timestamp,
				row.deletedAt ?? null,
				row.content,
				row.normalized ?? row.content,
				row.raw ?? JSON.stringify({ member: { nick: "Ada" }, author: { username: "ada" } }),
				timestamp,
			);
			fts.run(row.id, guild, channel, channel === "C2" ? "general" : "releases", row.normalized ?? row.content);
		}
		db.prepare("insert into message_embeddings values (?, 'ollama', 'native-model', 'v1', 2, ?, ?)").run(
			rows[0]?.id ?? OLD_MESSAGE,
			new Uint8Array([0, 0, 128, 63, 0, 0, 0, 0]),
			"2024-01-01T00:00:00Z",
		);
	} finally {
		db.close();
	}
}

function options(overrides: Partial<CrawlerSidecarOptions> = {}): CrawlerSidecarOptions {
	return { descriptor: discordDescriptor, crawlerDbPath: crawlerDb, sidecarDbPath: sidecarDb, ...overrides };
}

function open(overrides: Partial<CrawlerSidecarOptions> = {}): CrawlerSidecar {
	const sidecar = new CrawlerSidecar(options(overrides));
	opened.push(sidecar);
	return sidecar;
}

function status(): DiscrawlStatusResult {
	return {
		ok: true,
		code: 0,
		stdout: "",
		stderr: "",
		data: { databasePath: crawlerDb, messages: 1, guilds: 1, channels: 1 },
	};
}

function client() {
	return {
		status: vi.fn<NonNullable<DiscrawlSkillClient["status"]>>().mockResolvedValue(status()),
		search: vi.fn<DiscrawlSkillClient["search"]>().mockResolvedValue({
			ok: true,
			code: 0,
			stdout: "",
			stderr: "",
			hits: [],
			data: { hits: [] },
		}),
		doctor: vi.fn<DiscrawlSkillClient["doctor"]>(),
		sync: vi.fn<DiscrawlSkillClient["sync"]>(),
		embed: vi.fn<DiscrawlSkillClient["embed"]>(),
	};
}

function lexicalMethod(stub = client(), sidecarOptions: DiscrawlSidecarOptions = options()) {
	const sidecar = new DiscrawlSidecar(sidecarOptions, stub);
	opened.push(sidecar);
	const method = new SidecarRetrievalMethod({
		sidecar: (signal) => sidecar.get(signal),
		descriptor: discordDescriptor,
		mode: "bm25",
		datasourceId: "discord",
		skillType: "discord-archive",
		instanceId: "work",
		tags: ["discord"],
	});
	return { method, stub };
}

describe("Discord BM25 over the native archive", () => {
	it("retrieves an old relevant message outside the native FTS recency window", () => {
		seed([
			{ id: OLD_MESSAGE, content: "bug bug bug bug crash" },
			...Array.from({ length: 25 }, (_, index) => ({
				id: String(1513467741523415162n + BigInt(index)),
				content: `bug ${"routine update and unrelated conversation ".repeat(20)}`,
			})),
		]);
		const native = new DatabaseSync(crawlerDb, { readOnly: true });
		const recent = native
			.prepare(`select m.id from messages m
			join message_fts on message_fts.message_id = m.id
			where message_fts match 'bug' and m.deleted_at is null and trim(coalesce(m.normalized_content, '')) <> ''
			order by message_fts.rowid desc limit 20`)
			.all();
		native.close();
		expect(recent).toHaveLength(20);
		expect(recent.map((row) => row.id)).not.toContain(OLD_MESSAGE);

		const hits = open().searchLexical("bug", 3);
		expect(hits).toHaveLength(3);
		expect(hits[0]?.key).toBe(OLD_MESSAGE);
		expect(hits[0]?.score).toBeGreaterThan(hits[1]?.score ?? 0);
	});

	it("leaves the native database bytes and message embeddings unchanged", () => {
		seed([{ id: OLD_MESSAGE, content: "release checklist" }]);
		const before = createHash("sha256").update(readFileSync(crawlerDb)).digest("hex");
		const sidecar = open();
		expect(sidecar.searchLexical("release", 5)).toHaveLength(1);
		sidecar.close();
		expect(createHash("sha256").update(readFileSync(crawlerDb)).digest("hex")).toBe(before);
		const native = new DatabaseSync(crawlerDb, { readOnly: true });
		const vectors = native
			.prepare("select message_id, hex(embedding_blob) as vector, model from message_embeddings")
			.all();
		native.close();
		expect(vectors).toEqual([{ message_id: OLD_MESSAGE, vector: "0000803F00000000", model: "native-model" }]);
	});

	it.each([{ channelIds: ["C1"] }, { channelNames: ["releases"] }])(
		"applies guild and channel restrictions before the candidate limit: %j",
		async (allowlist) => {
			seed([
				{ id: "outside-guild", content: "bug bug bug", guild: "G2", channel: "C3" },
				{ id: "outside-channel", content: "bug bug bug", channel: "C2" },
				{ id: OLD_MESSAGE, content: "release notes mention a bug and other unrelated details" },
			]);
			const { method } = lexicalMethod(client(), options({ workspaceId: "G1", ...allowlist }));
			const hits = await method.retrieve("bug", { topK: 1 });
			expect(hits.map((hit) => hit.metadata.messageId)).toEqual([OLD_MESSAGE]);
		},
	);

	it("excludes deleted and empty normalized messages while preserving content and provenance", () => {
		seed([
			{ id: OLD_MESSAGE, content: "Original release\nnotes", normalized: "original release notes" },
			{ id: "deleted", content: "release", deletedAt: "2024-02-01T00:00:00Z" },
			{ id: "empty", content: "release" },
		]);
		const native = new DatabaseSync(crawlerDb);
		native.exec("update messages set normalized_content = ' ' where id = 'empty'");
		native.close();
		const hits = open().searchLexical("release", 5);
		expect(hits).toHaveLength(1);
		expect(hits[0]).toMatchObject({
			key: OLD_MESSAGE,
			content: "Original release\nnotes",
			publishedAt: Date.parse("2024-01-01T00:00:00Z"),
			metadata: {
				messageId: OLD_MESSAGE,
				guildId: "G1",
				guildName: "Engineering",
				channelId: "C1",
				channelName: "releases",
				authorName: "Ada",
			},
		});
	});

	it("uses the same source and evidence id as native Discord retrieval", async () => {
		seed([{ id: OLD_MESSAGE, content: "release notes" }]);
		const { method, stub } = lexicalMethod();
		const hits = [{ messageId: OLD_MESSAGE, content: "release notes", score: 1 }];
		stub.search.mockResolvedValue({ ok: true, code: 0, stdout: "", stderr: "", hits, data: { hits } });
		const [native] = await new DiscrawlFtsMethod({ client: stub, instanceId: "work" }).retrieve("release", {});
		const [ranked] = await method.retrieve("release", {});
		expect(ranked).toMatchObject({
			id: native?.id,
			source: native?.source,
			metadata: {
				method: "discord-sidecar-bm25",
				messageId: OLD_MESSAGE,
				datasourceId: "discord",
				instanceId: "work",
			},
		});
		expect(ranked?.source).toBe(`/discord/work/chunks/${OLD_MESSAGE}`);
	});

	it("ranks a language-index-only message that the base tokenizer cannot match", () => {
		seed([{ id: OLD_MESSAGE, content: "내일배포완료", normalized: "내일배포완료" }]);
		const native = new DatabaseSync(crawlerDb);
		native.exec(`
			create virtual table message_fts_ko using fts5(
				message_id unindexed, guild_id unindexed, channel_id unindexed, author_id unindexed,
				author_name, channel_name, content, tokenize = 'unicode61 remove_diacritics 0'
			);
		`);
		native
			.prepare(
				`insert into message_fts_ko
				(rowid, message_id, guild_id, channel_id, author_id, author_name, channel_name, content)
				values (1, ?, 'G1', 'C1', 'U1', 'Ada', 'releases', '내일 배포 완료')`,
			)
			.run(OLD_MESSAGE);
		// unicode61 keeps the unspaced base text as one token, so the base index misses the Korean term.
		expect(
			native.prepare("select count(*) as count from message_fts where message_fts match '배포'").get()?.count,
		).toBe(0);
		native.close();

		const hits = open().searchLexical("배포", 5);
		expect(hits.map((hit) => hit.key)).toEqual([OLD_MESSAGE]);
		expect(hits[0]?.metadata.channelName).toBe("releases");
	});

	it("fuses base and language rankings once when evidence appears in both", () => {
		seed([
			{ id: OLD_MESSAGE, content: "bug bug bug bug crash" },
			{ id: "korean-only", content: "내일배포완료", normalized: "내일배포완료" },
		]);
		const native = new DatabaseSync(crawlerDb);
		native.exec(
			"create virtual table message_fts_ko using fts5(message_id unindexed, guild_id unindexed, channel_id unindexed, author_id unindexed, author_name, channel_name, content)",
		);
		const insert = native.prepare("insert into message_fts_ko values (?, 'G1', 'C1', 'U1', 'Ada', 'releases', ?)");
		insert.run("korean-only", "내일 배포 완료");
		// The old strong hit also appears in the language index; it must not gain a second vote.
		insert.run(OLD_MESSAGE, "bug bug bug bug crash");
		native.close();

		const hits = open().searchLexical("bug 배포", 5);
		expect(hits.map((hit) => hit.key)).toEqual([OLD_MESSAGE, "korean-only"]);
	});
});

describe("Discord sidecar discovery", () => {
	it("discovers the native database lazily, reuses it, and never invokes native search", async () => {
		seed([{ id: OLD_MESSAGE, content: "release checklist" }]);
		const { method, stub } = lexicalMethod(client(), { descriptor: discordDescriptor, sidecarDbPath: sidecarDb });
		expect(stub.status).not.toHaveBeenCalled();
		expect(existsSync(sidecarDb)).toBe(false);
		await expect(method.retrieve("release", {})).resolves.toHaveLength(1);
		await expect(method.retrieve("release", {})).resolves.toHaveLength(1);
		expect(stub.status).toHaveBeenCalledTimes(1);
		expect(stub.search).not.toHaveBeenCalled();
		expect(stub.embed).not.toHaveBeenCalled();
	});

	it("bypasses status for an explicitly configured database", async () => {
		seed([{ id: OLD_MESSAGE, content: "release checklist" }]);
		const { method, stub } = lexicalMethod();
		await expect(method.retrieve("release", {})).resolves.toHaveLength(1);
		expect(stub.status).not.toHaveBeenCalled();
	});

	it("surfaces the original status failure and retries discovery on the next query", async () => {
		seed([{ id: OLD_MESSAGE, content: "release checklist" }]);
		const stub = client();
		stub.status.mockResolvedValueOnce({
			ok: false,
			reason: "nonzero-exit",
			code: 7,
			stdout: "",
			stderr: "cannot open /archive/team.db",
		});
		const { method } = lexicalMethod(stub, { descriptor: discordDescriptor, sidecarDbPath: sidecarDb });
		await expect(method.retrieve("release", {})).rejects.toThrow("cannot open /archive/team.db");
		expect(existsSync(sidecarDb)).toBe(false);
		await expect(method.retrieve("release", {})).resolves.toHaveLength(1);
		expect(stub.status).toHaveBeenCalledTimes(2);
	});

	it("skips discovery for empty queries, empty limits, and pre-aborted requests", async () => {
		const { method, stub } = lexicalMethod(client(), { descriptor: discordDescriptor, sidecarDbPath: sidecarDb });
		await expect(method.retrieve("  ", {})).resolves.toEqual([]);
		await expect(method.retrieve("release", { topK: 0 })).resolves.toEqual([]);
		const signal = AbortSignal.abort(new Error("query cancelled"));
		await expect(method.retrieve("release", { signal })).rejects.toThrow("query cancelled");
		expect(stub.status).not.toHaveBeenCalled();
		expect(existsSync(sidecarDb)).toBe(false);
	});

	it("forwards cancellation to status and does not open the archive after cancellation", async () => {
		seed([{ id: OLD_MESSAGE, content: "release checklist" }]);
		const stub = client();
		const controller = new AbortController();
		stub.status.mockImplementation(async () => {
			controller.abort(new Error("query cancelled"));
			return status();
		});
		const { method } = lexicalMethod(stub, { descriptor: discordDescriptor, sidecarDbPath: sidecarDb });
		await expect(method.retrieve("release", { signal: controller.signal })).rejects.toThrow("query cancelled");
		expect(stub.status).toHaveBeenCalledWith(controller.signal);
		expect(existsSync(sidecarDb)).toBe(false);
	});
});

describe("Discord sidecar factory configuration", () => {
	it("registers one composed hybrid method when the BM25 sidecar is enabled", () => {
		const { skills } = buildDatasourceSkills({ discord: {} }, root);
		expect(skills[0]?.retrievalMethods().map((method) => method.describe().name)).toEqual(["discord-hybrid"]);
		expect(existsSync(join(root, ".autorag"))).toBe(false);
	});

	it.each([false, { enabled: false }])("allows the operator to disable the sidecar: %j", (setting) => {
		const { skills } = buildDatasourceSkills({ discord: { connector: { sidecar: setting } } }, root);
		expect(skills[0]?.retrievalMethods().map((method) => method.describe().name)).toEqual(["discord-hybrid"]);
	});

	it("retains CLI-only retrieval when no workspace is configured", () => {
		const { skills } = buildDatasourceSkills({ discord: {} });
		expect(skills[0]?.retrievalMethods().some((method) => method.describe().name === "discord-sidecar-bm25")).toBe(
			false,
		);
	});

	it.each(["databasePath", "crawlerDbPath"])("honors a custom %s without a working discrawl binary", async (key) => {
		seed([
			{ id: "outside-guild", guild: "G2", channel: "C3", content: "release release release" },
			{ id: OLD_MESSAGE, content: "the release checklist and other general updates" },
		]);
		const connector = {
			binaryPath: join(root, "missing-discrawl"),
			guildId: "G1",
			...(key === "databasePath"
				? { databasePath: crawlerDb }
				: { databasePath: join(root, "wrong.db"), sidecar: { crawlerDbPath: crawlerDb } }),
		};
		const { skills } = buildDatasourceSkills({ discord: { connector } }, root);
		const value = skills[0];
		if (!(value instanceof DiscrawlSkill)) throw new Error("Expected a Discord skill");
		opened.push(value);
		const method = value.retrievalMethods().find((entry) => entry.describe().name === "discord-hybrid");
		const hits = await method?.retrieve("release", { topK: 1 });
		expect(hits?.map((hit) => hit.metadata.messageId)).toEqual([OLD_MESSAGE]);
	});
});
