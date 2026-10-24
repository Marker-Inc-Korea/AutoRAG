import { chmodSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import type { CrawlerDatasourceSkill, CrawlerSkillClient } from "../../../src/datasource/crawler-skill.ts";
import type { CrawlerSearchResult, CrawlerSyncResult } from "../../../src/datasource/crawler-types.ts";
import { CRAWLER_DESCRIPTORS } from "../../../src/datasource/sidecar/descriptors.ts";
import { CrawlerSidecar, type SidecarHit } from "../../../src/datasource/sidecar/index.ts";
import { sidecarFirstMethod } from "../../../src/datasource/sidecar/methods.ts";
import { LazykatokSkill, type LazykatokSkillClient } from "../../../src/datasource/skills/lazykatok/index.ts";
import { SlackSkill } from "../../../src/datasource/skills/slack/index.ts";
import { TelecrawlSkill } from "../../../src/datasource/skills/telecrawl/index.ts";
import { WacrawlSkill } from "../../../src/datasource/skills/wacrawl/index.ts";
import type { Embedder } from "../../../src/embedding/embedder.ts";
import type { RetrievalMethod } from "../../../src/retrieval/types.ts";

let root: string;
let crawlerDb: string;
/** Skills built by a test; closed before cleanup so Windows can delete their SQLite files. */
const opened: CrawlerDatasourceSkill[] = [];

function track<T extends CrawlerDatasourceSkill>(skill: T): T {
	opened.push(skill);
	return skill;
}

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-sidecar-skill-"));
	crawlerDb = join(root, "slacrawl.db");
	const db = new DatabaseSync(crawlerDb);
	db.exec(`
		create table channels (id text primary key, workspace_id text, name text);
		create table users (id text primary key, name text, real_name text);
		create table messages (
			channel_id text not null, ts text not null, workspace_id text not null, user_id text,
			text text not null, normalized_text text not null, deleted_ts text, primary key (channel_id, ts)
		);
		create virtual table message_fts using fts5(message_key unindexed, content);
		insert into channels values ('C1', 'T1', 'eng');
		insert into messages values ('C1', '10.000001', 'T1', null, 'we ship the release friday', 'we ship the release friday', null);
		insert into messages values ('C1', '20.000001', 'T1', null, 'pizza for lunch', 'pizza for lunch', null);
		insert into message_fts values ('C1|10.000001', 'we ship the release friday'), ('C1|20.000001', 'pizza for lunch');
	`);
	db.close();
});

afterEach(() => {
	for (const skill of opened.splice(0)) skill.close();
	rmSync(root, { recursive: true, force: true });
});

const VOCAB = ["ship", "release", "pizza", "lunch", "deploy"];
const embedder: Embedder = {
	identity: async () => ({ provider: "t", model: "bow", dimension: VOCAB.length }),
	embed: async (texts) =>
		texts.map((text) => {
			const words = new Set(text.toLowerCase().match(/[a-z]+/g) ?? []);
			return VOCAB.map((word) => (words.has(word) ? 1 : 0));
		}),
};

/** A CLI client that proves the sidecar is what answered: any call to it fails the test. */
const cliMustNotSearch: CrawlerSkillClient = {
	sync: async (): Promise<CrawlerSyncResult> => ({ ok: true, count: 7, stdout: "", stderr: "", code: 0 }),
	search: async (): Promise<CrawlerSearchResult> => {
		throw new Error("CLI search must not be used when the sidecar is enabled");
	},
};

function slackWithSidecar(extra: Record<string, unknown> = {}): SlackSkill {
	return track(
		new SlackSkill({
			client: cliMustNotSearch,
			datasourceId: "slack",
			instanceId: "main",
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.slack,
				crawlerDbPath: crawlerDb,
				sidecarDbPath: join(root, "sidecar.sqlite"),
				workspaceId: "T1",
				embedder,
				...extra,
			},
		}),
	);
}

describe("crawler skills with a sidecar", () => {
	it("registers one method per query: sidecar hybrid, with BM25 and the CLI only as fallbacks", () => {
		const methods = slackWithSidecar()
			.retrievalMethods()
			.map((method) => method.describe());
		expect(methods.map((method) => method.name)).toEqual(["slack-hybrid"]);
		expect(methods.map((method) => method.type)).toEqual(["hybrid"]);
		expect(methods.every((method) => method.datasourceId === "slack")).toBe(true);
	});

	it("indexes through the CLI first, then embeds new messages into the sidecar and reports the count", async () => {
		const skill = slackWithSidecar();
		const result = await skill.index();
		expect(result).toMatchObject({ ok: true, chunkCount: 7 });
		expect(result.diagnostics).toEqual([]);
		const hybrid = skill.retrievalMethods().find((method) => method.describe().name === "slack-hybrid");
		const hits = await hybrid?.retrieve("release deploy", { topK: 3 });
		expect(hits?.[0]?.content).toBe("we ship the release friday");
	});

	it("maps sidecar hits to path-opaque sources and honors the datasource scope", async () => {
		const skill = slackWithSidecar();
		await skill.index();
		const [hybrid] = skill.retrievalMethods();
		const [hit] = (await hybrid?.retrieve("release", { topK: 3 })) ?? [];
		expect(hit?.source).toBe("/slack/main/chunks/C1-10-000001");
		expect(hit?.source).not.toContain(root);
		expect(hit?.metadata).toMatchObject({
			method: "slack-hybrid",
			datasourceId: "slack",
			instanceId: "main",
			channelName: "eng",
		});
		expect(await hybrid?.retrieve("release", { topK: 3, scope: "/slack/other" })).toEqual([]);
		expect(await hybrid?.retrieve("release", { topK: 3, scope: "/slack/main" })).toHaveLength(1);
	});

	it("names a message exactly like the CLI-backed method does, so falling back to the CLI keeps evidence identities stable", async () => {
		const cli: CrawlerSkillClient = {
			sync: cliMustNotSearch.sync,
			search: async () => ({
				ok: true,
				hits: [
					{ id: "C1-10-000001", content: "we ship the release friday", score: 1, metadata: { channelId: "C1" } },
				],
				stdout: "",
				stderr: "",
				code: 0,
			}),
		};
		const cliOnly = new SlackSkill({ client: cli, datasourceId: "slack", instanceId: "main" });
		const withSidecar = track(
			new SlackSkill({
				client: cliMustNotSearch,
				datasourceId: "slack",
				instanceId: "main",
				sidecar: {
					descriptor: CRAWLER_DESCRIPTORS.slack,
					crawlerDbPath: crawlerDb,
					sidecarDbPath: join(root, "same.sqlite"),
					workspaceId: "T1",
				},
			}),
		);
		const [cliMethod] = cliOnly.retrievalMethods();
		const [sidecarMethod] = withSidecar.retrievalMethods();
		const fromCli = (await cliMethod?.retrieve("release", { topK: 3 }))?.[0];
		const fromSidecar = (await sidecarMethod?.retrieve("release", { topK: 3 }))?.[0];
		expect(fromSidecar?.source).toBe(fromCli?.source);
		expect(fromSidecar?.id).toBe(fromCli?.id);
	});

	it("falls back from a throwing sidecar to its BM25, then to the CLI-backed search, and not earlier", async () => {
		class BrokenSidecar extends CrawlerSidecar {
			private readonly breakLexical: boolean;
			constructor(breakLexical: boolean) {
				super({
					descriptor: CRAWLER_DESCRIPTORS.slack,
					crawlerDbPath: crawlerDb,
					sidecarDbPath: join(root, `broken-${breakLexical}.sqlite`),
					workspaceId: "T1",
					embedder,
				});
				this.breakLexical = breakLexical;
			}
			override async searchHybrid(): Promise<never> {
				throw new Error("sqlite: disk I/O error");
			}
			override searchLexical(query: string, topK: number): SidecarHit[] {
				if (this.breakLexical) throw new Error("sqlite: disk I/O error");
				return super.searchLexical(query, topK);
			}
		}
		const cliQueries: string[] = [];
		const cli: RetrievalMethod = {
			describe: () => ({
				name: "slack-fts",
				type: "bm25",
				description: "cli",
				status: "active",
				capabilities: [],
				datasourceId: "slack",
			}),
			retrieve: async (query) => {
				cliQueries.push(query);
				return [
					{ id: "slack:main:cli", content: "from cli", source: "/slack/main/chunks/cli", score: 1, metadata: {} },
				];
			},
		};
		for (const breakLexical of [false, true]) {
			const sidecar = new BrokenSidecar(breakLexical);
			try {
				const method = sidecarFirstMethod(
					{
						sidecar,
						descriptor: CRAWLER_DESCRIPTORS.slack,
						datasourceId: "slack",
						skillType: "slack",
						instanceId: "main",
						tags: [],
					},
					cli,
				);
				const hits = await method.retrieve("release", { topK: 3 });
				expect(hits[0]?.metadata).toMatchObject({ fallbackFrom: "slack-hybrid" });
				expect(hits[0]?.content).toBe(breakLexical ? "from cli" : "we ship the release friday");
				expect(cliQueries).toEqual(breakLexical ? ["release"] : []);
			} finally {
				sidecar.close();
			}
			cliQueries.length = 0;
		}
	});

	it("honors the channel allowlist for sidecar hits", async () => {
		const skill = track(
			new SlackSkill({
				client: cliMustNotSearch,
				datasourceId: "slack",
				channelNames: ["not-eng"],
				sidecar: {
					descriptor: CRAWLER_DESCRIPTORS.slack,
					crawlerDbPath: crawlerDb,
					sidecarDbPath: join(root, "s.sqlite"),
					workspaceId: "T1",
				},
			}),
		);
		const bm25 = skill.retrievalMethods().find((method) => method.describe().name === "slack-sidecar-bm25");
		expect(await bm25?.retrieve("release", { topK: 3 })).toEqual([]);
	});

	it("still succeeds when embedding fails, and reports a warning instead of an error", async () => {
		const down: Embedder = {
			identity: async () => {
				throw new Error("gateway down");
			},
			embed: async () => [],
		};
		const skill = slackWithSidecar({ embedder: down });
		const result = await skill.index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics).toEqual([
			expect.objectContaining({
				code: "datasource-index-failed",
				severity: "warning",
				message: expect.stringContaining("semantic-unavailable: gateway down"),
			}),
		]);
		const [hybrid] = skill.retrievalMethods();
		expect(await hybrid?.retrieve("release", { topK: 3 })).toHaveLength(1);
	});

	it("says how many messages still wait for a vector so a partial semantic index is not mistaken for a complete one", async () => {
		const skill = slackWithSidecar({ batchSize: 1, maxEmbedPerSync: 1 });
		const result = await skill.index();
		expect(result.ok).toBe(true);
		expect(result.diagnostics).toEqual([
			expect.objectContaining({
				code: "datasource-index-partial",
				severity: "info",
				message: expect.stringContaining("1 remaining"),
			}),
		]);
		// The next refresh (same sidecar file) embeds the last message and reports nothing.
		expect((await slackWithSidecar({ batchSize: 1, maxEmbedPerSync: 1 }).index()).diagnostics).toEqual([]);
	});

	it("with semantic search off, registers only BM25 and never warns about a missing embedder", async () => {
		const skill = track(
			new SlackSkill({
				client: cliMustNotSearch,
				datasourceId: "slack",
				sidecar: {
					descriptor: CRAWLER_DESCRIPTORS.slack,
					crawlerDbPath: crawlerDb,
					sidecarDbPath: join(root, "lexical.sqlite"),
					workspaceId: "T1",
				},
			}),
		);
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual(["slack-sidecar-bm25"]);
		expect(await skill.index()).toMatchObject({ ok: true, diagnostics: [] });
	});

	it("keeps a successful CLI sync successful when the sidecar is locked by another process", async () => {
		const skill = slackWithSidecar();
		await skill.index();
		const other = new DatabaseSync(join(root, "sidecar.sqlite"));
		other.exec("pragma busy_timeout = 0; begin immediate");
		const db = new DatabaseSync(crawlerDb);
		db.exec(
			"insert into messages (channel_id, ts, workspace_id, user_id, text, normalized_text) values ('C1', '30.000001', 'T1', 'U1', 'deploy now', 'deploy now'); insert into message_fts values ('C1|30.000001', 'deploy now')",
		);
		db.close();
		try {
			const result = await skill.index();
			expect(result).toMatchObject({ ok: true, chunkCount: 7 });
			expect(result.diagnostics).toEqual([
				expect.objectContaining({ severity: "warning", message: expect.stringContaining("semantic-unavailable") }),
			]);
		} finally {
			other.exec("rollback");
			other.close();
		}
	}, 15_000);

	it("does not touch the sidecar when the CLI sync itself failed", async () => {
		const failing: CrawlerSkillClient = {
			sync: async () => ({ ok: false, reason: "binary-missing", stdout: "", stderr: "", code: null }),
			search: cliMustNotSearch.search,
		};
		const skill = track(
			new SlackSkill({
				client: failing,
				sidecar: {
					descriptor: CRAWLER_DESCRIPTORS.slack,
					crawlerDbPath: crawlerDb,
					sidecarDbPath: join(root, "s.sqlite"),
					workspaceId: "T1",
					embedder,
				},
			}),
		);
		expect((await skill.index()).ok).toBe(false);
	});

	it("is inert without a sidecar option: only the CLI-backed method exists", () => {
		const skill = new SlackSkill({ client: cliMustNotSearch });
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual(["slack-fts"]);
	});

	it("shares the same wiring across telegram and whatsapp", () => {
		for (const [skill, id] of [
			[
				track(
					new TelecrawlSkill({
						client: cliMustNotSearch,
						sidecar: {
							descriptor: CRAWLER_DESCRIPTORS.telegram,
							crawlerDbPath: crawlerDb,
							sidecarDbPath: join(root, "t.sqlite"),
							embedder,
						},
					}),
				),
				"telegram",
			],
			[
				track(
					new WacrawlSkill({
						client: cliMustNotSearch,
						sidecar: {
							descriptor: CRAWLER_DESCRIPTORS.whatsapp,
							crawlerDbPath: crawlerDb,
							sidecarDbPath: join(root, "w.sqlite"),
							embedder,
						},
					}),
				),
				"whatsapp",
			],
		] as const) {
			expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual([`${id}-hybrid`]);
		}
	});
});

/** wacrawl/telecrawl archive shape, as the real CLIs write it (event_id unique; msg_id unique per chat only). */
function seedChatArchive(path: string): void {
	const db = new DatabaseSync(path);
	db.exec(`
		create table messages (
			rowid integer primary key autoincrement, event_id text not null unique, chat_jid text not null,
			chat_name text, msg_id text not null, sender_name text, ts integer not null, text text, deleted_at integer
		);
		create virtual table messages_fts using fts5(text, chat, sender, media);
		insert into messages (event_id, chat_jid, chat_name, msg_id, sender_name, ts, text) values
			('evt-1', 'family@s', 'Family', 'm1', 'Bo', 1700000000, 'dinner at seven'),
			('evt-2', 'work@g', 'Work', 'm1', 'Al', 1700000100, 'dinner with the team');
		insert into messages_fts (rowid, text, chat, sender, media) select rowid, text, chat_name, sender_name, '' from messages;
	`);
	db.close();
}

/** A fake crawler CLI that prints `output` for every call, so the real client and its `parseHits` run. */
function fakeCrawlerCli(name: string, output: unknown): string {
	const binary = join(root, `${name}-fake.mjs`);
	writeFileSync(binary, `#!/usr/bin/env node\nprocess.stdout.write(${JSON.stringify(JSON.stringify(output))});\n`);
	chmodSync(binary, 0o755);
	return binary;
}

describe("telegram/whatsapp: CLI and sidecar agree on message identity", () => {
	for (const [id, Skill] of [
		["whatsapp", WacrawlSkill],
		["telegram", TelecrawlSkill],
	] as const) {
		it(`${id}: a message found by both the CLI and the sidecar has one id and one source`, async () => {
			const archive = join(root, `${id}.db`);
			seedChatArchive(archive);
			// The real CLI JSON row carries both ids; message_id repeats across chats, event_id does not.
			const binaryPath = fakeCrawlerCli(id, [
				{ event_id: "evt-1", message_id: "m1", chat_jid: "family@s", chat_name: "Family", text: "dinner at seven" },
			]);
			const cliOnly = track(new Skill({ instanceId: "main", connectorOptions: { binaryPath } }));
			const withSidecar = track(
				new Skill({
					instanceId: "main",
					connectorOptions: { binaryPath },
					sidecar: {
						descriptor: CRAWLER_DESCRIPTORS[id],
						crawlerDbPath: archive,
						sidecarDbPath: join(root, `${id}.sqlite`),
					},
				}),
			);
			const [cliMethod] = cliOnly.retrievalMethods();
			const [sidecarMethod] = withSidecar.retrievalMethods();
			const fromCli = (await cliMethod?.retrieve("dinner seven", { topK: 3 }))?.[0];
			const fromSidecar = (await sidecarMethod?.retrieve("dinner seven", { topK: 3 }))?.[0];
			expect(fromCli).toBeDefined();
			expect(fromSidecar?.id).toBe(fromCli?.id);
			expect(fromSidecar?.source).toBe(fromCli?.source);
		});
	}

	it("applies the channel allowlist in SQL: an allowlisted match is found behind many excluded ones, and excluded chats are never embedded", async () => {
		const archive = join(root, "allow.db");
		const db = new DatabaseSync(archive);
		db.exec(`
			create table messages (
				rowid integer primary key autoincrement, event_id text not null unique, chat_jid text not null,
				chat_name text, sender_name text, ts integer not null, text text, deleted_at integer
			);
			create virtual table messages_fts using fts5(text, chat, sender, media);
		`);
		const insert = db.prepare(
			"insert into messages (event_id, chat_jid, chat_name, sender_name, ts, text) values (?, ?, ?, 'Bo', ?, ?)",
		);
		for (let i = 0; i < 30; i += 1)
			insert.run(`busy-${i}`, "busy@g", "Busy", 1700000000 + i, "release release release");
		insert.run("fam-1", "family@s", "Family", 1700000100, "release on sunday");
		db.exec(
			"insert into messages_fts (rowid, text, chat, sender, media) select rowid, text, chat_name, sender_name, '' from messages",
		);
		db.close();
		const embedded: string[] = [];
		const recording: Embedder = {
			identity: embedder.identity,
			embed: async (texts) => {
				embedded.push(...texts);
				return embedder.embed(texts);
			},
		};
		const skill = track(
			new WacrawlSkill({
				client: cliMustNotSearch,
				channelIds: ["family@s"],
				sidecar: {
					descriptor: CRAWLER_DESCRIPTORS.whatsapp,
					crawlerDbPath: archive,
					sidecarDbPath: join(root, "allow.sqlite"),
					embedder: recording,
				},
			}),
		);
		await skill.index();
		expect(embedded).toEqual(["Family\nrelease on sunday"]);
		const [hybrid] = skill.retrievalMethods();
		expect((await hybrid?.retrieve("release", { topK: 5 }))?.map((hit) => hit.content)).toEqual([
			"release on sunday",
		]);
	});
});

describe("kakao: LazykatokSkill with a read-only sidecar", () => {
	function seedKakaoArchive(file: string): void {
		const db = new DatabaseSync(file);
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
			insert into chunks values ('chunk_1', 'acct', 'room_1', '오픈소스 개발과제', '정철현', '2026-04-13T05:57:50+00:00', '2026-04-13T05:57:50+00:00', 'we ship the release friday', 3);
			insert into chunks values ('chunk_2', 'acct', 'room_2', '점심방', '김민수', '2026-04-14T03:00:00+00:00', '2026-04-14T03:00:00+00:00', 'pizza for lunch', 1);
			insert into chunks_fts (rowid, chunk_id, text) select rowid, chunk_id, text from chunks;
		`);
		db.close();
	}

	it("runs doctor -> sync -> sidecar.sync() without calling lazykatok's native fastembed index", async () => {
		const archive = join(root, "archive.sqlite3");
		seedKakaoArchive(archive);
		let nativeIndexCalled = false;
		const client: LazykatokSkillClient = {
			doctor: async () => ({ ok: true, data: { ready: true }, stdout: "", stderr: "", code: 0 }),
			sync: async () => ({
				ok: true,
				data: { synced: true, messageCount: 4, metadata: { chunks: 2 } },
				stdout: "",
				stderr: "",
				code: 0,
			}),
			index: async () => {
				nativeIndexCalled = true;
				throw new Error("native lazykatok index must not run when sidecar owns embeddings");
			},
			search: async () => ({
				ok: true,
				hits: [
					{
						chunkId: "chunk_1",
						content: "we ship the release friday",
						score: 1,
						metadata: { chatName: "오픈소스 개발과제" },
					},
				],
				data: { hits: [] },
				stdout: "",
				stderr: "",
				code: 0,
			}),
		};
		const skill = new LazykatokSkill({
			client,
			instanceId: "default",
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.kakao,
				crawlerDbPath: archive,
				sidecarDbPath: join(root, "kakao-sidecar.sqlite"),
				embedder,
			},
		});
		try {
			const methods = skill.retrievalMethods().map((method) => method.describe().name);
			expect(methods).toEqual(["kakao-hybrid"]);

			const indexed = await skill.index();
			expect(nativeIndexCalled).toBe(false);
			expect(indexed).toMatchObject({ ok: true, chunkCount: 2, diagnostics: [] });

			const [hybrid] = skill.retrievalMethods();
			const hybridHits = await hybrid?.retrieve("release deploy", { topK: 3 });
			expect(hybridHits?.[0]?.id).toBe("kakao:default:chunk_1");
			expect(hybridHits?.[0]?.source).toBe("/kakao/default/chunks/chunk_1");
			expect(hybridHits?.[0]?.content).toBe("we ship the release friday");
		} finally {
			skill.close();
		}
	});
});
