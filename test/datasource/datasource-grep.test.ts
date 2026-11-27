import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ExtensionAPI, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { createParsedMirrorToolsExtension } from "../../src/agent/parsed-mirror-tools.ts";
import { AliasedDatasourceSkill } from "../../src/datasource/aliased-skill.ts";
import { AttachmentRootsSkill } from "../../src/datasource/attachments.ts";
import { ConnectorDatasourceSkill } from "../../src/datasource/connector-skill.ts";
import { DescribedDatasourceSkill } from "../../src/datasource/described-skill.ts";
import {
	formatDatasourceGrepOutput,
	grepDatasourceSkills,
	grepDirectoryWithRipgrep,
	resolveDatasourceGrepTarget,
} from "../../src/datasource/grep.ts";
import { CRAWLER_DESCRIPTORS } from "../../src/datasource/sidecar/descriptors.ts";
import { CrawlerSidecar } from "../../src/datasource/sidecar/index.ts";
import { DiscrawlSkill } from "../../src/datasource/skills/discrawl/skill.ts";
import { LazykatokSkill } from "../../src/datasource/skills/lazykatok/skill.ts";
import { MailcrawlSkill } from "../../src/datasource/skills/mailcrawl/skill.ts";
import { NotionSkill } from "../../src/datasource/skills/notion/index.ts";
import { ObsidianSkill } from "../../src/datasource/skills/obsidian/skill.ts";
import { TelecrawlSkill } from "../../src/datasource/skills/telecrawl/index.ts";
import { WombatSkill } from "../../src/datasource/skills/wombat/skill.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";

let root: string;
let workspace: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ds-grep-"));
	workspace = join(root, "workspace");
	mkdirSync(workspace, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function createTelegramDb(dbPath: string): void {
	const db = new DatabaseSync(dbPath);
	db.exec(`
		CREATE TABLE messages (
			event_id TEXT PRIMARY KEY,
			chat_jid TEXT NOT NULL,
			chat_name TEXT,
			sender_name TEXT,
			ts INTEGER,
			text TEXT,
			deleted_at INTEGER DEFAULT 0
		);
		CREATE VIRTUAL TABLE messages_fts USING fts5(text, content='messages', content_rowid='rowid');
		INSERT INTO messages (event_id, chat_jid, chat_name, sender_name, ts, text, deleted_at)
		VALUES
			('tg-1', 'chat-alpha', 'Ops Room', 'Alice', 1760000000, 'Deploy key TG_TOKEN_8821 failed on node-3', 0),
			('tg-2', 'chat-alpha', 'Ops Room', 'Bob', 1760000060, 'Lowercase tg_token_8821 noted in log', 0),
			('tg-3', 'chat-secret', 'Excluded Room', 'Mallory', 1760000120, 'Deploy key TG_TOKEN_8821 should be filtered out', 0),
			('tg-4', 'chat-alpha', 'Ops Room', 'Alice', 1760000180, 'Deleted message with TG_TOKEN_8821', 1760000200);
		INSERT INTO messages_fts(rowid, text) SELECT rowid, text FROM messages;
	`);
	db.close();
}

function createDiscordDb(dbPath: string): void {
	const db = new DatabaseSync(dbPath);
	db.exec(`
		CREATE TABLE guilds (id TEXT PRIMARY KEY, name TEXT);
		CREATE TABLE channels (id TEXT PRIMARY KEY, name TEXT);
		CREATE TABLE messages (
			id TEXT PRIMARY KEY,
			guild_id TEXT,
			channel_id TEXT,
			author_id TEXT,
			content TEXT,
			normalized_content TEXT,
			created_at TEXT,
			deleted_at TEXT,
			raw_json TEXT
		);
		CREATE VIRTUAL TABLE message_fts USING fts5(message_id UNINDEXED, normalized_content);
		INSERT INTO guilds VALUES ('g-1', 'NomaDamas');
		INSERT INTO channels VALUES ('c-1', 'infra');
		INSERT INTO messages VALUES (
			'dc-101', 'g-1', 'c-1', 'u-1',
			'Slurm job SLURM_ERR_7741 stuck on B200',
			'Slurm job SLURM_ERR_7741 stuck on B200',
			'2026-08-10T12:00:00Z',
			NULL,
			'{"author":{"username":"jeffrey"}}'
		);
		INSERT INTO message_fts(message_id, normalized_content) VALUES ('dc-101', 'Slurm job SLURM_ERR_7741 stuck on B200');
	`);
	db.close();
}

function createNotcrawlDb(dbPath: string): void {
	const db = new DatabaseSync(dbPath);
	db.exec(`
		CREATE TABLE pages (
			id TEXT PRIMARY KEY,
			title TEXT NOT NULL,
			url TEXT,
			last_edited_time INTEGER,
			alive INTEGER NOT NULL
		);
		CREATE TABLE blocks (
			id TEXT PRIMARY KEY,
			page_id TEXT,
			type TEXT NOT NULL,
			text TEXT,
			display_order INTEGER NOT NULL DEFAULT 0,
			alive INTEGER NOT NULL
		);
		INSERT INTO pages VALUES ('pg-1', 'Q4 Launch Plan', 'https://notion.so/pg-1', 1760000000000, 1);
		INSERT INTO blocks VALUES ('blk-1', 'pg-1', 'paragraph', 'Tracking code NOTION_CODE_5512 for partner sync.', 1, 1);
		INSERT INTO blocks VALUES ('blk-2', 'pg-1', 'paragraph', 'Deleted block NOTION_CODE_5512 ignored.', 2, 0);
	`);
	db.close();
}

function createMailcrawlSqlite(dbPath: string): void {
	const db = new DatabaseSync(dbPath);
	db.exec(`
		CREATE TABLE messages (
			message_id TEXT PRIMARY KEY,
			account_id TEXT NOT NULL,
			mailbox TEXT NOT NULL,
			provider_key TEXT NOT NULL UNIQUE,
			thread_id TEXT NOT NULL,
			subject TEXT NOT NULL,
			from_address TEXT NOT NULL,
			to_addresses TEXT NOT NULL,
			date TEXT NOT NULL
		);
		CREATE TABLE chunks (
			chunk_id TEXT PRIMARY KEY,
			account_id TEXT NOT NULL,
			mailbox TEXT NOT NULL,
			message_id TEXT NOT NULL,
			thread_id TEXT NOT NULL,
			section TEXT NOT NULL,
			ordinal INTEGER NOT NULL,
			text TEXT NOT NULL
		);
		INSERT INTO messages VALUES (
			'acct:INBOX:m1', 'acct', 'INBOX', 'acct:INBOX:pk1', 'acct:INBOX:t1',
			'Wire receipt MAIL_INV_3309', 'billing@vendor.test', '["me@test"]', '2026-08-12T09:00:00Z'
		);
		INSERT INTO chunks VALUES (
			'chunk-1', 'acct', 'INBOX', 'acct:INBOX:m1', 'acct:INBOX:t1', 'latest', 0,
			'Line 1\nReference MAIL_INV_3309 paid.\nLine 3'
		);
	`);
	db.close();
}

describe("CrawlerSidecar.grepExact", () => {
	it("matches exact literal & regex in read-only crawler DB, honors channel allowlist and liveWhere", () => {
		const dbPath = join(root, "telecrawl.db");
		createTelegramDb(dbPath);

		const sidecar = new CrawlerSidecar({
			descriptor: CRAWLER_DESCRIPTORS.telegram,
			crawlerDbPath: dbPath,
			sidecarDbPath: join(workspace, ".autorag/datasources/telegram/default/sidecar.sqlite"),
			channelIds: ["chat-alpha"],
		});
		try {
			const exact = sidecar.grepExact("TG_TOKEN_8821", { literal: true });
			expect(exact).toHaveLength(1);
			expect(exact[0]?.key).toBe("tg-1");
			expect(exact[0]?.metadata.chatId).toBe("chat-alpha");

			const ci = sidecar.grepExact("TG_TOKEN_8821", { literal: true, ignoreCase: true });
			expect(ci.map((h) => h.key)).toEqual(["tg-2", "tg-1"]);
		} finally {
			sidecar.close();
		}
	});
});

describe("DatasourceSkill.grep across real datasource types", () => {
	it("supports wombat mirror ripgrep (with catalog/markdown parsing) and agent filter", async () => {
		const wombatDir = join(root, "wombat-data");
		const mirrorDir = join(wombatDir, "mirror", "claude-code", "autorag");
		mkdirSync(mirrorDir, { recursive: true });
		writeFileSync(
			join(mirrorDir, "sess-1__abc123.md"),
			[
				"# Session claude-code:sess-1",
				"- agent: claude-code",
				"- session_id: sess-1",
				"- source_path: /logs/sess-1.jsonl",
				"- cwd: /work/AutoRAG",
				"- started_at: 2026-10-16T01:00:00Z",
				"",
				"## Turn 1 | User | 2026-10-16T01:00:05Z [src:L1]",
				"",
				"Check WOMBAT_ERR_4401 in parser",
				"",
			].join("\n"),
			"utf8",
		);
		// Ensure .minsync internal directory is excluded.
		mkdirSync(join(wombatDir, "mirror", ".minsync"), { recursive: true });
		writeFileSync(join(wombatDir, "mirror", ".minsync", "hidden.md"), "WOMBAT_ERR_4401\n", "utf8");

		const skill = new WombatSkill({
			dataDir: wombatDir,
			client: {
				doctor: async () => ({
					ok: true,
					data: { ready: true, indexedSessions: 1, indexedTurns: 1 },
					stdout: "",
					stderr: "",
					code: 0,
				}),
				sync: async () => ({
					ok: true,
					data: { totalSessions: 1, totalMessages: 1, indexOk: true },
					stdout: "",
					stderr: "",
					code: 0,
				}),
				search: async () => ({ ok: true, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});

		const hits = await skill.grep!("WOMBAT_ERR_4401", { literal: true });
		expect(hits).toHaveLength(1);
		expect(hits[0]?.source).toContain("/wombat/default/");
		expect(hits[0]?.lineNumber).toBe(10);
		expect(hits[0]?.content).toContain("claude-code:sess-1");
		expect(hits[0]?.content).toContain("WOMBAT_ERR_4401");
	});

	it("supports mailcrawl exact grep via sidecar SQLite read when CLI grep is absent", async () => {
		const mailDir = join(root, "mail-data");
		mkdirSync(mailDir, { recursive: true });
		createMailcrawlSqlite(join(mailDir, "archive.sqlite"));

		const skill = new MailcrawlSkill({
			dataDir: mailDir,
			account: "acct",
			mailbox: "INBOX",
			client: {
				sync: async () => ({ ok: true, data: { messages: 1 }, stdout: "", stderr: "", code: 0 }),
				index: async () => ({ ok: true, data: { embedded: 1 }, stdout: "", stderr: "", code: 0 }),
				search: async () => ({ ok: true, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});

		const hits = await skill.grep!("MAIL_INV_3309", { literal: true, context: 1 });
		expect(hits.length).toBeGreaterThanOrEqual(1);
		expect(hits[0]?.source).toContain("/mailcrawl/default/");
		expect(hits.some((h) => h.content.includes("Reference MAIL_INV_3309 paid."))).toBe(true);
	});

	it("supports discrawl sidecar exact grep and preserves aliased/described/attachment wrappers", async () => {
		const dbPath = join(root, "discrawl.db");
		createDiscordDb(dbPath);

		const raw = new DiscrawlSkill({
			datasourceId: "discord",
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.discord,
				crawlerDbPath: dbPath,
				sidecarDbPath: join(workspace, ".autorag/datasources/nomadamas/default/sidecar.sqlite"),
				workspaceId: "g-1",
			},
			client: {
				doctor: async () => ({
					ok: true,
					data: { ready: true, configOk: true, databaseOk: true, ftsOk: true, embeddingsOk: false },
					stdout: "",
					stderr: "",
					code: 0,
				}),
				sync: async () => ({ ok: true, data: { messages: 1 }, stdout: "", stderr: "", code: 0 }),
				embed: async () => ({
					ok: true,
					data: { processed: 0, succeeded: 0, failed: 0, remainingBacklog: 0 },
					stdout: "",
					stderr: "",
					code: 0,
				}),
				search: async () => ({ ok: true, data: { hits: [] }, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});
		const wrapped: DatasourceSkill = new AttachmentRootsSkill(
			new DescribedDatasourceSkill(
				new AliasedDatasourceSkill(raw, { alias: "nomadamas", channelIds: ["c-1"] }),
				"NomaDamas Discord",
			),
			[],
		);

		const hits = await wrapped.grep!("SLURM_ERR_7741", { literal: true });
		expect(hits).toHaveLength(1);
		expect(hits[0]?.source).toBe("/nomadamas/default/chunks/dc-101");
		expect(hits[0]?.content).toContain("SLURM_ERR_7741");
	});

	it("supports obsidian staged mirror ripgrep, connector chunk-store grep, and notcrawl SQLite sidecar grep", async () => {
		// 1. Obsidian
		const vault = join(root, "vault");
		mkdirSync(join(vault, "Projects"), { recursive: true });
		const notePath = join(vault, "Projects", "Alpha.md");
		writeFileSync(notePath, "# Alpha\nKey OBS_KEY_9090 here\n", "utf8");
		const stagedDir = join(workspace, ".autorag/datasources/obsidian/default/minsync/files/Projects");
		mkdirSync(stagedDir, { recursive: true });
		writeFileSync(join(stagedDir, "Alpha.md"), "# Alpha\nKey OBS_KEY_9090 here\n", "utf8");
		writeFileSync(
			join(workspace, ".autorag/datasources/obsidian/default/staging.json"),
			JSON.stringify({
				version: 1,
				entries: { "Projects/Alpha.md": { sourcePath: notePath, size: 28, mtimeNs: "1" } },
			}),
			"utf8",
		);
		const obs = new ObsidianSkill({ vaultPath: vault, workspaceRoot: workspace });
		const obsHits = await obs.grep!("OBS_KEY_9090", { literal: true });
		expect(obsHits).toHaveLength(1);
		expect(obsHits[0]?.source).toBe("/obsidian/default/Projects/Alpha.md");

		// 2. ConnectorDatasourceSkill (GitHub / Gist / RSS)
		const gh = new ConnectorDatasourceSkill(
			{
				skillName: "github",
				skillType: "github-issues",
				description: "GitHub",
				capabilities: ["issues"],
				defaultTags: ["github"],
				contentType: "issue",
				manifestDescription: "GitHub",
			},
			{
				workspaceRoot: workspace,
				connector: {
					fetch: async () => ({
						ok: true,
						documents: [
							{
								docId: "issue-42",
								hierarchy: ["org", "repo", "42"],
								title: "Fix GH_ERR_123",
								content: "Stack trace shows GH_ERR_123 on boot.",
								metadata: {},
							},
						],
					}),
				},
			},
		);
		await gh.index();
		const ghHits = await gh.grep!("GH_ERR_123", { literal: true });
		expect(ghHits).toHaveLength(1);
		expect(ghHits[0]?.source).toContain("/github/default/");

		// 3. Notion (notcrawl SQLite sidecar)
		const notcrawlDb = join(root, "notcrawl.db");
		createNotcrawlDb(notcrawlDb);
		const notion = new NotionSkill({
			connectorOptions: { databasePath: notcrawlDb },
			client: {
				sync: async () => ({ ok: true, count: 1, stdout: "", stderr: "", code: 0 }),
				search: async () => ({ ok: true, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});
		const notionHits = await notion.grep!("NOTION_CODE_5512", { literal: true });
		expect(notionHits).toHaveLength(1);
		expect(notionHits[0]?.source).toContain("/notion/default/");
		expect(notionHits[0]?.content).toContain("Q4 Launch Plan");

		// 4. Lazykatok native CLI fallback when sidecar is absent
		const kakaoCli = new LazykatokSkill({
			client: {
				doctor: async () => ({ ok: true, data: { ready: true }, stdout: "", stderr: "", code: 0 }),
				sync: async () => ({ ok: true, data: { synced: true }, stdout: "", stderr: "", code: 0 }),
				index: async () => ({ ok: true, data: { chunkCount: 1 }, stdout: "", stderr: "", code: 0 }),
				search: async () => ({ ok: true, hits: [], data: { hits: [] }, stdout: "", stderr: "", code: 0 }),
				grep: async (pattern) => ({
					ok: true,
					data: [
						{
							chunkId: "chk-42",
							chatId: "room-1",
							chatName: "Ops Sync",
							senderNickname: "Alex",
							startedAt: "2026-06-01T09:00:00Z",
							endedAt: "2026-06-01T09:01:00Z",
							lineNumber: 2,
							lineEnd: 2,
							matchedLine: `code=${pattern}`,
							snippet: `code=${pattern}`,
							contextBefore: [],
							contextAfter: [],
							parentChunkIds: [],
						},
					],
					stdout: "",
					stderr: "",
					code: 0,
				}),
			},
		});
		const kakaoHits = await kakaoCli.grep!("KATOK_ERR_88", { literal: true });
		expect(kakaoHits).toHaveLength(1);
		expect(kakaoHits[0]?.source).toBe("/kakao/default/chunks/chk-42");
		expect(kakaoHits[0]?.label).toBe("Ops Sync | Alex");
		expect(kakaoHits[0]?.content).toContain("KATOK_ERR_88");
	});
});

describe("grep tool integration with virtual datasource paths", () => {
	it("routes `grep(pattern, path='/telegram/...')` and `grep(pattern, path='/wombat')` through datasource exact grep and registers [eN] evidence", async () => {
		const tgDb = join(root, "tg.db");
		createTelegramDb(tgDb);
		const tgSkill = new TelecrawlSkill({
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.telegram,
				crawlerDbPath: tgDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/telegram/default/sidecar.sqlite"),
			},
			client: {
				sync: async () => ({ ok: true, count: 2, stdout: "", stderr: "", code: 0 }),
				search: async () => ({ ok: true, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});

		const ledger = new EvidenceLedger();
		const factory = createParsedMirrorToolsExtension({
			workspaceRoot: workspace,
			searchPaths: [],
			ledger,
			datasourceSkills: [tgSkill],
		});
		const tools = new Map<string, ToolDefinition>();
		factory({
			registerTool: (def: ToolDefinition) => {
				tools.set(def.name, def);
			},
		} as unknown as ExtensionAPI);

		const grepTool = tools.get("grep")!;
		expect(grepTool.description).toContain("'/telegram'");
		const res = await grepTool.execute(
			"call-1",
			{ pattern: "TG_TOKEN_8821", path: "/telegram", literal: true },
			undefined,
			undefined,
			{ cwd: workspace } as never,
		);
		const text = res.content.find((c) => c.type === "text")?.text ?? "";
		expect(text).toContain("[e1]");
		expect(text).toContain("/telegram/default/chunks/tg-1");
		expect(text).toContain("TG_TOKEN_8821");
		expect(ledger.lookup("e1", { allowLocalFiles: false })[0]?.source).toBe("/telegram/default/chunks/tg-3");
		expect(ledger.lookup("e2", { allowLocalFiles: false })[0]?.source).toBe("/telegram/default/chunks/tg-1");

		// Calling `read` on a virtual datasource path returns clear guidance instead of ENOENT.
		const readTool = tools.get("read")!;
		const readRes = await readTool.execute(
			"call-read-virtual",
			{ path: "/telegram/default/chunks/tg-1" },
			undefined,
			undefined,
			{ cwd: workspace } as never,
		);
		const readText = readRes.content.find((c) => c.type === "text")?.text ?? "";
		expect(readText).toContain("virtual datasource identifier");
		expect(readText).toContain("grep(");
	});

	it("preserves absence of grep() through datasource wrappers for non-greppable skills", () => {
		const nonGreppable: DatasourceSkill = {
			describe: () => ({
				name: "browser",
				id: "browser",
				type: "hister-web-history",
				description: "Browser history",
				capabilities: ["web"],
				tags: ["browser"],
				status: "active",
				requiresExternalCli: false,
				datasourceId: "browser",
			}),
			polling: () => ({ mode: "none" }),
			index: async () => ({
				ok: true,
				instanceId: "default",
				skill: "browser",
				chunkCount: 0,
				indexedAt: 0,
				diagnostics: [],
			}),
			retrievalMethods: () => [],
			describeSources: () => [],
			skillManifest: () => ({ name: "datasource-browser", description: "Browser", content: "Browser" }),
		};
		const wrapped = new AttachmentRootsSkill(
			new DescribedDatasourceSkill(new AliasedDatasourceSkill(nonGreppable, { alias: "web-hist" }), "Desc"),
			[],
		);
		expect(wrapped.grep).toBeUndefined();
		expect(resolveDatasourceGrepTarget("/web-hist", [wrapped])).toBeUndefined();
		expect(resolveDatasourceGrepTarget("/datasources", [wrapped])).toBeUndefined();
	});

	it("greps across all greppable datasources when path is '/datasources'", async () => {
		const tgDb = join(root, "tg2.db");
		createTelegramDb(tgDb);
		const tgSkill = new TelecrawlSkill({
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.telegram,
				crawlerDbPath: tgDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/telegram/default/sidecar.sqlite"),
			},
			client: {
				sync: async () => ({ ok: true, count: 2, stdout: "", stderr: "", code: 0 }),
				search: async () => ({ ok: true, hits: [], stdout: "", stderr: "", code: 0 }),
			},
		});

		expect(resolveDatasourceGrepTarget("/datasources", [tgSkill])).toEqual({ kind: "all", skills: [tgSkill] });
		expect(resolveDatasourceGrepTarget("/telegram/default", [tgSkill])).toEqual({
			kind: "single",
			skill: tgSkill,
			scope: "/telegram/default",
		});

		const result = await grepDatasourceSkills([tgSkill], "TG_TOKEN_8821", { literal: true });
		expect(result.hits.length).toBe(2);
		const formatted = formatDatasourceGrepOutput(result.hits, new EvidenceLedger(), {
			limit: 50,
			limitReached: false,
		});
		expect(formatted.output).toContain("[e1]");
		expect(formatted.output).toContain("TG_TOKEN_8821");
	});

	it("grepDirectoryWithRipgrep supports literal/regex and fallback when rgPath is unavailable", async () => {
		const dir = join(root, "md-dir");
		mkdirSync(dir, { recursive: true });
		writeFileSync(join(dir, "sample.md"), "alpha\nTARGET_XYZ_99\nomega\n", "utf8");
		const hits = await grepDirectoryWithRipgrep({
			rootDir: dir,
			pattern: "TARGET_XYZ_99",
			literal: true,
			context: 1,
			rgPath: "/nonexistent/rg-binary",
		});
		expect(hits).toHaveLength(1);
		expect(hits[0]?.lineNumber).toBe(2);
		expect(hits[0]?.snippet).toContain("alpha");
		expect(hits[0]?.snippet).toContain("omega");
	});
});
