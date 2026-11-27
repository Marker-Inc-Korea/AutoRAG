/**
 * Live manual QA & E2E verification for exact-match datasource `grep` across:
 * 1. Real `wombat` binary (`wombat sync` + `wombat grep --json` + AutoRAG `WombatSkill.grep` via `grep(path="/wombat")`)
 * 2. Real `mailcrawl` CLI (`mailcrawl sync` + `mailcrawl grep --json` + AutoRAG `MailcrawlSkill.grep` via `grep(path="/mailcrawl")`)
 * 3. Real read-only SQLite sidecars (`telegram`, `discord` aliased as `nomadamas`, `slack`, `whatsapp`, `kakao`, `notion`)
 * 4. Real `ObsidianSkill` staged mirror + `ConnectorDatasourceSkill` chunk store + `/datasources` fan-out
 * 5. Full `AutoRAGAgent` session executing `grep` calls on virtual datasource paths and registering `[eN]` citations
 *
 * Isolation: every path (`AUTORAG_HOME`, `AUTORAG_CONFIG`, workspace, datasource dirs) lives
 * under a fresh temp directory. Real `~/.autorag/config.json` SHA-1 is verified unchanged before and after.
 *
 * Run: bun scripts/manual-qa/run-qa-datasource-grep-live.ts
 */

import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { homedir, tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { MockBackend } from "jev-use";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
} from "../../src/agent/query-routing.ts";
import { AliasedDatasourceSkill } from "../../src/datasource/aliased-skill.ts";
import { CRAWLER_DESCRIPTORS } from "../../src/datasource/sidecar/descriptors.ts";
import { DiscrawlSkill } from "../../src/datasource/skills/discrawl/skill.ts";
import { MailcrawlSkill } from "../../src/datasource/skills/mailcrawl/skill.ts";
import { NotionSkill } from "../../src/datasource/skills/notion/index.ts";
import { TelecrawlSkill } from "../../src/datasource/skills/telecrawl/index.ts";
import { WombatSkill } from "../../src/datasource/skills/wombat/skill.ts";
import { citationsRelevant } from "../../test/helpers/citations-relevant.ts";

function shaOfIfExists(path: string): string {
	if (!existsSync(path)) return "absent";
	return createHash("sha1").update(readFileSync(path)).digest("hex");
}

const realConfigPath = join(homedir(), ".autorag", "config.json");
const realConfigBefore = shaOfIfExists(realConfigPath);

const tmpRoot = realpathSync(mkdtempSync(join(tmpdir(), "autorag-ds-grep-qa-")));
const workspace = join(tmpRoot, "workspace");
const docs = join(tmpRoot, "docs");
const autoragHome = join(tmpRoot, ".autorag-home");
process.env.AUTORAG_HOME = autoragHome;
process.env.AUTORAG_CONFIG = join(autoragHome, "config.json");
mkdirSync(workspace, { recursive: true });
mkdirSync(docs, { recursive: true });
mkdirSync(autoragHome, { recursive: true });
writeFileSync(join(docs, "readme.txt"), "local placeholder doc\n", "utf8");

let failures = 0;
function check(name: string, pass: boolean, detail?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${detail ? ` — ${detail}` : ""}`);
}

const wombatBin = process.env.WOMBAT_BIN ?? "wombat";
const mailcrawlCli = process.env.MAILCRAWL_CLI;

try {
	// 1. Live wombat binary sync + grep
	const wombatData = join(tmpRoot, "wombat-data");
	const claudeSessions = join(tmpRoot, "claude-sessions/proj-qa");
	mkdirSync(claudeSessions, { recursive: true });
	writeFileSync(
		join(claudeSessions, "qa-sess-1.jsonl"),
		`${JSON.stringify({
			type: "user",
			sessionId: "qa-sess-1",
			cwd: "/work/qa",
			timestamp: "2026-11-20T09:00:00Z",
			message: { role: "user", content: "Track down LIVE_WOMBAT_CODE_9102 in session logs" },
		})}\n${JSON.stringify({
			type: "assistant",
			sessionId: "qa-sess-1",
			timestamp: "2026-11-20T09:00:05Z",
			message: { role: "assistant", content: "LIVE_WOMBAT_CODE_9102 occurred in retry loop." },
		})}\n`,
		"utf8",
	);

	const wombatRunnable = (() => {
		try {
			execFileSync(wombatBin, ["--version"], { stdio: "pipe" });
			return true;
		} catch {
			return false;
		}
	})();
	const wombatHasGrep = (() => {
		if (!wombatRunnable) return false;
		try {
			execFileSync(wombatBin, ["grep", "--help"], { stdio: "pipe" });
			return true;
		} catch {
			return false;
		}
	})();

	if (wombatRunnable) {
		execFileSync(
			wombatBin,
			[
				"--data-dir",
				wombatData,
				"--source",
				`claude-code=${claudeSessions}`,
				"--embedder",
				"none",
				"--json",
				"sync",
			],
			{ stdio: "pipe" },
		);
		if (wombatHasGrep) {
			const wombatGrepOut = execFileSync(
				wombatBin,
				["--data-dir", wombatData, "--json", "grep", "--literal", "LIVE_WOMBAT_CODE_9102"],
				{ encoding: "utf8" },
			);
			const wombatHits = JSON.parse(wombatGrepOut) as Array<{ catalog_key: string; turn_index: number }>;
			check(
				"wombat CLI live sync + grep",
				wombatHits.length === 2 && wombatHits[0]?.catalog_key === "claude-code:qa-sess-1",
				`hits=${wombatHits.length}`,
			);
		}
	} else {
		check("wombat binary present", false, `missing ${wombatBin}`);
	}

	// 2. Live mailcrawl CLI sync + grep
	const mailData = join(tmpRoot, "mailcrawl-data");
	const fixturePath = join(tmpRoot, "mail-fixture.json");
	writeFileSync(
		fixturePath,
		JSON.stringify([
			{
				accountId: "qa-acct",
				mailbox: "INBOX",
				providerKey: "qa-m-1",
				messageId: "qa-m-1",
				threadId: "qa-t-1",
				subject: "Settlement LIVE_MAIL_INV_7782",
				from: "ops@vendor.example",
				to: ["qa@acme.example"],
				cc: [],
				date: "2026-11-20T10:00:00Z",
				text: "Please verify LIVE_MAIL_INV_7782 before Friday.",
			},
		]),
		"utf8",
	);

	const runMailcrawl = (args: string[]): string =>
		mailcrawlCli !== undefined && mailcrawlCli.length > 0
			? execFileSync("node", [mailcrawlCli, ...args], { encoding: "utf8" })
			: execFileSync("mailcrawl", args, { encoding: "utf8" });

	const mailcrawlHasGrep = (() => {
		try {
			runMailcrawl(["grep", "--help"]);
			return true;
		} catch {
			return false;
		}
	})();

	runMailcrawl(["--data-dir", mailData, "sync", "--source", "fixture", "--fixture", fixturePath, "--json"]);
	if (mailcrawlHasGrep) {
		const mailGrepOut = runMailcrawl([
			"--data-dir",
			mailData,
			"grep",
			"--literal",
			"--json",
			"LIVE_MAIL_INV_7782",
		]);
		const mailHits = JSON.parse(mailGrepOut) as Array<{ mode: string; line: string }>;
		check(
			"mailcrawl CLI live sync + grep",
			mailHits.length >= 2 && mailHits.every((h) => h.mode === "grep"),
			`hits=${mailHits.length}`,
		);
	}

	// 3. Read-only SQLite sidecar archives (Telegram, Discord aliased as nomadamas, Notion)
	const tgDbPath = join(tmpRoot, "telecrawl.db");
	{
		const db = new DatabaseSync(tgDbPath);
		db.exec(`
			CREATE TABLE messages (event_id TEXT PRIMARY KEY, chat_jid TEXT, chat_name TEXT, sender_name TEXT, ts INTEGER, text TEXT, deleted_at INTEGER DEFAULT 0);
			CREATE VIRTUAL TABLE messages_fts USING fts5(text, content='messages', content_rowid='rowid');
			INSERT INTO messages VALUES ('tg-live-1', 'c1', 'QA Ops', 'Gajae', 1763600000, 'Build hash LIVE_TG_HASH_3344 passed', 0);
			INSERT INTO messages_fts(rowid, text) SELECT rowid, text FROM messages;
		`);
		db.close();
	}

	const dcDbPath = join(tmpRoot, "discrawl.db");
	{
		const db = new DatabaseSync(dcDbPath);
		db.exec(`
			CREATE TABLE guilds (id TEXT PRIMARY KEY, name TEXT);
			CREATE TABLE channels (id TEXT PRIMARY KEY, name TEXT);
			CREATE TABLE messages (id TEXT PRIMARY KEY, guild_id TEXT, channel_id TEXT, author_id TEXT, content TEXT, normalized_content TEXT, created_at TEXT, deleted_at TEXT, raw_json TEXT);
			CREATE VIRTUAL TABLE message_fts USING fts5(message_id UNINDEXED, normalized_content);
			INSERT INTO guilds VALUES ('g1', 'NomaDamas');
			INSERT INTO channels VALUES ('ch1', 'gpu');
			INSERT INTO messages VALUES ('dc-live-1', 'g1', 'ch1', 'u1', 'Node LIVE_DC_NODE_5511 rebooted', 'Node LIVE_DC_NODE_5511 rebooted', '2026-11-20T11:00:00Z', NULL, '{"author":{"username":"jeffrey"}}');
			INSERT INTO message_fts VALUES ('dc-live-1', 'Node LIVE_DC_NODE_5511 rebooted');
		`);
		db.close();
	}

	const notionDbPath = join(tmpRoot, "notcrawl.db");
	{
		const db = new DatabaseSync(notionDbPath);
		db.exec(`
			CREATE TABLE pages (id TEXT PRIMARY KEY, title TEXT NOT NULL, url TEXT, last_edited_time INTEGER, alive INTEGER NOT NULL);
			CREATE TABLE blocks (id TEXT PRIMARY KEY, page_id TEXT, type TEXT NOT NULL, text TEXT, display_order INTEGER NOT NULL DEFAULT 0, alive INTEGER NOT NULL);
			INSERT INTO pages VALUES ('np-1', 'Release Checklist', 'https://notion.so/np-1', 1763600000000, 1);
			INSERT INTO blocks VALUES ('nb-1', 'np-1', 'paragraph', 'Verify flag LIVE_NOTION_FLAG_6622 before cut.', 1, 1);
		`);
		db.close();
	}

	// 4. Full AutoRAGAgent session executing `grep` on `/wombat`, `/mailcrawl`, `/telegram`, `/nomadamas`, `/notion`, `/datasources`
	const wombatSkill = new WombatSkill({
		dataDir: wombatData,
		binaryPath: wombatBin,
	});
	const mailSkill = new MailcrawlSkill({
		dataDir: mailData,
		account: "qa-acct",
		mailbox: "INBOX",
	});
	const tgSkill = new TelecrawlSkill({
		sidecar: {
			descriptor: CRAWLER_DESCRIPTORS.telegram,
			crawlerDbPath: tgDbPath,
			sidecarDbPath: join(workspace, ".autorag/datasources/telegram/default/sidecar.sqlite"),
		},
	});
	const dcSkill = new AliasedDatasourceSkill(
		new DiscrawlSkill({
			datasourceId: "discord",
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.discord,
				crawlerDbPath: dcDbPath,
				sidecarDbPath: join(workspace, ".autorag/datasources/nomadamas/default/sidecar.sqlite"),
				workspaceId: "g1",
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
		}),
		{ alias: "nomadamas" },
	);
	const notionSkill = new NotionSkill({
		connectorOptions: { databasePath: notionDbPath },
	});

	const reg = registerFauxProvider({ api: `faux-qa-ds-grep-${randomUUID()}`, models: [{ id: "qa-model" }] });
	reg.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall("grep", { i: "Grep wombat sessions", pattern: "LIVE_WOMBAT_CODE_9102", path: "/wombat", literal: true }),
				fauxToolCall("grep", { i: "Grep mailcrawl emails", pattern: "LIVE_MAIL_INV_7782", path: "/mailcrawl", literal: true }),
				fauxToolCall("grep", { i: "Grep telegram chats", pattern: "LIVE_TG_HASH_3344", path: "/telegram", literal: true }),
				fauxToolCall("grep", { i: "Grep nomadamas discord", pattern: "LIVE_DC_NODE_5511", path: "/nomadamas", literal: true }),
				fauxToolCall("grep", { i: "Grep notion pages", pattern: "LIVE_NOTION_FLAG_6622", path: "/notion", literal: true }),
			],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			[fauxToolCall("grep", { i: "Grep all datasources", pattern: "LIVE_", path: "/datasources", literal: true })],
			{ stopReason: "toolUse" },
		),
		fauxAssistantMessage(
			"Found all exact codes across wombat [e1], mailcrawl [e3], telegram [e5], discord/nomadamas [e6], and notion [e7] (and [e2], [e4]).",
		),
	]);

	const jev = citationsRelevant(
		new MockBackend({
			[QUERY_ROUTE_QUESTION_ID]: {
				answer: "extra",
				distribution: { local: 0.1, organize: 0.1, extra: 0.8 },
				confidence: 0.8,
			},
			[DECOMPOSE_QUESTION_ID]: { answer: 0.1 },
			[FOLLOW_UP_QUESTION_ID]: { answer: 0.1 },
		}),
	);

	const agent = new AutoRAGAgent({
		model: reg.getModel(),
		apiKey: "qa-key",
		searchPaths: [docs],
		workspacePath: workspace,
		memoryPath: join(autoragHome, "memory.json"),
		minSync: { autoInstall: false, embedder: { id: "none" } },
		datasourceSkills: [wombatSkill, mailSkill, tgSkill, dcSkill, notionSkill],
		jikji: false,
		everything: false,
		fsearch: false,
		dupey: false,
		filee: false,
		webSearch: false,
		jev: { backend: jev },
	});

	const toolOutputs: { name: string; text: string }[] = [];
	agent.subscribe((event) => {
		if (event.type === "tool_execution_end") {
			const text = event.result.content
				.filter((c: { type: string }) => c.type === "text")
				.map((c: { text?: string }) => c.text ?? "")
				.join("\n");
			toolOutputs.push({ name: event.toolName, text });
		}
	});

	try {
		const res = await agent.searchDocuments("Find all LIVE_ codes across every datasource");
		const allToolText = toolOutputs.map((t) => t.text).join("\n");
		const sources = res.results.map((r) => r.source);
		check("Agent returned grounded citations", res.results.length >= 5, `results=${res.results.length}`);
		check("Cited /wombat hit", sources.some((s) => s.startsWith("/wombat/")), sources.join(", "));
		check("Cited /mailcrawl hit", sources.some((s) => s.startsWith("/mailcrawl/")), sources.join(", "));
		check("Cited /telegram hit", sources.some((s) => s.startsWith("/telegram/")), sources.join(", "));
		check("Cited /nomadamas (discord alias) hit", sources.some((s) => s.startsWith("/nomadamas/")), sources.join(", "));
		check("Cited /notion hit", sources.some((s) => s.startsWith("/notion/")), sources.join(", "));
		check(
			"/datasources fan-out grep found all 5 live datasources",
			["/wombat/", "/mailcrawl/", "/telegram/", "/nomadamas/", "/notion/"].every((p) => allToolText.includes(p)),
		);
	} finally {
		tgSkill.close();
		reg.unregister();
	}

	// 5. Read-only live probe on real host archives (zero writes, strictly read-only)
	const realWombatMirror = join(homedir(), ".wombat", "mirror");
	if (existsSync(realWombatMirror) && wombatHasGrep) {
		const t0 = Date.now();
		const out = execFileSync(wombatBin, ["--json", "grep", "--literal", "--limit", "5", "AutoRAG"], {
			encoding: "utf8",
		});
		const elapsed = Date.now() - t0;
		const parsed = JSON.parse(out) as unknown[];
		check("Real ~/.wombat/mirror live grep", parsed.length > 0, `hits=${parsed.length}, ${elapsed}ms`);
	}

	const realMailDir = join(homedir(), ".mailcrawl", "gmail-2000");
	if (existsSync(join(realMailDir, "archive.sqlite")) && mailcrawlHasGrep) {
		const t0 = Date.now();
		const out = runMailcrawl([
			"--data-dir",
			realMailDir,
			"grep",
			"--literal",
			"--ignore-case",
			"--limit",
			"5",
			"--json",
			"github",
		]);
		const elapsed = Date.now() - t0;
		const parsed = JSON.parse(out) as unknown[];
		check("Real ~/.mailcrawl/gmail-2000 live grep", parsed.length > 0, `hits=${parsed.length}, ${elapsed}ms`);
	}

	const realTgDb = join(homedir(), ".telecrawl", "telecrawl.db");
	if (existsSync(realTgDb)) {
		const t0 = Date.now();
		const realTg = new TelecrawlSkill({
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.telegram,
				crawlerDbPath: realTgDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/telegram-ro/default/sidecar.sqlite"),
			},
		});
		const hits = await realTg.grep("http", { literal: true, ignoreCase: true, limit: 5 });
		realTg.close();
		check("Real ~/.telecrawl/telecrawl.db sidecar live grep", hits.length > 0, `hits=${hits.length}, ${Date.now() - t0}ms`);
	}

	const realSlackDb = join(homedir(), ".slacrawl", "slacrawl.db");
	if (existsSync(realSlackDb)) {
		const t0 = Date.now();
		const { SlackSkill } = await import("../../src/datasource/skills/slack/index.ts");
		const realSlack = new SlackSkill({
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.slack,
				crawlerDbPath: realSlackDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/slack-ro/default/sidecar.sqlite"),
			},
		});
		const hits = await realSlack.grep("AutoRAG", { literal: true, ignoreCase: true, limit: 5 });
		realSlack.close();
		check("Real ~/.slacrawl/slacrawl.db sidecar live grep", hits.length > 0, `hits=${hits.length}, ${Date.now() - t0}ms`);
	}

	const realWaDb = join(homedir(), ".wacrawl", "wacrawl.db");
	if (existsSync(realWaDb)) {
		const t0 = Date.now();
		const { WacrawlSkill } = await import("../../src/datasource/skills/wacrawl/index.ts");
		const realWa = new WacrawlSkill({
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.whatsapp,
				crawlerDbPath: realWaDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/whatsapp-ro/default/sidecar.sqlite"),
			},
		});
		const hits = await realWa.grep("Bali", { literal: true, ignoreCase: true, limit: 5 });
		realWa.close();
		check("Real ~/.wacrawl/wacrawl.db sidecar live grep", hits.length > 0, `hits=${hits.length}, ${Date.now() - t0}ms`);
	}

	const realKakaoDb = join(homedir(), "Library", "Application Support", "katok", "archive.sqlite3");
	if (existsSync(realKakaoDb)) {
		const t0 = Date.now();
		const { LazykatokSkill } = await import("../../src/datasource/skills/lazykatok/skill.ts");
		const { LazykatokClient } = await import("../../src/datasource/skills/lazykatok/client.ts");
		const realKakao = new LazykatokSkill({
			client: new LazykatokClient({}),
			sidecar: {
				descriptor: CRAWLER_DESCRIPTORS.kakao,
				crawlerDbPath: realKakaoDb,
				sidecarDbPath: join(workspace, ".autorag/datasources/kakao-ro/default/sidecar.sqlite"),
			},
		});
		const hits = await realKakao.grep("http", { literal: true, ignoreCase: true, limit: 5 });
		realKakao.close();
		check("Real katok archive.sqlite3 sidecar live grep", hits.length > 0, `hits=${hits.length}, ${Date.now() - t0}ms`);
	}
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
	const realConfigAfter = shaOfIfExists(realConfigPath);
	check(
		"Host ~/.autorag/config.json untouched",
		realConfigBefore === realConfigAfter,
		`before=${realConfigBefore.slice(0, 12)} after=${realConfigAfter.slice(0, 12)}`,
	);
}

if (failures > 0) {
	console.error(`\nManual QA failed with ${failures} failure(s).`);
	process.exit(1);
}
console.log("\nAll live datasource grep QA checks passed.");
