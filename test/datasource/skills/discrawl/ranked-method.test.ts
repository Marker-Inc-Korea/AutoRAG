import { createHash } from "node:crypto";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../../../src/agent/agent.ts";
import { AliasedDatasourceSkill } from "../../../../src/datasource/aliased-skill.ts";
import { discordDescriptor } from "../../../../src/datasource/sidecar/descriptors.ts";
import { DiscrawlClient } from "../../../../src/datasource/skills/discrawl/client.ts";
import {
	DiscrawlRankedMethod,
	type DiscrawlRankedMethodOptions,
} from "../../../../src/datasource/skills/discrawl/ranked-method.ts";
import { DiscrawlSkill, type DiscrawlSkillClient } from "../../../../src/datasource/skills/discrawl/skill.ts";
import type {
	DiscrawlSearchHit,
	DiscrawlSearchMode,
	DiscrawlSearchResult,
} from "../../../../src/datasource/skills/discrawl/types.ts";
import { buildDatasourceSkills } from "../../../../src/datasource/skills/factory.ts";
import { RetrievalEngine } from "../../../../src/retrieval/engine.ts";
import type { RetrievalMethod, RetrievalResult } from "../../../../src/retrieval/types.ts";

const OLD_MESSAGE = "1513467741523415161";
const OLD_SOURCE = `/discord/work/chunks/${OLD_MESSAGE}`;
let root: string;
let archive: string;
const opened: DiscrawlSkill[] = [];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-discord-ranked-"));
	archive = join(root, "discrawl.db");
});

afterEach(() => {
	vi.restoreAllMocks();
	for (const skill of opened.splice(0)) skill.close();
	rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
});

function ok(hits: readonly DiscrawlSearchHit[]): DiscrawlSearchResult {
	return { ok: true, code: 0, stdout: "", stderr: "", hits, data: { hits } };
}

function seedArchive(): DiscrawlSearchHit[] {
	const db = new DatabaseSync(archive);
	try {
		db.exec(
			[
				"create table guilds (id text primary key, name text)",
				"create table channels (id text primary key, guild_id text, name text)",
				"create table messages (id text primary key, guild_id text, channel_id text, author_id text, created_at text, deleted_at text, content text, normalized_content text, raw_json text)",
				"create virtual table message_fts using fts5(message_id unindexed, guild_id unindexed, channel_id unindexed, author_id unindexed, author_name, channel_name, content)",
				"insert into guilds values ('G1', 'Engineering')",
				"insert into channels values ('C1', 'G1', 'releases')",
			].join(";"),
		);
		const rows = [
			{ messageId: OLD_MESSAGE, content: "bug bug bug bug crash" },
			...Array.from({ length: 25 }, (_, index) => ({
				messageId: String(BigInt(OLD_MESSAGE) + BigInt(index + 1)),
				content: `bug ${"routine update and unrelated conversation ".repeat(20)}`,
			})),
		];
		const message = db.prepare("insert into messages values (?, 'G1', 'C1', 'U1', ?, null, ?, ?, '{}')");
		const fts = db.prepare("insert into message_fts values (?, 'G1', 'C1', 'U1', 'Ada', 'releases', ?)");
		for (const [index, row] of rows.entries()) {
			message.run(row.messageId, new Date(Date.UTC(2024, 0, index + 1)).toISOString(), row.content, row.content);
			fts.run(row.messageId, row.content);
		}
		// Match the native FTS recency window that omitted the older strong hit.
		return db
			.prepare(
				"select m.id as messageId, m.content from messages m join message_fts on message_fts.message_id = m.id where message_fts match 'bug' order by message_fts.rowid desc limit 20",
			)
			.all()
			.map((row) => ({
				messageId: String(row.messageId),
				content: String(row.content),
				score: 100,
				guildId: "G1",
				channelId: "C1",
				channelName: "releases",
			}));
	} finally {
		db.close();
	}
}

function nativeClient(recent: readonly DiscrawlSearchHit[]) {
	return {
		search: vi
			.fn<DiscrawlSkillClient["search"]>()
			.mockImplementation(async (mode, _query, options) =>
				ok(mode === "semantic" ? [] : recent.slice(0, options?.topK ?? 20)),
			),
		doctor: vi.fn<DiscrawlSkillClient["doctor"]>(),
		sync: vi.fn<DiscrawlSkillClient["sync"]>(),
		embed: vi.fn<DiscrawlSkillClient["embed"]>(),
	};
}

function archiveSkill(client: DiscrawlSkillClient, defaultMode: DiscrawlSearchMode = "hybrid"): DiscrawlSkill {
	const skill = new DiscrawlSkill({
		client,
		instanceId: "work",
		defaultMode,
		sidecar: { descriptor: discordDescriptor, crawlerDbPath: archive, sidecarDbPath: join(root, "sidecar.sqlite") },
	});
	opened.push(skill);
	return skill;
}

describe("Discord ranking through the public retrieval paths", () => {
	it.each(["all", "selected"] as const)(
		"keeps the old strong BM25 hit first through %s retrieval at topK=1",
		async (path) => {
			const recent = seedArchive();
			expect(recent).toHaveLength(20);
			expect(recent.map((hit) => hit.messageId)).not.toContain(OLD_MESSAGE);
			const before = createHash("sha256").update(readFileSync(archive)).digest("hex");
			const client = nativeClient(recent);
			const engine = new RetrievalEngine();
			engine.registerMany(archiveSkill(client).retrievalMethods());
			const result =
				path === "all"
					? await engine.retrieve("bug", { topK: 1 })
					: await engine.retrieveSelected("bug", { datasourceIds: ["discord"] }, { topK: 1 });
			expect(result.results.map((hit) => hit.source)).toEqual([OLD_SOURCE]);
			expect(result.results[0]?.metadata.retrievalMethods).toEqual(["discord-hybrid"]);
			expect(result.diagnostics).toEqual([]);
			expect(client.search.mock.calls.map(([mode]) => mode).sort()).toEqual(["fts", "semantic"]);
			expect(createHash("sha256").update(readFileSync(archive)).digest("hex")).toBe(before);
		},
	);

	it("preserves ranking and source identity when selecting a factory-built connection alias", async () => {
		const client = nativeClient(seedArchive());
		vi.spyOn(DiscrawlClient.prototype, "search").mockImplementation(client.search);
		const { skills } = buildDatasourceSkills(
			{
				engineering: {
					type: "discord",
					instanceId: "work",
					connector: { databasePath: archive },
				},
			},
			root,
		);
		const skill = skills[0];
		if (!(skill instanceof AliasedDatasourceSkill)) throw new Error("Expected a connection alias");
		// The alias has no close hook; release the underlying test-owned archive.
		opened.push((skill as unknown as { skill: DiscrawlSkill }).skill);
		const engine = new RetrievalEngine();
		engine.registerMany(skill.retrievalMethods());
		const result = await engine.retrieveSelected(
			"bug",
			{ datasourceIds: ["engineering"] },
			{ topK: 1, scope: "/engineering/work/**" },
		);
		expect(result.results.map((hit) => hit.source)).toEqual([`/engineering/work/chunks/${OLD_MESSAGE}`]);
		expect(result.results[0]?.metadata.retrievalMethods).toEqual(["engineering-hybrid"]);
		expect(result.results[0]?.metadata.datasourceId).toBe("engineering");
		expect(result.diagnostics).toEqual([]);
		expect(client.search.mock.calls.map(([mode]) => mode).sort()).toEqual(["fts", "semantic"]);
	});

	it("keeps scoped BM25 results and raw native failure details at topK=3", async () => {
		const recent = seedArchive();
		const client = nativeClient(recent);
		client.search.mockImplementation(async (mode) =>
			mode === "semantic"
				? { ok: false, reason: "nonzero-exit", code: 7, stdout: "", stderr: "discrawl: /archive/team.db locked" }
				: ok(recent),
		);
		const engine = new RetrievalEngine();
		engine.registerMany(archiveSkill(client).retrievalMethods());
		const result = await engine.retrieve("bug", { topK: 3, scope: "/discord/work/**" });
		expect(result.results).toHaveLength(3);
		expect(result.results[0]?.source).toBe(OLD_SOURCE);
		expect(result.results.every((hit) => hit.source.startsWith("/discord/work/chunks/"))).toBe(true);
		expect(result.results[0]?.metadata.retrievalDiagnostics).toEqual([
			{
				method: "discord-semantic",
				reason: expect.stringContaining(
					"discord search --mode semantic failed (nonzero-exit, exit code 7): discrawl: /archive/team.db locked",
				),
			},
		]);
	});

	it("keeps the same ranking in the actual dedicated datasource tool at topK=1", async () => {
		const client = nativeClient(seedArchive());
		const skill = archiveSkill(client);
		const agent = new AutoRAGAgent({
			searchPaths: [root],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			memoryEmbedder: false,
			minSync: { autoInstall: false },
			jikji: false,
			everything: false,
			fsearch: false,
			webSearch: false,
			datasourceSkills: [skill],
		});
		const tool = (agent as unknown as { tools: readonly AgentTool[] }).tools.find(
			(entry) => entry.name === "search_datasource_discord",
		);
		if (tool === undefined) throw new Error("Discord's dedicated tool was not registered");
		const result = await tool.execute("discord-ranking", { query: "bug", topK: 1 });
		expect(result.details.sources).toEqual([OLD_SOURCE]);
		const text = result.content[0]?.type === "text" ? result.content[0].text : "";
		expect(text).toContain("bug bug bug bug crash");
		expect(client.search.mock.calls.map(([mode]) => mode).sort()).toEqual(["fts", "semantic"]);
	});

	it("retains a native language-index hit when the base BM25 index has no match", async () => {
		seedArchive();
		const db = new DatabaseSync(archive);
		try {
			db.prepare("insert into messages values (?, 'G1', 'C1', 'U1', ?, null, ?, ?, '{}')").run(
				"korean-only",
				"2024-02-01T00:00:00Z",
				"내일배포완료",
				"내일배포완료",
			);
			db.prepare("insert into message_fts values (?, 'G1', 'C1', 'U1', 'Ada', 'releases', ?)").run(
				"korean-only",
				"내일배포완료",
			);
			db.exec("create virtual table message_fts_ko using fts5(message_id unindexed, content)");
			db.prepare("insert into message_fts_ko values (?, ?)").run("korean-only", "내일 배포 완료");
			expect(
				db.prepare("select count(*) as count from message_fts where message_fts match '배포'").get()?.count,
			).toBe(0);
			expect(
				db.prepare("select count(*) as count from message_fts_ko where message_fts_ko match '배포'").get()?.count,
			).toBe(1);
		} finally {
			db.close();
		}
		const client = nativeClient([{ messageId: "korean-only", content: "내일배포완료", score: 1 }]);
		const engine = new RetrievalEngine();
		engine.registerMany(archiveSkill(client).retrievalMethods());
		const result = await engine.retrieve("배포", { topK: 1 });
		expect(result.results.map((hit) => hit.source)).toEqual(["/discord/work/chunks/korean-only"]);
	});

	it.each(["fts", "semantic", "hybrid"] as const)("registers only the configured %s method with a sidecar", (mode) => {
		const skill = archiveSkill(nativeClient([]), mode);
		expect(skill.retrievalMethods().map((method) => method.describe().name)).toEqual([`discord-${mode}`]);
		expect(skill.retrievalMethods()[0]?.describe().datasourceId).toBe("discord");
	});
});

function hit(id: string, score: number, metadata: Record<string, unknown> = {}, instance = "work"): RetrievalResult {
	return {
		id: `discord:${instance}:${id}`,
		source: `/discord/${instance}/chunks/${id}`,
		content: id,
		score,
		metadata: { guildId: "G1", channelId: "C1", channelName: "releases", ...metadata },
	};
}

function backend(name: string, hits: RetrievalResult[] = []) {
	return {
		describe: () => ({
			name,
			type: "bm25" as const,
			description: name,
			status: "active" as const,
			capabilities: [],
			datasourceId: "discord",
			tags: ["discord"],
		}),
		retrieve: vi.fn<RetrievalMethod["retrieve"]>().mockResolvedValue(hits),
	};
}

function ranked(options: Partial<DiscrawlRankedMethodOptions> = {}) {
	const lexical = backend("discord-sidecar-bm25", [hit("strong", 0.001), hit("weak", 0.0001)]);
	const nativeFts = backend("discord-fts", [hit("weak", 100), hit("native-only", 1000)]);
	const semantic = backend("discord-semantic");
	return {
		lexical,
		nativeFts,
		semantic,
		method: new DiscrawlRankedMethod({ mode: "hybrid", lexical, nativeFts, semantic, ...options }),
	};
}

describe("DiscrawlRankedMethod composition", () => {
	it("assigns monotonic FTS scores and deduplicates native matches before the outer merger", async () => {
		const { method, semantic } = ranked({ mode: "fts" });
		const direct = await method.retrieve("bug", { topK: 3 });
		expect(direct.map((entry) => entry.content)).toEqual(["strong", "weak", "native-only"]);
		expect(direct[0]!.score).toBeGreaterThan(direct[1]!.score);
		expect(direct[1]!.score).toBeGreaterThan(direct[2]!.score);
		const engine = new RetrievalEngine();
		engine.register(method);
		expect((await engine.retrieve("bug", { topK: 3 })).results.map((entry) => entry.content)).toEqual([
			"strong",
			"weak",
			"native-only",
		]);
		expect(semantic.retrieve).not.toHaveBeenCalled();
	});

	it("fuses semantic evidence with the lexical ranking once", async () => {
		const semantic = backend("discord-semantic", [hit("weak", 0.9), hit("semantic-only", 0.8)]);
		const { method } = ranked({ semantic });
		const result = await method.retrieve("bug", { topK: 4 });
		expect(result.map((entry) => entry.content)).toEqual(["weak", "strong", "semantic-only", "native-only"]);
		expect(new Set(result.map((entry) => entry.source)).size).toBe(4);
	});

	it("keeps semantic-only retrieval independent and propagates its original failure", async () => {
		const semantic = backend("discord-semantic", [hit("semantic-only", 0.8)]);
		const { method, lexical, nativeFts } = ranked({ mode: "semantic", semantic });
		expect(await method.retrieve("query", { topK: 1 })).toEqual([hit("semantic-only", 0.8)]);
		const error = new Error("native semantic: model unavailable at /models/discord");
		semantic.retrieve.mockRejectedValueOnce(error);
		await expect(method.retrieve("query", { topK: 1 })).rejects.toBe(error);
		expect(lexical.retrieve).not.toHaveBeenCalled();
		expect(nativeFts.retrieve).not.toHaveBeenCalled();
	});

	it("keeps BM25 order and raw degradation details when native semantic fails", async () => {
		const { method, semantic } = ranked();
		semantic.retrieve.mockRejectedValue(new Error("discrawl semantic exit 7: /archive/db locked"));
		const result = await method.retrieve("bug", { topK: 2 });
		expect(result.map((entry) => entry.content)).toEqual(["strong", "weak"]);
		expect(result[0]?.metadata.diagnostic).toBe("semantic-unavailable");
		expect(result[0]?.metadata.retrievalDiagnostics).toEqual([
			{
				method: "discord-semantic",
				reason: expect.stringContaining("discrawl semantic exit 7: /archive/db locked"),
			},
		]);
	});

	it("falls back to native FTS with the sidecar failure preserved", async () => {
		const { method, lexical } = ranked({ mode: "fts" });
		lexical.retrieve.mockRejectedValue(new Error("cannot open /archive/team.db"));
		const result = await method.retrieve("bug", { topK: 1 });
		expect(result.map((entry) => entry.content)).toEqual(["weak"]);
		expect(result[0]?.metadata.retrievalDiagnostics).toEqual([
			{ method: "discord-sidecar-bm25", reason: expect.stringContaining("cannot open /archive/team.db") },
		]);
	});

	it("does not hide a failed branch behind an otherwise empty successful result", async () => {
		const lexical = backend("discord-sidecar-bm25");
		const nativeFts = backend("discord-fts");
		const semantic = backend("discord-semantic");
		const error = new Error("semantic CLI exited 2: embeddings missing");
		semantic.retrieve.mockRejectedValue(error);
		const { method } = ranked({ lexical, nativeFts, semantic });
		await expect(method.retrieve("query", {})).rejects.toBe(error);
	});

	it("preserves every raw failure when all retrieval branches fail", async () => {
		const { method, lexical, nativeFts, semantic } = ranked();
		lexical.retrieve.mockRejectedValue(new Error("sidecar: /archive/db unreadable"));
		nativeFts.retrieve.mockRejectedValue(new Error("native FTS: exit 4"));
		semantic.retrieve.mockRejectedValue(new Error("native semantic: model missing"));
		const error = await method.retrieve("query", {}).catch((failure: unknown) => failure);
		expect(error).toBeInstanceOf(AggregateError);
		expect((error as Error).message).toContain("sidecar: /archive/db unreadable");
		expect((error as Error).message).toContain("native FTS: exit 4");
		expect((error as Error).message).toContain("native semantic: model missing");
	});

	it.each([
		{ channelIds: ["C1"] },
		{ channelNames: ["releases"] },
		{ channelIds: ["other"], channelNames: ["releases"] },
	])("applies source, guild and channel restrictions to every branch: %j", async (channels) => {
		const entries = [
			hit("wrong-instance", 1, {}, "other"),
			hit("wrong-guild", 1, { guildId: "G2" }),
			hit("wrong-channel", 1, { channelId: "C2", channelName: "general" }),
			hit("allowed", 1),
		];
		const { method } = ranked({
			workspaceId: "G1",
			...channels,
			lexical: backend("discord-sidecar-bm25", entries),
			nativeFts: backend("discord-fts", entries),
			semantic: backend("discord-semantic", entries),
		});
		const result = await method.retrieve("query", { topK: 1, scope: "/discord/work/**" });
		expect(result.map((entry) => entry.content)).toEqual(["allowed"]);
	});

	it("forwards cancellation and never turns it into a lexical fallback", async () => {
		const controller = new AbortController();
		const error = new Error("query cancelled by caller");
		const { method, semantic } = ranked();
		semantic.retrieve.mockImplementation(async (_query, options) => {
			expect(options.signal).toBe(controller.signal);
			controller.abort(error);
			return [];
		});
		await expect(method.retrieve("query", { signal: controller.signal })).rejects.toBe(error);
	});

	it.each([{ query: "" }, { query: "query", topK: 0 }])(
		"skips all backends for empty work: %j",
		async ({ query, topK }) => {
			const { method, lexical, nativeFts, semantic } = ranked();
			await expect(method.retrieve(query, { topK })).resolves.toEqual([]);
			for (const method of [lexical, nativeFts, semantic]) expect(method.retrieve).not.toHaveBeenCalled();
		},
	);

	it("rejects a pre-aborted request before invoking any backend", async () => {
		const { method, lexical, nativeFts, semantic } = ranked();
		const error = new Error("already cancelled");
		await expect(method.retrieve("query", { signal: AbortSignal.abort(error) })).rejects.toBe(error);
		for (const method of [lexical, nativeFts, semantic]) expect(method.retrieve).not.toHaveBeenCalled();
	});
});
