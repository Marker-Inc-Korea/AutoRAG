import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DatasourceChunkStore } from "../../src/datasource/chunk-store.ts";
import type { ConnectorFetchResult, DatasourceConnector } from "../../src/datasource/connector.ts";
import { sanitizeIdSegment, sanitizeOpaqueText } from "../../src/datasource/connector.ts";
import { ConnectorDatasourceSkill, type ConnectorSkillDefinition } from "../../src/datasource/connector-skill.ts";

const DEFINITION: ConnectorSkillDefinition = {
	skillName: "slack",
	skillType: "slack-workspace",
	description: "Slack workspace datasource",
	capabilities: ["chat", "api", "polling"],
	defaultTags: ["slack", "chat", "pii"],
	contentType: "chat",
	manifestDescription: "Search indexed Slack messages and threads.",
};

class StubConnector implements DatasourceConnector {
	public result: ConnectorFetchResult = { ok: true, documents: [] };
	public calls = 0;

	async fetch(): Promise<ConnectorFetchResult> {
		this.calls += 1;
		return this.result;
	}
}

let tmpRoot: string;

beforeEach(() => {
	tmpRoot = mkdtempSync(join(tmpdir(), "autorag-connector-skill-"));
});

afterEach(() => {
	rmSync(tmpRoot, { recursive: true, force: true });
});

describe("sanitizeOpaqueText", () => {
	it("suppresses paths, urls, emails, and long tokens", () => {
		for (const dirty of [
			"failed at /Users/me/Library",
			"see C:\\Users\\me",
			"user bob@example.com denied",
			"visit https://internal.example.com",
			"token xoxb-123456789012345678901234567890 rejected",
		]) {
			expect(sanitizeOpaqueText(dirty)).toBe(
				"datasource operation failed; details suppressed for datasource privacy",
			);
		}
	});

	it("passes short clean text through", () => {
		expect(sanitizeOpaqueText("rate limit exceeded")).toBe("rate limit exceeded");
	});
});

describe("sanitizeIdSegment", () => {
	it("keeps already-safe segments", () => {
		expect(sanitizeIdSegment("general")).toBe("general");
		expect(sanitizeIdSegment("C0123.thread-9")).toBe("C0123.thread-9");
	});

	it("replaces unsafe characters and stays distinct via hashing", () => {
		const a = sanitizeIdSegment("issue #12/comment");
		const b = sanitizeIdSegment("issue #13/comment");
		expect(a).not.toBe(b);
		expect(a).toMatch(/^[A-Za-z0-9._-]+$/);
		expect(a).not.toContain("/");
	});
});

describe("DatasourceChunkStore", () => {
	it("chunks, persists, reloads, and searches documents", () => {
		const store = new DatasourceChunkStore({ skillName: "slack", instanceId: "ws-1", workspaceRoot: tmpRoot });
		const count = store.replaceDocuments([
			{
				docId: "msg-1",
				hierarchy: ["channels", "general"],
				title: "Deploy announcement",
				content: "We deploy the payments service on Friday after the freeze lifts.",
			},
			{ docId: "msg-2", hierarchy: ["channels", "random"], content: "Lunch options near the office are limited." },
		]);
		expect(count).toBe(2);

		const reloaded = new DatasourceChunkStore({ skillName: "slack", instanceId: "ws-1", workspaceRoot: tmpRoot });
		expect(reloaded.load()).toBe(true);
		const hits = reloaded.search("payments deploy Friday", 5);
		expect(hits.length).toBeGreaterThan(0);
		expect(hits[0]?.chunk.docId).toBe("msg-1");
	});

	it("splits long documents into bounded chunks with distinct ids", () => {
		const store = new DatasourceChunkStore({ skillName: "notion", instanceId: "ws", maxChunkChars: 50 });
		const count = store.replaceDocuments([
			{ docId: "page-1", content: `${"alpha ".repeat(20)}\n\n${"beta ".repeat(20)}` },
		]);
		expect(count).toBeGreaterThan(1);
		const ids = store.chunks().map((chunk) => chunk.chunkId);
		expect(new Set(ids).size).toBe(ids.length);
		for (const chunk of store.chunks()) {
			expect(chunk.content.length).toBeLessThanOrEqual(50);
		}
	});

	it("matches agglutinated Korean tokens and English inflections by prefix", () => {
		const store = new DatasourceChunkStore({ skillName: "slack", instanceId: "kr" });
		store.replaceDocuments([
			{ docId: "m1", content: "스테이징 인증서를 이번 주에 교체해 주세요" },
			{ docId: "m2", content: "Incremental indexing shipped in the new release." },
			{ docId: "m3", content: "점심 메뉴 추천 받습니다" },
		]);
		// "인증서" must match the particle-suffixed "인증서를"; "교체" matches "교체해".
		expect(store.search("인증서 교체", 3)[0]?.chunk.docId).toBe("m1");
		// "index" must match "indexing" by prefix.
		expect(store.search("index release", 3)[0]?.chunk.docId).toBe("m2");
		// Exact hits outrank prefix-only hits.
		const store2 = new DatasourceChunkStore({ skillName: "x", instanceId: "y" });
		store2.replaceDocuments([
			{ docId: "exact", content: "deploy now" },
			{ docId: "prefix", content: "deployment now" },
		]);
		expect(store2.search("deploy now", 2)[0]?.chunk.docId).toBe("exact");
	});

	it("returns empty for empty queries and empty stores without throwing", () => {
		const store = new DatasourceChunkStore({ skillName: "rss", instanceId: "feeds" });
		expect(store.search("", 5)).toEqual([]);
		expect(store.search("anything", 5)).toEqual([]);
	});
});

describe("ConnectorDatasourceSkill", () => {
	it("publishes descriptor with datasourceId, default tags, and instances", () => {
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector: new StubConnector(), instanceId: "ws-1" });
		expect(skill.describe()).toMatchObject({
			name: "slack",
			id: "slack",
			type: "slack-workspace",
			datasourceId: "slack",
			instanceId: "ws-1",
			instances: ["ws-1"],
			status: "active",
			tags: ["slack", "chat", "pii"],
		});
	});

	it("indexes connector documents and serves scoped lexical retrieval", async () => {
		const connector = new StubConnector();
		connector.result = {
			ok: true,
			documents: [
				{
					docId: "m1",
					hierarchy: ["channels", "general"],
					content: "Quarterly budget approved by finance in the general channel.",
				},
				{ docId: "m2", hierarchy: ["channels", "random"], content: "Random chatter about budget lunch." },
			],
		};
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector, instanceId: "ws-1", workspaceRoot: tmpRoot });

		const indexResult = await skill.index();
		expect(indexResult).toMatchObject({ ok: true, skill: "slack", instanceId: "ws-1", chunkCount: 2 });

		const [method] = skill.retrievalMethods();
		expect(method?.describe()).toMatchObject({ name: "slack-lexical", datasourceId: "slack" });
		const results = await method?.retrieve("budget", { topK: 10 });
		expect(results?.length).toBe(2);
		expect(results?.every((r) => r.source.startsWith("/slack/ws-1/chunks/"))).toBe(true);

		const scoped = await method?.retrieve("budget", { topK: 10, scope: "/slack/other/**" });
		expect(scoped).toEqual([]);
		const allowed = await method?.retrieve("budget", { topK: 10, allowedScopes: ["/slack/ws-1/**"] });
		expect(allowed?.length).toBe(2);
	});

	it("maps connector failures to path-opaque diagnostics without throwing", async () => {
		const connector = new StubConnector();
		connector.result = { ok: false, reason: "auth", message: "invalid token for bob@example.com" };
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector, instanceId: "ws-1" });

		const result = await skill.index();
		expect(result).toMatchObject({ ok: false, code: "datasource-auth-error" });
		const serialized = JSON.stringify(result);
		expect(serialized).not.toContain("bob@example.com");
		expect(serialized).not.toContain("/Users/");
	});

	it("maps rate-limit and permission failures onto dedicated codes", async () => {
		const connector = new StubConnector();
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector, instanceId: "ws-1" });

		connector.result = { ok: false, reason: "rate-limited" };
		expect(await skill.index()).toMatchObject({ ok: false, code: "datasource-rate-limited" });
		connector.result = { ok: false, reason: "permission" };
		expect(await skill.index()).toMatchObject({ ok: false, code: "datasource-permission-denied" });
		connector.result = { ok: false, reason: "not-configured" };
		expect(await skill.index()).toMatchObject({ ok: false, code: "datasource-unavailable" });
	});

	it("survives a throwing connector with an unavailable diagnostic", async () => {
		const throwing: DatasourceConnector = {
			async fetch() {
				throw new Error("boom at /secret/path");
			},
		};
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector: throwing, instanceId: "ws-1" });
		const result = await skill.index();
		expect(result).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect(JSON.stringify(result)).not.toContain("/secret/path");
	});

	it("tracks polling metadata across successful and failed indexing", async () => {
		const connector = new StubConnector();
		const skill = new ConnectorDatasourceSkill(DEFINITION, {
			connector,
			instanceId: "ws-1",
			pollingIntervalMs: 60_000,
		});
		expect(skill.polling()).toMatchObject({ mode: "poll", intervalMs: 60_000, lastIndexedAt: undefined });

		await skill.index();
		expect(skill.polling().lastIndexedAt).toEqual(expect.any(Number));
		expect(skill.polling().lastPolledAt).toEqual(expect.any(Number));

		connector.result = { ok: false, reason: "unavailable" };
		await skill.index();
		expect(skill.polling().lastError).toBeDefined();
	});

	it("emits a datasource-empty info diagnostic when the fetch yields nothing", async () => {
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector: new StubConnector(), instanceId: "ws-1" });
		const result = await skill.index();
		expect(result.ok).toBe(true);
		if (result.ok) {
			expect(result.diagnostics).toContainEqual(expect.objectContaining({ code: "datasource-empty" }));
		}
	});

	it("applies the dedupe window so repeated docs in one batch collapse", async () => {
		const connector = new StubConnector();
		connector.result = {
			ok: true,
			documents: [
				{ docId: "item-1", content: "Breaking news about markets." },
				{ docId: "item-1", content: "Breaking news about markets (duplicate delivery)." },
			],
		};
		const skill = new ConnectorDatasourceSkill(DEFINITION, {
			connector,
			instanceId: "feeds",
			dedupeWindowMs: 60_000,
		});
		const result = await skill.index();
		expect(result).toMatchObject({ ok: true, chunkCount: 1 });
	});

	it("exposes a path-opaque progressive-disclosure manifest and source descriptions", async () => {
		const connector = new StubConnector();
		connector.result = {
			ok: true,
			documents: [{ docId: "m1", hierarchy: ["channels", "general"], content: "hello world" }],
		};
		const skill = new ConnectorDatasourceSkill(DEFINITION, { connector, instanceId: "ws-1" });
		await skill.index();

		const manifest = skill.skillManifest();
		expect(manifest.name).toBe("datasource-slack");
		expect(manifest.content).toContain("search_datasource_documents");
		expect(manifest.content).toContain("/slack/ws-1");
		expect(manifest.content).not.toContain("/Users/");

		const sources = skill.describeSources();
		expect(sources.map((s) => s.source)).toContain("/slack/ws-1");
		expect(sources.map((s) => s.source)).toContain("/slack/ws-1/channels/general");
		for (const source of sources) {
			expect(source.source).not.toContain("#");
			expect(source.source).toMatch(/^\/slack\//);
		}
	});
});
