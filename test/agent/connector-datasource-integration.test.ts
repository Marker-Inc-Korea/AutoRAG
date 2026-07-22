import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { GitHubSkill } from "../../src/datasource/skills/github/index.ts";
import { ObsidianSkill } from "../../src/datasource/skills/obsidian/index.ts";
import { RssSkill } from "../../src/datasource/skills/rss/index.ts";
import { SlackSkill } from "../../src/datasource/skills/slack/index.ts";
import { createMockFetch } from "../fixtures/mock-fetch.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-connector-integration-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function slackSkill(): SlackSkill {
	const mock = createMockFetch([
		{
			match: "conversations.list",
			json: { ok: true, channels: [{ id: "C01", name: "general" }], response_metadata: {} },
		},
		{
			match: "conversations.history",
			json: {
				ok: true,
				messages: [{ ts: "1700000001.000100", user: "U1", text: "Budget approved for the launch" }],
			},
		},
	]);
	return new SlackSkill({
		instanceId: "ws-1",
		workspaceRoot: tmpDir,
		connectorOptions: { token: "xoxb-secret", fetchImpl: mock.fetchImpl },
	});
}

function githubSkill(): GitHubSkill {
	const mock = createMockFetch([
		{
			match: "/repos/acme/app/issues",
			json: [
				{
					number: 7,
					title: "Fix login retry loop",
					body: "Login retries forever when the token expires.",
					state: "open",
					updated_at: "2024-03-01T00:00:00.000Z",
					labels: [],
				},
			],
		},
	]);
	return new GitHubSkill({
		instanceId: "acme",
		workspaceRoot: tmpDir,
		connectorOptions: { repos: ["acme/app"], fetchImpl: mock.fetchImpl },
	});
}

describe("AutoRAGAgent with connector-backed datasource skills", () => {
	it("indexes during refresh and searches via search_datasource_documents under trusted scopes", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: false,
			datasourceSkills: [slackSkill(), githubSkill()],
			datasourceAccess: {
				allowedTags: ["slack", "github"],
				allowedScopes: ["/slack/ws-1/**", "/github/acme/**"],
			},
		});

		const refresh = await agent.refresh(true, { methods: ["datasources"] });
		expect(refresh.datasources?.every((result) => result.ok)).toBe(true);

		const tool = createSearchDatasourceDocumentsTool(agent);
		const slackResponse = await tool.execute("call-1", { query: "budget approved launch" });
		expect(slackResponse.details.resultCount).toBeGreaterThan(0);
		expect(slackResponse.details.sources.every((source) => /^\/(slack|github)\//u.test(source))).toBe(true);

		const githubResponse = await tool.execute("call-2", { query: "login retry token expires" });
		expect(githubResponse.details.sources.some((source) => source.startsWith("/github/acme/chunks/"))).toBe(true);

		// Scope narrows to slack only.
		const scoped = await tool.execute("call-3", { query: "login retry token expires", scope: "/slack/**" });
		expect(scoped.details.sources.every((source) => source.startsWith("/slack/"))).toBe(true);
	});

	it("stays default-deny for connector skills without trusted access", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: false,
			datasourceSkills: [slackSkill()],
		});
		await agent.refresh(true, { methods: ["datasources"] });

		const { results } = await agent.searchDatasourceDocuments("budget approved");
		expect(results).toEqual([]);

		const tool = createSearchDatasourceDocumentsTool(agent);
		const response = await tool.execute("call-deny", {
			query: "budget approved",
			scope: "/slack/**",
			allowedTags: ["slack"],
			allowedScopes: ["/slack/**"],
		} as never);
		expect(response.details.resultCount).toBe(0);
	});

	it("announces authorized connector skills in the system prompt and loads them on demand", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: false,
			datasourceSkills: [slackSkill(), githubSkill()],
			datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack/ws-1/**"] },
		});

		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("datasource-slack");
		// github is not authorized: omitted entirely (default-deny).
		expect(prompt).not.toContain("datasource-github");

		const loadTool = createLoadDatasourceSkillTool(agent);
		const loaded = await loadTool.execute("call-load", { name: "datasource-slack" });
		expect(loaded.details).toEqual({ skill: "datasource-slack", loaded: true });
		const denied = await loadTool.execute("call-denied", { name: "datasource-github" });
		expect(denied.details).toEqual({ skill: "datasource-github", loaded: false });
	});

	it("degrades to path-opaque diagnostics when a connector fails during refresh", async () => {
		const mock = createMockFetch([{ match: "conversations.list", status: 401 }]);
		const failing = new SlackSkill({
			instanceId: "ws-1",
			workspaceRoot: tmpDir,
			connectorOptions: { token: "xoxb-secret", fetchImpl: mock.fetchImpl },
		});
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: false,
			datasourceSkills: [failing],
			datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack/**"] },
		});

		const refresh = await agent.refresh(true, { methods: ["datasources"] });
		expect(refresh.datasources?.[0]).toMatchObject({ ok: false, code: "datasource-auth-error" });
		const status = await agent.getRefreshStatus();
		expect(status.components.datasources).toBe("degraded");
		const serialized = JSON.stringify(status);
		expect(serialized).not.toContain("xoxb-secret");
		expect(serialized).not.toContain(tmpDir);
	});

	it("wires factory-built skills end-to-end (config -> setup -> refresh -> search)", async () => {
		const vault = join(tmpDir, "vault");
		mkdirSync(join(vault, "notes"), { recursive: true });
		writeFileSync(join(vault, "notes", "decisions.md"), "# Decisions\nWe chose Postgres over MySQL for the core DB.");

		const { skills, unknown } = buildDatasourceSkills(
			{
				obsidian: { instanceId: "vault-1", connector: { vaultPath: vault } },
				slack: false,
			},
			tmpDir,
		);
		expect(unknown).toEqual([]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["obsidian"]);

		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: false,
			datasourceSkills: skills,
			datasourceAccess: { allowedTags: ["obsidian"], allowedScopes: ["/obsidian/**"] },
		});
		await agent.refresh(true, { methods: ["datasources"] });
		const { results } = await agent.searchDatasourceDocuments("postgres core database decision");
		expect(results.length).toBeGreaterThan(0);
		expect(results[0]?.source).toMatch(/^\/obsidian\/vault-1\/chunks\//);
		expect(JSON.stringify(results)).not.toContain(vault);
	});

	it("keeps chunk stores persistent across agent instances (search without re-index)", async () => {
		const first = new ObsidianSkill({
			instanceId: "vault-1",
			workspaceRoot: tmpDir,
			connectorOptions: { vaultPath: join(tmpDir, "vault") },
		});
		mkdirSync(join(tmpDir, "vault"), { recursive: true });
		writeFileSync(join(tmpDir, "vault", "note.md"), "# Note\nRelease scheduled for September.");
		expect(await first.index()).toMatchObject({ ok: true, chunkCount: 1 });

		// New skill instance, same workspace: loads persisted chunks lazily.
		const second = new ObsidianSkill({
			instanceId: "vault-1",
			workspaceRoot: tmpDir,
			connectorOptions: { vaultPath: join(tmpDir, "vault") },
		});
		const [method] = second.retrievalMethods();
		const hits = await method?.retrieve("release September", { topK: 5 });
		expect(hits?.length).toBe(1);
	});

	it("keeps rss dedupe window active through the agent refresh path", async () => {
		const mock = createMockFetch([
			{
				match: "feeds.example.com",
				text: `<?xml version="1.0"?><rss version="2.0"><channel><title>News</title><item><title>Story A</title><guid>a-1</guid><description>Alpha beta gamma.</description></item><item><title>Story A</title><guid>a-1</guid><description>Alpha beta gamma repeat.</description></item></channel></rss>`,
			},
		]);
		const skill = new RssSkill({
			instanceId: "feeds",
			workspaceRoot: tmpDir,
			connectorOptions: { feeds: [{ url: "https://feeds.example.com/a.xml" }], fetchImpl: mock.fetchImpl },
		});
		expect(await skill.index()).toMatchObject({ ok: true, chunkCount: 1 });
	});
});
