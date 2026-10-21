import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { singleDatasourceToolName } from "../../src/agent/search-single-datasource-tool.ts";
import type { CrawlerSkillClient } from "../../src/datasource/crawler-skill.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { GitHubSkill } from "../../src/datasource/skills/github/index.ts";
import { RssSkill } from "../../src/datasource/skills/rss/index.ts";
import { SlackSkill } from "../../src/datasource/skills/slack/index.ts";
import { installFakeMinSync } from "../fixtures/fake-minsync.ts";
import { createMockFetch } from "../fixtures/mock-fetch.ts";

let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-connector-integration-"));
});

/** The generated per-connection tool the agent registers for `datasourceId`. */
function agentDatasourceTool(agent: AutoRAGAgent, datasourceId: string): AgentTool {
	const tools = (agent as unknown as { tools: readonly AgentTool[] }).tools;
	const name = singleDatasourceToolName(datasourceId);
	const tool = tools.find((entry) => entry.name === name);
	if (tool === undefined) throw new Error(`agent exposes no generated tool ${name}`);
	return tool;
}

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function slackSkill(): SlackSkill {
	return new SlackSkill({
		instanceId: "ws-1",
		client: slackClient(),
	});
}

function slackClient(failing = false): CrawlerSkillClient {
	return {
		sync: async () =>
			failing
				? { ok: false, reason: "binary-missing", stdout: "", stderr: "", code: null }
				: { ok: true, count: 1, stdout: "", stderr: "", code: 0 },
		search: async () => ({
			ok: true,
			hits: [
				{
					id: "C01-1700000001-000100",
					content: "Budget approved for the launch",
					score: 1,
					title: "#general",
					metadata: { channelId: "C01" },
				},
			],
			stdout: "",
			stderr: "",
			code: 0,
		}),
	};
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
	it("indexes during refresh and searches each connection through its own generated tool", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: [slackSkill(), githubSkill()],
		});

		const refresh = await agent.refresh(true, { methods: ["datasources"] });
		expect(refresh.datasources?.every((result) => result.ok)).toBe(true);

		const slackTool = agentDatasourceTool(agent, "slack");
		const slackResponse = await slackTool.execute("call-1", { query: "budget approved launch" });
		expect(slackResponse.details.resultCount).toBeGreaterThan(0);
		expect(slackResponse.details.sources.every((source: string) => source.startsWith("/slack/"))).toBe(true);

		const githubTool = agentDatasourceTool(agent, "github");
		const githubResponse = await githubTool.execute("call-2", { query: "login retry token expires" });
		expect(githubResponse.details.sources.some((source: string) => source.startsWith("/github/acme/chunks/"))).toBe(
			true,
		);
		expect(githubResponse.details.sources.every((source: string) => source.startsWith("/github/"))).toBe(true);

		// Scope narrows within the connection.
		const scoped = await slackTool.execute("call-3", { query: "login retry token expires", scope: "/slack/**" });
		expect(scoped.details.sources.every((source: string) => source.startsWith("/slack/"))).toBe(true);
	});

	it("announces configured connector skills in the system prompt and loads them on demand", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: [slackSkill(), githubSkill()],
		});

		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("datasource-slack");
		expect(prompt).toContain("datasource-github");

		const loadTool = createLoadDatasourceSkillTool(agent);
		const loaded = await loadTool.execute("call-load", { name: "datasource-slack" });
		expect(loaded.details).toEqual({ skill: "datasource-slack", loaded: true });
		const loadedGithub = await loadTool.execute("call-load-github", { name: "datasource-github" });
		expect(loadedGithub.details).toEqual({ skill: "datasource-github", loaded: true });
	});

	it("degrades to path-opaque diagnostics when an external crawler fails during refresh", async () => {
		const failing = new SlackSkill({
			instanceId: "ws-1",
			client: slackClient(true),
		});
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: [failing],
		});

		const refresh = await agent.refresh(true, { methods: ["datasources"] });
		expect(refresh.datasources?.[0]).toMatchObject({ ok: false, code: "datasource-unavailable" });
		const status = await agent.getRefreshStatus();
		expect(status.components.datasources).toBe("degraded");
		const serialized = JSON.stringify(status);
		expect(serialized).not.toContain(tmpDir);
	});

	it("indexes a factory-built Obsidian vault through MinSync and filters a folder through the agent", async () => {
		const vault = join(tmpDir, "vault");
		mkdirSync(join(vault, "Decisions"), { recursive: true });
		mkdirSync(join(vault, "Daily"), { recursive: true });
		writeFileSync(
			join(vault, "Decisions", "database.md"),
			"# Decisions\nWe chose Postgres over MySQL for the core database.",
		);
		writeFileSync(join(vault, "Daily", "2026-10-01.md"), "# Daily\nRead about Postgres vacuum tuning.");
		const fake = installFakeMinSync(join(tmpDir, "bin"));

		const { skills, unknown } = buildDatasourceSkills(
			{ obsidian: { instanceId: "vault-1", connector: { vaultPath: vault } }, slack: false },
			tmpDir,
			{ minSync: { binaryPath: fake.binaryPath, autoInstall: false } },
		);
		expect(unknown).toEqual([]);
		expect(skills.map((skill) => skill.describe().name)).toEqual(["obsidian"]);
		expect(skills[0]?.describe().requiresExternalCli).toBeUndefined();

		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: skills,
		});
		const refresh = await agent.refresh(true, { methods: ["datasources"] });
		expect(refresh.datasources?.find((result) => result.skill === "obsidian")).toMatchObject({
			ok: true,
			chunkCount: 2,
		});

		const everywhere = await agent.searchSingleDatasourceDocuments("obsidian", "postgres");
		expect(everywhere.results.map((result) => result.source).sort()).toEqual([
			"/obsidian/vault-1/Daily/2026-10-01.md",
			"/obsidian/vault-1/Decisions/database.md",
		]);

		const decisions = await agent.searchSingleDatasourceDocuments("obsidian", "postgres", {
			scope: "/obsidian/vault-1/Decisions",
		});
		expect(decisions.results.map((result) => result.source)).toEqual(["/obsidian/vault-1/Decisions/database.md"]);
		expect(decisions.results[0]?.metadata.path).toBe(join(vault, "Decisions", "database.md"));
	});

	it("serves two vaults and an aliased connection from separate MinSync workspaces", async () => {
		const work = join(tmpDir, "work-vault");
		const home = join(tmpDir, "home-vault");
		mkdirSync(join(work, "obsidian"), { recursive: true });
		mkdirSync(home, { recursive: true });
		writeFileSync(join(work, "obsidian", "plan.md"), "# Plan\nquarterly planning notes");
		writeFileSync(join(home, "recipes.md"), "# Recipes\nquarterly sourdough planning");
		const fake = installFakeMinSync(join(tmpDir, "bin"));

		const { skills, unknown } = buildDatasourceSkills(
			{
				"work-notes": { type: "obsidian", connector: { vaultPath: work } },
				"home-notes": { type: "obsidian", connector: { vaultPath: home } },
			},
			tmpDir,
			{ minSync: { binaryPath: fake.binaryPath, autoInstall: false } },
		);
		expect(unknown).toEqual([]);
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: skills,
		});
		await agent.refresh(true, { methods: ["datasources"] });

		const workHits = await agent.searchSingleDatasourceDocuments("work-notes", "quarterly");
		expect(workHits.results.map((result) => result.source)).toEqual(["/work-notes/work-notes/obsidian/plan.md"]);
		const scoped = await agent.searchSingleDatasourceDocuments("work-notes", "quarterly", {
			scope: "/work-notes/work-notes/obsidian",
		});
		expect(scoped.results).toHaveLength(1);
		const homeHits = await agent.searchSingleDatasourceDocuments("home-notes", "quarterly");
		expect(homeHits.results.map((result) => result.source)).toEqual(["/home-notes/home-notes/recipes.md"]);
	});

	it("reports an Obsidian vault with no MinSync binary as degraded without failing the other refresh methods", async () => {
		const vault = join(tmpDir, "vault");
		mkdirSync(vault, { recursive: true });
		writeFileSync(join(vault, "note.md"), "# Note\nsomething");
		const { skills } = buildDatasourceSkills({ obsidian: { connector: { vaultPath: vault } } }, tmpDir, {
			minSync: { binaryPath: join(tmpDir, "missing-minsync"), autoInstall: false },
		});
		const agent = new AutoRAGAgent({
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: tmpDir,
			minSync: { autoInstall: false },
			everything: false,
			fsearch: false,
			datasourceSkills: skills,
		});

		const refresh = await agent.refresh(true, { methods: ["datasources"] });

		expect(refresh.datasources?.[0]).toMatchObject({ ok: false, code: "datasource-unavailable" });
		expect((await agent.getRefreshStatus()).components.datasources).toBe("degraded");
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
