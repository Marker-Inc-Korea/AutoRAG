/**
 * Manual QA harness for the nine connector-backed datasource skills
 * (#1300 #1301 #1302 #1303 #1304 #1305 #1311 #1314 #1316).
 *
 * Spins up a local mock of every external API (plus real filesystem
 * fixtures for Obsidian and mail exports), builds all skills through the
 * trusted config factory, registers them on a real AutoRAGAgent, then walks
 * the checklist in docs/manual-qa-datasources.md: setup -> refresh/index ->
 * skill announcement -> load_datasource_skill -> search_datasource_documents
 * -> scope narrowing -> default-deny -> diagnostic opacity.
 *
 * Run: npx tsx scripts/manual-qa/run-qa.ts  (or bun scripts/manual-qa/run-qa.ts)
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
// @ts-expect-error plain mjs helper without type declarations
import { startMockServices } from "./mock-services.mjs";

interface CheckResult {
	name: string;
	pass: boolean;
	note?: string;
}

const results: CheckResult[] = [];
function check(name: string, pass: boolean, note?: string): void {
	results.push({ name, pass, note });
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-manual-qa-"));
const { server, port } = (await startMockServices()) as { server: import("node:http").Server; port: number };
const base = `http://127.0.0.1:${port}`;

try {
	// --- filesystem fixtures (obsidian vault + mail exports) ---
	const vault = join(tmpRoot, "vault");
	mkdirSync(join(vault, "projects"), { recursive: true });
	writeFileSync(
		join(vault, "projects", "roadmap.md"),
		"---\ntags: [planning]\n---\n# Roadmap\nThe mobile app beta launches in October.",
	);
	const mailDir = join(tmpRoot, "mail");
	mkdirSync(mailDir, { recursive: true });
	writeFileSync(
		join(mailDir, "budget.eml"),
		[
			"From: cfo@example.com",
			"To: leads@example.com",
			"Subject: FY25 budget freeze",
			"Date: Mon, 03 Jun 2024 10:00:00 +0000",
			"",
			"Hiring is frozen until FY25 budgets are approved.",
		].join("\r\n"),
	);
	const docsDir = join(tmpRoot, "docs");
	mkdirSync(docsDir, { recursive: true });
	writeFileSync(join(docsDir, "readme.txt"), "Local corpus placeholder.");

	// --- 1. Setup: build all nine skills from trusted config (factory path) ---
	const { skills, unknown } = buildDatasourceSkills(
		{
			slack: { connector: { baseUrl: `${base}/slack`, token: "qa-slack-token" } },
			discord: { connector: { baseUrl: `${base}/discord`, token: "qa-discord-token", guildId: "qa-guild" } },
			notion: { connector: { baseUrl: `${base}/notion`, token: "qa-notion-token" } },
			github: { connector: { baseUrl: `${base}/github`, repos: ["qa-org/qa-repo"] } },
			gdrive: { connector: { baseUrl: `${base}/gdrive`, token: "qa-gdrive-token" } },
			gmail: { connector: { baseUrl: `${base}/gmail`, token: "qa-gmail-token" } },
			"mail-export": { connector: { paths: [mailDir] } },
			obsidian: { connector: { vaultPath: vault } },
			rss: { connector: { feeds: [{ url: `${base}/rss/feed.xml` }] } },
		},
		tmpRoot,
	);
	check("setup: factory builds all nine skills", skills.length === 9 && unknown.length === 0);

	const agent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		bm25: false,
		datasourceSkills: skills,
		datasourceAccess: {
			allowedTags: ["slack", "discord", "notion", "github", "gdrive", "gmail", "mail-export", "obsidian", "rss"],
			allowedScopes: ["/**"],
		},
	});

	// --- 2. Indexing through agent refresh ---
	const refresh = await agent.refresh(true, { methods: ["datasources"] });
	const indexed = refresh.datasources ?? [];
	for (const result of indexed) {
		check(
			`index: ${result.skill} refresh`,
			result.ok,
			result.ok ? `${result.chunkCount} chunk(s)` : `${result.code}: ${result.message}`,
		);
	}

	// --- 3. Progressive disclosure: skills announced + loadable ---
	const prompt = agent.getSystemPrompt();
	const names = ["slack", "discord", "notion", "github", "gdrive", "gmail", "mail-export", "obsidian", "rss"];
	check(
		"prompt: all authorized skills announced",
		names.every((name) => prompt.includes(`datasource-${name}`)),
	);
	check(
		"prompt: no fixture paths or tokens leak",
		!prompt.includes(tmpRoot) && !prompt.includes("qa-slack-token") && !prompt.includes("127.0.0.1"),
	);
	const loadTool = createLoadDatasourceSkillTool(agent);
	const loaded = await loadTool.execute("qa-load", { name: "datasource-slack" });
	check("skill: load_datasource_skill returns instructions", loaded.details.loaded === true);

	// --- 4. Search via the agent tool per skill ---
	const searchTool = createSearchDatasourceDocumentsTool(agent);
	const queries: Record<string, string> = {
		slack: "payment gateway migration July",
		discord: "community meetup Seoul August",
		notion: "on-call engineer restart cluster",
		github: "Korean queries tokenized ranking",
		gdrive: "vendor contract cancellation notice",
		gmail: "Gangnam office move September",
		"mail-export": "hiring frozen budget approved",
		obsidian: "mobile app beta October",
		rss: "release incremental indexing",
	};
	for (const [skillName, query] of Object.entries(queries)) {
		const response = await searchTool.execute(`qa-${skillName}`, { query, topK: 5 });
		const hit = response.details.sources.find((source) => source.startsWith(`/${skillName}/`));
		check(`search: ${skillName} returns scoped hit`, hit !== undefined, hit ?? "no hit");
	}

	// --- 5. Scope narrowing (tool arg can only narrow) ---
	const narrowed = await searchTool.execute("qa-narrow", {
		query: "payment gateway migration July",
		scope: "/gmail/**",
	});
	check(
		"scope: narrowing excludes other skills",
		narrowed.details.sources.every((source) => source.startsWith("/gmail/")),
	);

	// --- 6. Default-deny agent (no trusted access) ---
	const denied = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		bm25: false,
		datasourceSkills: buildDatasourceSkills(
			{ rss: { connector: { feeds: [{ url: `${base}/rss/feed.xml` }] } } },
			tmpRoot,
		).skills,
	});
	await denied.refresh(true, { methods: ["datasources"] });
	const deniedSearch = await denied.searchDatasourceDocuments("release incremental indexing");
	check("security: default-deny returns no results", deniedSearch.results.length === 0);
	check("security: denied prompt hides skills", !denied.getSystemPrompt().includes("datasource-rss"));

	// --- 7. Auth failure diagnostics stay opaque ---
	const badAuth = buildDatasourceSkills(
		{ slack: { connector: { baseUrl: `${base}/slack`, token: "wrong-token" } } },
		tmpRoot,
	).skills;
	const badAgent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		bm25: false,
		datasourceSkills: badAuth,
		datasourceAccess: { allowedTags: ["slack"], allowedScopes: ["/slack/**"] },
	});
	const badRefresh = await badAgent.refresh(true, { methods: ["datasources"] });
	const failure = badRefresh.datasources?.[0];
	const serializedFailure = JSON.stringify(failure);
	check("diagnostics: wrong token maps to auth error", failure?.ok === false && failure.code === "datasource-auth-error");
	check(
		"diagnostics: failure payload is path/PII-opaque",
		!serializedFailure.includes("wrong-token") &&
			!serializedFailure.includes(tmpRoot) &&
			!serializedFailure.includes("127.0.0.1"),
	);

	// --- summary ---
	const failed = results.filter((result) => !result.pass);
	console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
	if (failed.length > 0) {
		console.log("FAILED CHECKS:");
		for (const result of failed) console.log(` - ${result.name}${result.note ? ` (${result.note})` : ""}`);
		process.exitCode = 1;
	}
} finally {
	server.close();
	rmSync(tmpRoot, { recursive: true, force: true });
}
