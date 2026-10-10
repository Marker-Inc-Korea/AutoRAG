/**
 * Manual QA harness for connector-backed datasource skills
 * (#1300 #1301 #1302 #1303 #1304 #1305 #1311 #1314 #1316).
 *
 * Spins up a local mock of every external API (plus real filesystem
 * fixtures for Obsidian and mail exports), builds all skills through the
 * trusted config factory, registers them on a real AutoRAGAgent, then walks
 * the checklist in docs/manual-qa-datasources.md: setup -> refresh/index ->
 * skill announcement -> load_datasource_skill -> per-connection
 * search_datasource_<id> tools -> optional scope narrowing.
 *
 * Run: npx tsx scripts/manual-qa/run-qa.ts  (or bun scripts/manual-qa/run-qa.ts)
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { singleDatasourceToolName } from "../../src/agent/search-single-datasource-tool.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
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

	// --- 1. Setup: build connector-backed skills from trusted config (factory path) ---
	const { skills, unknown } = buildDatasourceSkills(
		{
			github: { connector: { baseUrl: `${base}/github`, repos: ["qa-org/qa-repo"] } },
			"mail-export": { connector: { paths: [mailDir] } },
			obsidian: { connector: { vaultPath: vault } },
			rss: { connector: { feeds: [{ url: `${base}/rss/feed.xml` }] } },
		},
		tmpRoot,
	);
	check("setup: factory builds all four HTTP/filesystem skills", skills.length === 4 && unknown.length === 0);

	const agent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		datasourceSkills: skills,
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
	const names = ["github", "mail-export", "obsidian", "rss"];
	check(
		"prompt: all configured skills announced",
		names.every((name) => prompt.includes(`datasource-${name}`)),
	);
	check(
		"prompt: no fixture paths or tokens leak",
		!prompt.includes(tmpRoot) && !prompt.includes("127.0.0.1"),
	);
	const loadTool = createLoadDatasourceSkillTool(agent);
	const loaded = await loadTool.execute("qa-load", { name: "datasource-github" });
	check("skill: load_datasource_skill returns instructions", loaded.details.loaded === true);

	// --- 4. Search via each connection's own generated tool ---
	const agentTools = (agent as unknown as { tools: readonly AgentTool[] }).tools;
	const generatedToolNames = agentTools
		.map((tool) => tool.name)
		.filter((name) => name.startsWith("search_datasource_"));
	check(
		"search: every configured connection has its own generated tool and no fan-out tool exists",
		names.every((name) => generatedToolNames.includes(singleDatasourceToolName(name))) &&
		!generatedToolNames.includes("search_datasource_documents"),
		generatedToolNames.join(", "),
	);
	const queries: Record<string, string> = {
		github: "Korean queries tokenized ranking",
		"mail-export": "hiring frozen budget approved",
		obsidian: "mobile app beta October",
		rss: "release incremental indexing",
	};
	for (const [skillName, query] of Object.entries(queries)) {
		const searchTool = agentTools.find((tool) => tool.name === singleDatasourceToolName(skillName));
		if (searchTool === undefined) {
			check(`search: ${skillName} returns scoped hit`, false, `no generated tool for ${skillName}`);
			continue;
		}
		const response = await searchTool.execute(`qa-${skillName}`, { query, topK: 5 });
		const hit = response.details.sources.find((source: string) => source.startsWith(`/${skillName}/`));
		check(`search: ${skillName} returns scoped hit`, hit !== undefined, hit ?? "no hit");
	}

	// --- 5. Optional scope narrowing (query filter only) ---
	const mailExportTool = agentTools.find((tool) => tool.name === singleDatasourceToolName("mail-export"));
	const narrowed = await mailExportTool?.execute("qa-narrow", {
		query: "hiring frozen budget approved",
		scope: "/mail-export/**",
	});
	check(
		"scope: narrowing excludes other skills",
		narrowed !== undefined && narrowed.details.sources.every((source: string) => source.startsWith("/mail-export/")),
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
