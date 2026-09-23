/**
 * Live manual QA against real external systems that need no provisioned
 * credentials: the public GitHub REST API (#1303) and a real RSS feed
 * (#1316), flowing through the full agent path (setup -> refresh -> search).
 *
 * Skills requiring tenant credentials (Slack, Discord, Notion, Drive)
 * are covered by run-qa.ts against protocol-accurate mocks, and the
 * filesystem skills (obsidian, mail-export) run on real files there too.
 *
 * Run: bun scripts/manual-qa/run-qa-live.ts
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { singleDatasourceToolName } from "../../src/agent/search-single-datasource-tool.ts";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";

const tmpRoot = mkdtempSync(join(tmpdir(), "autorag-live-qa-"));
const docsDir = join(tmpRoot, "docs");
mkdirSync(docsDir, { recursive: true });
writeFileSync(join(docsDir, "readme.txt"), "placeholder");

let failures = 0;
function check(name: string, pass: boolean, note?: string): void {
	if (!pass) failures += 1;
	console.log(`${pass ? "PASS" : "FAIL"}  ${name}${note ? ` — ${note}` : ""}`);
}

try {
	const { skills } = buildDatasourceSkills(
		{
			github: { connector: { repos: ["Marker-Inc-Korea/AutoRAG"], maxPages: 1, maxDocuments: 100 } },
			rss: { connector: { feeds: [{ url: "https://hnrss.org/frontpage" }], maxItemsPerFeed: 30 } },
		},
		tmpRoot,
	);
	const agent = new AutoRAGAgent({
		searchPaths: [docsDir],
		workspacePath: tmpRoot,
		minSync: false,
		datasourceSkills: skills,
		datasourceAccess: { allowedTags: ["github", "rss"], allowedScopes: ["/github/**", "/rss/**"] },
	});

	const refresh = await agent.refresh(true, { methods: ["datasources"] });
	for (const result of refresh.datasources ?? []) {
		check(
			`live index: ${result.skill}`,
			result.ok,
			result.ok ? `${result.chunkCount} chunk(s)` : `${result.code}: ${result.message}`,
		);
	}

	const agentTools = (agent as unknown as { tools: readonly AgentTool[] }).tools;
	const generatedToolNames = agentTools
		.map((entry) => entry.name)
		.filter((name) => name.startsWith("search_datasource_"));
	check(
		"live tools: every authorized connection has its own generated tool and no fan-out datasource tool",
		generatedToolNames.length === 2 &&
		generatedToolNames.includes(singleDatasourceToolName("github")) &&
		generatedToolNames.includes(singleDatasourceToolName("rss")) &&
		!generatedToolNames.includes("search_datasource_documents"),
		generatedToolNames.join(", "),
	);
	const githubTool = agentTools.find((entry) => entry.name === singleDatasourceToolName("github"));
	const githubHits = await githubTool?.execute("live-gh", {
		query: "datasource skill retrieval",
		topK: 5,
		scope: "/github/**",
	});
	check(
		"live search: github issues return scoped hits",
		(githubHits?.details.sources.length ?? 0) > 0 &&
		(githubHits?.details.sources ?? []).every((source: string) => source.startsWith("/github/")),
		githubHits?.details.sources[0],
	);

	const rssTool = agentTools.find((entry) => entry.name === singleDatasourceToolName("rss"));
	const rssHits = await rssTool?.execute("live-rss", { query: "the a and", topK: 5, scope: "/rss/**" });
	check(
		"live search: rss frontpage returns scoped hits",
		(rssHits?.details.sources.length ?? 0) > 0 &&
		(rssHits?.details.sources ?? []).every((source: string) => source.startsWith("/rss/")),
		rssHits?.details.sources[0],
	);

	console.log(failures === 0 ? "\nLIVE QA PASSED" : `\nLIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}
