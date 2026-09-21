/**
 * Live tenant QA for the credential-gated skills: Slack (#1300),
 * Notion (#1302).
 *
 * Detects which tokens are present in the environment and live-tests each
 * available skill against the REAL service through the full agent path
 * (refresh -> index -> each connection's own search_datasource_<id> tool).
 * Skills without a token are skipped with setup instructions.
 *
 * Tokens (set any subset):
 *  - SLACK_BOT_TOKEN     https://api.slack.com/apps → Create App → OAuth &
 *    Permissions → Bot Token Scopes: channels:read, channels:history,
 *    groups:read, groups:history → Install to Workspace → copy xoxb- token.
 *    Then /invite the bot into at least one channel.
 *  - NOTION_TOKEN        https://www.notion.so/my-integrations → New
 *    integration → copy secret. Then share ≥1 page with the integration
 *    (page ⋯ menu → Connections → your integration).
 *
 * Run:  bun scripts/manual-qa/run-qa-tenant-live.ts ["query1" "query2" ...]
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { singleDatasourceToolName } from "../../src/agent/search-single-datasource-tool.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../../src/datasource/skills/factory.ts";

const queries = process.argv.slice(2);

const available: Record<string, boolean> = {
	slack: Boolean(process.env.SLACK_BOT_TOKEN),
	notion: Boolean(process.env.NOTION_TOKEN),
};
const enabled = Object.entries(available).filter(([, ok]) => ok).map(([name]) => name);
const skipped = Object.entries(available).filter(([, ok]) => !ok).map(([name]) => name);

if (skipped.length > 0) {
	console.log(`Skipped (no token): ${skipped.join(", ")} — see this file's header for setup steps.`);
}
if (enabled.length === 0) {
	console.log("No tenant tokens set; nothing to live-test.");
	process.exit(0);
}
console.log(`Live-testing against real services: ${enabled.join(", ")}\n`);

const ws = mkdtempSync(join(tmpdir(), "tenant-live-qa-"));
const docs = join(ws, "docs");
mkdirSync(docs, { recursive: true });
writeFileSync(join(docs, "placeholder.txt"), "placeholder");

const config: Record<string, { connector: Record<string, unknown> }> = {
	slack: { connector: { tokenEnv: "SLACK_BOT_TOKEN" } },
	notion: { connector: { tokenEnv: "NOTION_TOKEN" } },
};
const datasources = Object.fromEntries(enabled.map((name) => [name, config[name]])) as DatasourcesConfig;
const { skills } = buildDatasourceSkills(datasources, ws);

const agent = new AutoRAGAgent({
	searchPaths: [docs],
	workspacePath: ws,
	minSync: false,
	datasourceSkills: skills,
	datasourceAccess: { allowedTags: enabled, allowedScopes: enabled.map((name) => `/${name}/**`) },
});

console.log("Indexing real tenant data via agent.refresh() ...");
const refresh = await agent.refresh(true, { methods: ["datasources"] });
let failures = 0;
for (const result of refresh.datasources ?? []) {
	if (result.ok) {
		console.log(`  ${result.skill}: OK — ${result.chunkCount} chunk(s) indexed`);
	} else {
		failures += 1;
		console.log(`  ${result.skill}: FAILED — ${result.code}: ${result.message}`);
	}
}

const agentTools = (agent as unknown as { tools: readonly AgentTool[] }).tools;
const effectiveQueries = queries.length > 0 ? queries : ["meeting", "프로젝트", "invoice payment", "일정"];
for (const connection of enabled) {
	const tool = agentTools.find((entry) => entry.name === singleDatasourceToolName(connection));
	if (tool === undefined) {
		failures += 1;
		console.log(`\n${connection}: FAILED — no generated search tool for this connection`);
		continue;
	}
	for (const query of effectiveQueries) {
		const response = await tool.execute(`tenant-${connection}-${query}`, { query, topK: 3 });
		console.log(`\n[${connection}] Q: "${query}" → ${response.details.resultCount} hit(s)`);
		const text = response.content.map((part) => (part.type === "text" ? part.text : "")).join("");
		console.log(text.split("\n").slice(0, 10).join("\n"));
	}
}

if (failures > 0) process.exitCode = 1;
