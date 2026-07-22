/**
 * Live tenant QA for the credential-gated skills: Slack (#1300),
 * Notion (#1302), Google Drive (#1301), Gmail (#1304).
 *
 * Detects which tokens are present in the environment and live-tests each
 * available skill against the REAL service through the full agent path
 * (refresh -> index -> search_datasource_documents). Skills without a token
 * are skipped with setup instructions.
 *
 * Tokens (set any subset):
 *  - SLACK_BOT_TOKEN     https://api.slack.com/apps → Create App → OAuth &
 *    Permissions → Bot Token Scopes: channels:read, channels:history,
 *    groups:read, groups:history → Install to Workspace → copy xoxb- token.
 *    Then /invite the bot into at least one channel.
 *  - NOTION_TOKEN        https://www.notion.so/my-integrations → New
 *    integration → copy secret. Then share ≥1 page with the integration
 *    (page ⋯ menu → Connections → your integration).
 *  - GDRIVE_ACCESS_TOKEN OAuth2 access token with drive.readonly scope
 *    (e.g. https://developers.google.com/oauthplayground → Drive API v3 →
 *    authorize → copy access token; expires ~1h).
 *  - GMAIL_ACCESS_TOKEN  Same playground flow with gmail.readonly scope.
 *
 * Run:  bun scripts/manual-qa/run-qa-tenant-live.ts ["query1" "query2" ...]
 */

import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../../src/datasource/skills/factory.ts";

const queries = process.argv.slice(2);

const available: Record<string, boolean> = {
	slack: Boolean(process.env.SLACK_BOT_TOKEN),
	notion: Boolean(process.env.NOTION_TOKEN),
	gdrive: Boolean(process.env.GDRIVE_ACCESS_TOKEN),
	gmail: Boolean(process.env.GMAIL_ACCESS_TOKEN),
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
	gdrive: { connector: { tokenEnv: "GDRIVE_ACCESS_TOKEN" } },
	gmail: { connector: { tokenEnv: "GMAIL_ACCESS_TOKEN", labelIds: ["INBOX"] } },
};
const datasources = Object.fromEntries(enabled.map((name) => [name, config[name]])) as DatasourcesConfig;
const { skills } = buildDatasourceSkills(datasources, ws);

const agent = new AutoRAGAgent({
	searchPaths: [docs],
	workspacePath: ws,
	minSync: false,
	bm25: false,
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

const tool = createSearchDatasourceDocumentsTool(agent);
const effectiveQueries = queries.length > 0 ? queries : ["meeting", "프로젝트", "invoice payment", "일정"];
for (const query of effectiveQueries) {
	const response = await tool.execute(`tenant-${query}`, { query, topK: 3 });
	console.log(`\nQ: "${query}" → ${response.details.resultCount} hit(s)`);
	const text = response.content.map((part) => (part.type === "text" ? part.text : "")).join("");
	console.log(text.split("\n").slice(0, 10).join("\n"));
}

if (failures > 0) process.exitCode = 1;
