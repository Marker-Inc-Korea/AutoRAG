/**
 * Live manual QA against real external systems that need no provisioned
 * credentials: the public GitHub REST API (#1303) and a real RSS feed
 * (#1316), flowing through the full agent path (setup -> refresh -> search).
 *
 * Skills requiring tenant credentials (Slack, Discord, Notion, Drive,
 * Gmail) are covered by run-qa.ts against protocol-accurate mocks, and the
 * filesystem skills (obsidian, mail-export) run on real files there too.
 *
 * Run: bun scripts/manual-qa/run-qa-live.ts
 */

import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createSearchDatasourceDocumentsTool } from "../../src/agent/search-datasource-tool.ts";
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
		bm25: false,
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

	const tool = createSearchDatasourceDocumentsTool(agent);
	const githubHits = await tool.execute("live-gh", { query: "datasource skill retrieval", topK: 5, scope: "/github/**" });
	check(
		"live search: github issues return scoped hits",
		githubHits.details.sources.length > 0 &&
			githubHits.details.sources.every((source) => source.startsWith("/github/")),
		githubHits.details.sources[0],
	);

	const rssHits = await tool.execute("live-rss", { query: "the a and", topK: 5, scope: "/rss/**" });
	check(
		"live search: rss frontpage returns scoped hits",
		rssHits.details.sources.length > 0 && rssHits.details.sources.every((source) => source.startsWith("/rss/")),
		rssHits.details.sources[0],
	);

	console.log(failures === 0 ? "\nLIVE QA PASSED" : `\nLIVE QA: ${failures} failure(s)`);
	if (failures > 0) process.exitCode = 1;
} finally {
	rmSync(tmpRoot, { recursive: true, force: true });
}
