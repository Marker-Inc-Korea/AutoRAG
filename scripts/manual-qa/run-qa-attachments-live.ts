/**
 * Live QA: every connected datasource hands its attachment folders to the agent.
 *
 * Builds the skills from the real `~/.autorag/config.json`, then for each datasource that reports attachment
 * folders: prints them, loads the skill manifest through the real `load_datasource_skill` tool, and runs one
 * real `search_datasource_<id>` call, asserting the folders trail the results. Read-only (no refresh/sync).
 *
 *   bun scripts/manual-qa/run-qa-attachments-live.ts [query]
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createLoadDatasourceSkillTool } from "../../src/agent/datasource-skill.ts";
import { resolveAutoRAGHome, resolveUserHome } from "../../src/config/home.ts";
import { buildDatasourceSkills, type DatasourcesConfig } from "../../src/datasource/skills/factory.ts";
import { createSingleDatasourceSearchTools } from "../../src/agent/search-single-datasource-tool.ts";

const query = process.argv[2] ?? "사진";
const config = JSON.parse(readFileSync(join(resolveAutoRAGHome(), "config.json"), "utf8")) as {
	datasources?: DatasourcesConfig;
};
const workspace = resolveUserHome();
const { skills, unknown } = buildDatasourceSkills(config.datasources, workspace);
console.log(`built ${skills.length} datasource skills; unknown: ${unknown.join(", ") || "none"}`);

const agent = new AutoRAGAgent({
	searchPaths: [],
	workspacePath: join(resolveAutoRAGHome(), "workspace"),
	jikji: false,
	everything: false,
	fsearch: false,
	minSync: { autoInstall: false },
	datasourceSkills: skills,
});
const text = (r: { content: readonly { type: string; text?: string }[] }) => r.content.map((p) => p.text ?? "").join("");
let failures = 0;
const check = (ok: boolean, label: string) => {
	console.log(`${ok ? "PASS" : "FAIL"}  ${label}`);
	if (!ok) failures += 1;
};

for (const skill of skills) {
	const id = skill.describe().datasourceId ?? skill.describe().name;
	const roots = skill.attachmentRoots?.() ?? [];
	console.log(`\n== ${id}: ${roots.length} attachment folder(s)`);
	for (const root of roots) console.log(`   ${root.id}: ${root.path}`);
	if (roots.length === 0) continue;

	const manifest = text(
		await createLoadDatasourceSkillTool(agent).execute("load", { name: skill.skillManifest().name }),
	);
	check(roots.every((root) => manifest.includes(root.path)), `${id}: load_datasource_skill lists every folder`);

	const [tool] = createSingleDatasourceSearchTools(agent, [
		{ datasourceId: id, description: id, instanceScopes: [], attachmentRoots: () => skill.attachmentRoots?.() ?? [] },
	]);
	const out = text(await (tool as NonNullable<typeof tool>).execute("search", { query, topK: 3 }));
	const hit = !out.startsWith("No datasource results");
	console.log(`   search "${query}": ${hit ? "hits" : "no hits"}`);
	if (hit) check(roots.every((root) => out.includes(root.path)), `${id}: search result ends with the folders`);
}
process.exit(failures === 0 ? 0 : 1);
