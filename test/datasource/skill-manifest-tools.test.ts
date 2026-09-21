import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { datasourceSearchToolName } from "../../src/datasource/tool-naming.ts";

/**
 * Manifest contract for the per-datasource search tools: every built-in skill
 * manifest must name its dedicated `search_datasource_<id>` tool (a tool the
 * agent actually registers), carry a Native CLI section, and never reference a
 * tool that no longer exists. Aliased connections must name the alias's tool,
 * not the template's.
 */
describe("datasource skill manifests and dedicated search tools", () => {
	const { skills, unknown } = buildDatasourceSkills({
		kakao: true,
		discord: true,
		slack: true,
		notion: true,
		whatsapp: true,
		telegram: true,
		mailcrawl: true,
		obsidian: true,
		clawgallery: true,
		github: true,
		rss: true,
		spotlight: true,
		"cloud-drive": true,
		"mail-export": true,
	});

	it("builds every requested built-in skill", () => {
		expect(unknown).toEqual([]);
		expect(skills.length).toBe(14);
	});

	it("every manifest names its dedicated tool, references no removed tool, and documents the native CLI", () => {
		for (const skill of skills) {
			const descriptor = skill.describe();
			const datasourceId = descriptor.datasourceId;
			expect(datasourceId).toBeDefined();
			const manifest = skill.skillManifest();
			const toolName = datasourceSearchToolName(datasourceId!);
			expect(manifest.content, `${datasourceId} names its tool`).toContain(toolName);
			// The removed fan-out datasource tool must not survive anywhere: a manifest
			// that steers the model at a tool the agent never registers is a defect.
			expect(manifest.content, `${datasourceId} references no removed tool`).not.toContain(
				"search_datasource_documents",
			);
			expect(manifest.content, `${datasourceId} has a Native CLI section`).toContain("## Native CLI");
		}
	});

	it("an aliased connection's manifest names the alias's tool, never the template's", () => {
		const { skills: aliasedSkills, unknown: aliasedUnknown } = buildDatasourceSkills({
			"kakao-work": { type: "kakao" },
		});
		expect(aliasedUnknown).toEqual([]);
		const skill = aliasedSkills[0];
		expect(skill).toBeDefined();
		const content = skill!.skillManifest().content;
		expect(content).toContain("search_datasource_kakao_work");
		// No other per-datasource tool reference may survive the rewrite.
		expect(content).not.toMatch(/search_datasource_kakao(?![a-z0-9_])/);
	});
});
