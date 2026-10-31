import { describe, expect, it } from "vitest";
import { buildDatasourceSkills } from "../../src/datasource/skills/factory.ts";
import { datasourceSearchToolName } from "../../src/datasource/tool-naming.ts";
import { withPlatform } from "../helpers/platform.ts";

/**
 * Manifest contract for the per-datasource search tools: every built-in skill
 * manifest must name its dedicated `search_datasource_<id>` tool (a tool the
 * agent actually registers), never reference a tool that no longer exists, and
 * carry a Native CLI section when it is backed by an external CLI. Aliased
 * connections must name the alias's tool, not the template's.
 */
describe("datasource skill manifests and dedicated search tools", () => {
	const { skills, unknown } = withPlatform("darwin", () =>
		buildDatasourceSkills({
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
			"cloud-drive": true,
			lark: true,
		}),
	);

	it("builds every requested built-in skill", () => {
		expect(unknown).toEqual([]);
		expect(skills.length).toBe(13);
	});

	it("every manifest names its dedicated tool and references no removed tool", () => {
		for (const skill of skills) {
			const datasourceId = skill.describe().datasourceId;
			expect(datasourceId).toBeDefined();
			const manifest = skill.skillManifest();
			expect(manifest.content, `${datasourceId} names its tool`).toContain(datasourceSearchToolName(datasourceId!));
			// The removed fan-out datasource tool must not survive anywhere: a manifest
			// that steers the model at a tool the agent never registers is a defect.
			expect(manifest.content, `${datasourceId} references no removed tool`).not.toContain(
				"search_datasource_documents",
			);
		}
	});

	it("a CLI-backed manifest documents its native CLI, and a MinSync-backed vault claims none", () => {
		for (const skill of skills) {
			const descriptor = skill.describe();
			const content = skill.skillManifest().content;
			if (descriptor.requiresExternalCli === true) {
				expect(content, `${descriptor.datasourceId} has a Native CLI section`).toContain("## Native CLI");
			}
		}
		const obsidian = skills.find((skill) => skill.describe().datasourceId === "obsidian");
		expect(obsidian?.describe().requiresExternalCli).toBeUndefined();
		expect(obsidian?.skillManifest().content).not.toContain("## Native CLI");
	});

	it("an aliased connection's manifest names the alias's tool, never the template's", () => {
		const { skills: aliasedSkills, unknown: aliasedUnknown } = withPlatform("darwin", () =>
			buildDatasourceSkills({ "kakao-work": { type: "kakao" } }),
		);
		expect(aliasedUnknown).toEqual([]);
		const skill = aliasedSkills[0];
		expect(skill).toBeDefined();
		const content = skill!.skillManifest().content;
		expect(content).toContain("search_datasource_kakao_work");
		// No other per-datasource tool reference may survive the rewrite.
		expect(content).not.toMatch(/search_datasource_kakao(?![a-z0-9_])/);
	});
});
