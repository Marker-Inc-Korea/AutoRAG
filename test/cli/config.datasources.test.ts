import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { buildAgentOptions, ConfigError, resolveConfig } from "../../src/cli/config.ts";
import type { DatasourceSkill } from "../../src/datasource/types.ts";

let tmpRoot: string;

beforeEach(() => {
	tmpRoot = mkdtempSync(join(tmpdir(), "autorag-config-datasources-"));
});

afterEach(() => {
	rmSync(tmpRoot, { recursive: true, force: true });
});

function writeConfig(config: Record<string, unknown>): string {
	const configPath = join(tmpRoot, "config.json");
	writeFileSync(configPath, JSON.stringify(config));
	return configPath;
}

describe("CLI config datasources wiring", () => {
	it("materializes configured datasource skills and trusted access into agent options", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				rss: { connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] } },
				obsidian: { instanceId: "vault", connector: { vaultPath: join(tmpRoot, "vault") } },
				slack: false,
				github: { enabled: false },
			},
			datasourceAccess: { allowedTags: ["rss", "obsidian"], allowedScopes: ["/rss/**", "/obsidian/**"] },
		});
		const config = resolveConfig({ flags: { config: configPath } });
		expect(config.datasources).toBeDefined();
		expect(config.datasourceAccess).toEqual({
			allowedTags: ["rss", "obsidian"],
			allowedScopes: ["/rss/**", "/obsidian/**"],
		});

		const options = buildAgentOptions(config);
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];
		expect(skills.map((skill) => skill.describe().name).sort()).toEqual(["obsidian", "rss"]);
		expect(skills.find((skill) => skill.describe().name === "obsidian")?.describe().instanceId).toBe("vault");
		expect(options.datasourceAccess).toEqual(config.datasourceAccess);
	});

	it("stays default-deny when datasourceAccess is omitted", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: { rss: { connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] } } },
		});
		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		expect(options.datasourceSkills).toHaveLength(1);
		expect(options.datasourceAccess).toBeUndefined();
	});

	it("skips unknown datasource skill names and keeps known skills", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				rss: { connector: { feeds: [{ url: "https://feeds.example.com/a.xml" }] } },
				"discord-nomadamas": {},
				"slack-local": {},
			},
		});
		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];
		expect(skills.map((skill) => skill.describe().name)).toEqual(["rss"]);
		expect(options.startupDiagnostics).toEqual([
			{
				code: "unknown-datasource-skill",
				severity: "warning",
				message: "Unknown datasource skill(s) in config were skipped: discord-nomadamas, slack-local",
				source: "datasources",
			},
		]);
	});

	it("materializes WhatsApp through the external wacrawl backend", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				whatsapp: { instanceId: "personal", connector: { binaryPath: "/opt/bin/wacrawl" } },
			},
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills).toHaveLength(1);
		expect(skills[0]?.describe()).toMatchObject({
			name: "whatsapp",
			type: "whatsapp-archive",
			instanceId: "personal",
			requiresExternalCli: true,
		});
	});

	it("materializes Telegram through the external telecrawl backend", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				telegram: { instanceId: "personal", connector: { binaryPath: "/opt/bin/telecrawl" } },
			},
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills).toHaveLength(1);
		expect(skills[0]?.describe()).toMatchObject({
			name: "telegram",
			type: "telegram-archive",
			instanceId: "personal",
			requiresExternalCli: true,
		});
	});

	it("materializes ClawGallery through the external CLI backend", () => {
		const configPath = writeConfig({
			datasources: {
				screenshots: {
					type: "clawgallery",
					instanceId: "personal",
					connector: { binaryPath: "/tmp/clawgallery", syncVisual: true, vdrBackend: "vsplade" },
				},
			},
		});
		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		expect(options.datasourceSkills).toHaveLength(1);
		expect(options.datasourceSkills?.[0]?.describe()).toMatchObject({
			id: "screenshots",
			type: "clawgallery",
			instanceId: "personal",
		});
	});

	it("materializes Slack through the external slacrawl backend", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				slack: { instanceId: "workspace", connector: { binaryPath: "/opt/bin/slacrawl", syncSource: "primary" } },
			},
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills).toHaveLength(1);
		expect(skills[0]?.describe()).toMatchObject({
			name: "slack",
			type: "slack-archive",
			instanceId: "workspace",
			requiresExternalCli: true,
		});
	});

	it("materializes Notion through the external notcrawl backend", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				notion: { instanceId: "workspace", connector: { binaryPath: "/opt/bin/notcrawl" } },
			},
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills).toHaveLength(1);
		expect(skills[0]?.describe()).toMatchObject({
			name: "notion",
			type: "notion-archive",
			instanceId: "workspace",
			requiresExternalCli: true,
		});
	});

	it("materializes provider-neutral cloud drives through the rclone CLI datasource", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				"cloud-drive": {
					instanceId: "team-drive",
					connector: {
						backend: "rclone",
						provider: "onedrive",
						remote: "onedrive:Team Docs",
						include: ["**/*.pdf", "**/*.md"],
						exclude: ["Archive/**"],
						concurrency: 4,
						bandwidthLimit: "10M",
					},
				},
			},
			datasourceAccess: { allowedTags: ["cloud-drive"], allowedScopes: ["/cloud-drive/**"] },
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills).toHaveLength(1);
		expect(skills[0]?.describe()).toMatchObject({
			name: "cloud-drive",
			type: "rclone-drive",
			instanceId: "team-drive",
			requiresExternalCli: true,
		});
		expect(skills[0]?.skillManifest().content).toContain("OneDrive");
	});

	it("registers each rclone connection alias as a distinct datasource skill", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				"personal-google-drive": {
					type: "cloud-drive",
					instanceId: "personal",
					connector: { provider: "google-drive", remote: "personal-gdrive:" },
				},
				"company-onedrive": {
					type: "cloud-drive",
					instanceId: "work",
					connector: { provider: "onedrive", remote: "company-onedrive:Documents" },
				},
			},
			datasourceAccess: {
				allowedTags: ["cloud-drive"],
				allowedScopes: ["/personal-google-drive/**", "/company-onedrive/**"],
			},
		});

		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills.map((skill) => skill.describe().name).sort()).toEqual([
			"company-onedrive",
			"personal-google-drive",
		]);
		expect(skills.map((skill) => skill.skillManifest().name).sort()).toEqual([
			"datasource-company-onedrive",
			"datasource-personal-google-drive",
		]);
		expect(skills[0]?.describe().datasourceId).not.toBe(skills[1]?.describe().datasourceId);
	});

	it("includes operator-authored datasource descriptions in the agent skill", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				"personal-drive": {
					type: "cloud-drive",
					description: "보통 프로젝트 문서와 계약서 검색에 사용하는 개인 Google Drive입니다.",
					connector: { provider: "google-drive", remote: "personal:" },
				},
			},
		});
		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skill = (options.datasourceSkills ?? [])[0];
		expect(skill?.describe().description).toContain("개인 Google Drive");
		expect(skill?.skillManifest().description).toContain("개인 Google Drive");
		expect(skill?.skillManifest().content).toContain("계약서 검색");
	});

	it("registers connector and crawler aliases through their datasource type", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: {
				"personal-mail": {
					type: "gmail",
					connector: { backend: "himalaya", account: "personal" },
				},
				"company-github": {
					type: "github",
					connector: { repos: ["acme/docs"] },
				},
				"engineering-slack": {
					type: "slack",
					connector: { binaryPath: "/missing/slacrawl" },
				},
				"family-kakao": {
					type: "kakao",
					channels: { names: ["가족방"] },
					connector: { binaryPath: "/missing/katok" },
				},
			},
			datasourceAccess: {
				allowedTags: ["gmail", "github", "slack", "kakaotalk"],
				allowedScopes: ["/personal-mail/**", "/company-github/**", "/engineering-slack/**", "/family-kakao/**"],
			},
		});
		const options = buildAgentOptions(resolveConfig({ flags: { config: configPath } }));
		const skills = (options.datasourceSkills ?? []) as readonly DatasourceSkill[];

		expect(skills.map((skill) => skill.describe().name).sort()).toEqual([
			"company-github",
			"engineering-slack",
			"family-kakao",
			"personal-mail",
		]);
		for (const skill of skills) {
			expect(skill.skillManifest().name).toBe(`datasource-${skill.describe().name}`);
			expect(skill.describeSources()[0]?.source).toContain(`/${skill.describe().name}/`);
		}
		expect(skills.find((skill) => skill.describe().name === "family-kakao")?.skillManifest().content).toContain(
			"가족방",
		);
	});

	it("rejects malformed datasources and datasourceAccess sections", () => {
		const badDatasources = writeConfig({ searchPaths: [tmpRoot], datasources: ["rss"] });
		expect(() => resolveConfig({ flags: { config: badDatasources } })).toThrow(ConfigError);
		const badAccess = join(tmpRoot, "config2.json");
		writeFileSync(badAccess, JSON.stringify({ searchPaths: [tmpRoot], datasourceAccess: "all" }));
		expect(() => resolveConfig({ flags: { config: badAccess } })).toThrow(ConfigError);
	});
});
