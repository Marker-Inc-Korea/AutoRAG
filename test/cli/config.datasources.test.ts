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

	it("rejects unknown datasource skill names with a ConfigError", () => {
		const configPath = writeConfig({
			searchPaths: [tmpRoot],
			workspacePath: tmpRoot,
			datasources: { dropbox: {} },
		});
		expect(() => buildAgentOptions(resolveConfig({ flags: { config: configPath } }))).toThrow(ConfigError);
	});

	it("rejects malformed datasources and datasourceAccess sections", () => {
		const badDatasources = writeConfig({ searchPaths: [tmpRoot], datasources: ["rss"] });
		expect(() => resolveConfig({ flags: { config: badDatasources } })).toThrow(ConfigError);
		const badAccess = join(tmpRoot, "config2.json");
		writeFileSync(badAccess, JSON.stringify({ searchPaths: [tmpRoot], datasourceAccess: "all" }));
		expect(() => resolveConfig({ flags: { config: badAccess } })).toThrow(ConfigError);
	});
});
