import { existsSync, mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	ManagedCliConfigManager,
	type ManagedCliConfigProvider,
	type ManagedCliConfigStatus,
	type ManagedCliLaunchContext,
	ManagedCliRegistry,
} from "../../src/cli/managed-cli-config.ts";
import { createCrawlerManagedCliProvider } from "../../src/datasource/crawler-managed-config.ts";
import { createRcloneManagedCliProvider } from "../../src/datasource/skills/gdrive/rclone-managed-config.ts";
import { createHimalayaManagedCliProvider } from "../../src/datasource/skills/gmail/himalaya-managed-config.ts";
import { createQmdManagedCliProvider } from "../../src/datasource/skills/obsidian/config.ts";
import { createJikjiManagedCliProvider } from "../../src/jikji/managed-config.ts";
import { createMinSyncManagedCliProvider } from "../../src/minsync/managed-config.ts";

let workspace: string;

beforeEach(() => {
	workspace = mkdtempSync(join(tmpdir(), "autorag-managed-cli-"));
});

afterEach(() => {
	rmSync(workspace, { recursive: true, force: true });
});

function provider(overrides: Partial<ManagedCliConfigProvider> = {}): ManagedCliConfigProvider {
	return {
		tool: "fixture",
		aliases: ["fixture-cli"],
		managedConfigPath: (context) =>
			join(context.workspace, ".autorag", "tools", "fixture", "default", "fixture.conf"),
		readConfig: (path) => JSON.parse(readFileSync(path, "utf8")),
		renderConfig: (config, existing) =>
			JSON.stringify({
				...(existing && typeof existing === "object" ? existing : {}),
				...(config && typeof config === "object" ? config : {}),
			}),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: { FIXTURE_CONFIG: context.configPath },
			prefixArgs: ["--config", context.configPath],
			configPath: context.configPath,
			...(overrides.materialize ? await overrides.materialize(context) : {}),
		}),
		inspect: async (context): Promise<ManagedCliConfigStatus> => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "prefix-args",
			missingRequirements: [],
			drift: [],
			...(overrides.inspect ? await overrides.inspect(context) : {}),
		}),
		...overrides,
	};
}

describe("ManagedCliRegistry", () => {
	it("resolves registered binary aliases", () => {
		const registry = new ManagedCliRegistry();
		registry.register(provider());

		expect(registry.resolve("fixture-cli")?.tool).toBe("fixture");
		expect(() => registry.register(provider())).toThrow(/already registered/);
	});

	it("resolves every managed CLI integration and its binary aliases", () => {
		const registry = new ManagedCliRegistry();
		const providers = ["discrawl", "katok", "wacrawl", "telecrawl", "slacrawl", "notcrawl"].map((tool) =>
			createCrawlerManagedCliProvider(tool),
		);
		providers.push(
			createQmdManagedCliProvider(),
			createMinSyncManagedCliProvider(),
			createJikjiManagedCliProvider(),
			createRcloneManagedCliProvider(),
			createHimalayaManagedCliProvider(),
		);
		for (const provider of providers) registry.register(provider);
		for (const tool of [
			"discrawl",
			"katok",
			"wacrawl",
			"telecrawl",
			"slacrawl",
			"notcrawl",
			"qmd",
			"minsync",
			"jikji",
			"rclone",
			"himalaya",
		]) {
			expect(registry.resolve(tool)?.tool).toBe(tool);
		}
	});
});

describe("ManagedCliConfigManager", () => {
	it("materializes atomic workspace-local config and launch context", async () => {
		const registry = new ManagedCliRegistry();
		registry.register(
			provider({
				materialize: async (context) => ({
					ownership: context.ownership,
					cwd: context.workspace,
					env: {},
					prefixArgs: [],
					configPath: context.configPath,
				}),
			}),
		);
		const manager = new ManagedCliConfigManager({ workspace, registry });

		const launch = await manager.materialize("fixture", { settings: { mode: "managed" } });

		expect(launch.ownership).toBe("managed");
		expect(launch.configPath).toContain(join(workspace, ".autorag", "tools", "fixture"));
		expect(existsSync(launch.configPath)).toBe(true);
		expect(JSON.parse(readFileSync(launch.configPath, "utf8"))).toEqual({ settings: { mode: "managed" } });
	});

	it("preserves unowned fields and rejects secret values", async () => {
		const registry = new ManagedCliRegistry();
		registry.register(provider());
		const manager = new ManagedCliConfigManager({ workspace, registry });

		await manager.materialize("fixture", { preserved: true });
		await expect(manager.materialize("fixture", { owned: 1 })).resolves.toBeDefined();
		expect(JSON.parse(readFileSync(join(workspace, ".autorag/tools/fixture/default/fixture.conf"), "utf8"))).toEqual({
			preserved: true,
			owned: 1,
		});
		await expect(manager.materialize("fixture", { token: "secret" })).rejects.toThrow(/secret/i);
	});

	it("fails closed when external ownership points inside managed state", async () => {
		const registry = new ManagedCliRegistry();
		registry.register(provider());
		const manager = new ManagedCliConfigManager({ workspace, registry });

		await expect(
			manager.materialize("fixture", {
				ownership: "external",
				configPath: join(workspace, ".autorag/tools/fixture/default/fixture.conf"),
			}),
		).rejects.toThrow(/ownership|external/i);
	});
});
