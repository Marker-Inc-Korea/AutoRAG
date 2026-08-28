import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../../../cli/managed-cli-config.ts";

/** Provider for qmd's environment-based config/cache transport. */
export function createQmdManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "qmd",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) =>
			join(context.workspace, ".autorag", "datasources", "obsidian", context.instance, "managed.json"),
		readConfig: () => undefined,
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => {
			const configDir =
				context.ownership === "external"
					? context.configPath
					: join(context.workspace, ".autorag", "datasources", "obsidian", context.instance, "config");
			const cacheDir = join(context.workspace, ".autorag", "datasources", "obsidian", context.instance, "cache");
			return {
				ownership: context.ownership,
				cwd: context.workspace,
				env: { QMD_CONFIG_DIR: configDir, XDG_CACHE_HOME: cacheDir, QMD_TRUST_LOCAL_CONFIG: "1" },
				prefixArgs: [],
				configPath: context.configPath,
			};
		},
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "environment",
			missingRequirements: [],
			drift: [],
		}),
	};
}
