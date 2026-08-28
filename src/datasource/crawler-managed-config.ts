import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../cli/managed-cli-config.ts";

/**
 * Configuration transport for crawler CLIs. The crawler command language is
 * deliberately absent; only workspace/archive transport is provided.
 */
export function createCrawlerManagedCliProvider(tool: string, binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool,
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) => join(context.workspace, ".autorag", "datasources", tool, "managed.json"),
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: {},
			prefixArgs:
				context.ownership === "external"
					? ["--config", context.configPath]
					: [
							"--db",
							typeof (context.config as Record<string, unknown>).databasePath === "string"
								? (context.config as Record<string, string>).databasePath
								: join(context.workspace, ".autorag", "datasources", tool, "archive.db"),
							...(typeof (context.config as Record<string, unknown>).sourcePath === "string"
								? ["--source", (context.config as Record<string, string>).sourcePath]
								: []),
						],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "database-prefix-arg",
			missingRequirements: [],
			drift: [],
		}),
	};
}
