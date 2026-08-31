import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../../../cli/managed-cli-config.ts";

export function createMailcrawlManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "mailcrawl",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) => join(context.workspace, ".autorag", "datasources", "mailcrawl", context.instance, "managed.json"),
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: {
				MAILCRAWL_DATA_DIR:
					typeof (context.config as Record<string, unknown>).dataDir === "string"
						? (context.config as Record<string, string>).dataDir
						: join(context.workspace, ".autorag", "datasources", "mailcrawl", context.instance, "data"),
			},
			prefixArgs: [],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "MAILCRAWL_DATA_DIR",
			missingRequirements: [],
			drift: [],
		}),
	};
}

