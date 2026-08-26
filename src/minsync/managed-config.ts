import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../cli/managed-cli-config.ts";

export function createMinSyncManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "minsync",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd:
				typeof (context.config as Record<string, unknown>).workspacePath === "string"
					? (context.config as Record<string, string>).workspacePath
					: context.workspace,
			env: {},
			prefixArgs: [],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "cwd",
			missingRequirements: [],
			drift: [],
		}),
	};
}
