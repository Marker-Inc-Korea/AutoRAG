import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../cli/managed-cli-config.ts";

/** Jikji provider owns only process transport; native prepare/find stay opaque. */
export function createJikjiManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "jikji",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
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
