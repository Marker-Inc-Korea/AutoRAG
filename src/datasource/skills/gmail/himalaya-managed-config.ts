import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../../../cli/managed-cli-config.ts";

/** Himalaya config contains operator credentials; AutoRAG only transports its path. */
export function createHimalayaManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "himalaya",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) =>
			join(context.workspace, ".autorag", "datasources", "himalaya", context.instance, "config.toml"),
		renderConfig: () => "",
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: { HIMALAYA_CONFIG: context.configPath },
			prefixArgs: [],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "HIMALAYA_CONFIG",
			missingRequirements: [],
			drift: [],
		}),
	};
}
