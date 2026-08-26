import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../../../cli/managed-cli-config.ts";

/** Rclone configuration is credential-free; credentials stay in operator config. */
export function createRcloneManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "rclone",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) =>
			join(context.workspace, ".autorag", "datasources", "rclone", context.instance, "rclone.conf"),
		readConfig: () => undefined,
		renderConfig: () => "",
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: { RCLONE_CONFIG: context.configPath },
			prefixArgs: [],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "RCLONE_CONFIG",
			missingRequirements: [],
			drift: [],
		}),
	};
}
