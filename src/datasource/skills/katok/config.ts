import { join } from "node:path";
import type {
	ManagedCliConfigProvider,
	ManagedCliContext,
	ManagedCliLaunchContext,
} from "../../../cli/managed-cli-config.ts";

/**
 * Katok uses a workspace flag rather than a config-file flag. The provider
 * therefore owns only the managed workspace transport; native commands and
 * their arguments remain entirely caller-controlled.
 */
export function createKatokManagedCliProvider(binaryPath?: string): ManagedCliConfigProvider {
	return {
		tool: "katok",
		...(binaryPath === undefined ? {} : { binaryPaths: [binaryPath] }),
		managedConfigPath: (context) => join(context.workspace, ".autorag", "datasources", "katok", "managed.json"),
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => ({
			ownership: context.ownership,
			cwd: context.workspace,
			env: {},
			prefixArgs: ["--workspace", context.ownership === "external" ? context.configPath : join(context.workspace, ".autorag", "datasources", "katok")],
			configPath: context.configPath,
		}),
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "workspace-prefix-arg",
			missingRequirements: [],
			drift: [],
		}),
	};
}
