import { chmodSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { ManagedCliConfigProvider, ManagedCliLaunchContext } from "../../../cli/managed-cli-config.ts";

export function validateMailcrawlInstanceId(instanceId: string): void {
	if (!/^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u.test(instanceId)) {
		throw new Error("mailcrawl instanceId must be a safe single path segment");
	}
}

export function ensurePrivateMailcrawlDataDir(path: string): void {
	mkdirSync(path, { recursive: true, mode: 0o700 });
	if (process.platform !== "win32") chmodSync(path, 0o700);
}

export function createMailcrawlManagedCliProvider(
	binaryPath?: string,
	binaryPaths: readonly string[] = [],
): ManagedCliConfigProvider {
	const paths = [...new Set([binaryPath, ...binaryPaths].filter((path): path is string => path !== undefined))];
	return {
		tool: "mailcrawl",
		...(paths.length === 0 ? {} : { binaryPaths: paths }),
		managedConfigPath: (context) => {
			validateMailcrawlInstanceId(context.instance);
			return join(context.workspace, ".autorag", "datasources", "mailcrawl", context.instance, "managed.json");
		},
		renderConfig: (config) => JSON.stringify(config),
		materialize: async (context): Promise<ManagedCliLaunchContext> => {
			validateMailcrawlInstanceId(context.instance);
			const dataDir =
				typeof (context.config as Record<string, unknown>).dataDir === "string"
					? (context.config as Record<string, string>).dataDir
					: join(context.workspace, ".autorag", "datasources", "mailcrawl", context.instance, "data");
			ensurePrivateMailcrawlDataDir(dataDir);
			return {
				ownership: context.ownership,
				cwd: context.workspace,
				env: {
					MAILCRAWL_DATA_DIR: dataDir,
				},
				prefixArgs: [],
				configPath: context.configPath,
			};
		},
		inspect: async (context) => ({
			ownership: context.ownership,
			configPath: context.configPath,
			appliedBy: "MAILCRAWL_DATA_DIR",
			missingRequirements: [],
			drift: [],
		}),
	};
}
