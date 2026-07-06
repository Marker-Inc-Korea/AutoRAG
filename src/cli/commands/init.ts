import { join } from "node:path";
import { type CliConfig, ConfigError, DEFAULT_CONFIG_FILENAME, writeDefaultConfig } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag init` — write a default `autorag.config.json` into the current
 * working directory. Flag values (search-paths csv, workspace, memory-path,
 * model-provider/model-id) are folded into the generated config; omitted flags
 * fall back to the built-in defaults. Refuses to clobber an existing file
 * unless `--force` is set.
 */
export async function runInit(ctx: CommandContext): Promise<number> {
	const flags = ctx.flags;
	const partial: Partial<CliConfig> = {};

	const searchPathsFlag = typeof flags["search-paths"] === "string" ? flags["search-paths"] : undefined;
	if (searchPathsFlag) {
		partial.searchPaths = searchPathsFlag
			.split(",")
			.map((entry) => entry.trim())
			.filter((entry) => entry.length > 0);
	}
	if (typeof flags.workspace === "string") partial.workspacePath = flags.workspace;
	if (typeof flags["memory-path"] === "string") partial.memoryPath = flags["memory-path"];

	const provider = typeof flags["model-provider"] === "string" ? flags["model-provider"] : undefined;
	const id = typeof flags["model-id"] === "string" ? flags["model-id"] : undefined;
	if (provider && id) partial.model = { provider, id };

	try {
		writeDefaultConfig(join(ctx.cwd, DEFAULT_CONFIG_FILENAME), partial, { force: flags.force === true });
	} catch (error) {
		if (error instanceof ConfigError) {
			ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
			return 2;
		}
		throw error;
	}

	const envelope = { ok: true, wrote: [DEFAULT_CONFIG_FILENAME] };
	if (ctx.json) {
		ctx.stdout(JSON.stringify(envelope));
	} else {
		ctx.stdout(`Wrote ${DEFAULT_CONFIG_FILENAME}.`);
	}
	return 0;
}
