import { RetrievalMemory } from "../../memory/memory.ts";
import { resolveConfig } from "../config.ts";
import { renderError, renderMemory } from "../output.ts";
import type { CommandContext } from "./types.ts";

const USAGE = "Usage: autorag memory inspect";

/**
 * `autorag memory inspect` — render a path-opaque snapshot of the retrieval
 * memory (curated results, feedback signals, insights, signal defaults).
 * Model-free. The memory file is resolved from config and loaded read-only;
 * only the rendered schema is emitted, never the storage path.
 */
export async function runMemory(ctx: CommandContext): Promise<number> {
	const subcommand = ctx.positionals[0];
	if (subcommand !== "inspect") {
		ctx.stderr(renderError(new Error(USAGE), { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const memory = new RetrievalMemory({ storagePath: config.memoryPath });
		memory.load();
		ctx.stdout(renderMemory(memory.getSchema(), { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
