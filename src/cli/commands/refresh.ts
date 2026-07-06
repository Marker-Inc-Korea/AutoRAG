import { AutoRAGAgent } from "../../agent/agent.ts";
import { buildAgentOptions, resolveConfig } from "../config.ts";
import { renderError, renderRefresh } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag refresh` — parse configured search paths and resync every active
 * index (parsed mirror, BM25, MinSync, datasources, jikji). Model-free: no LLM
 * is constructed. Output is rendered through the path-opaque refresh renderer;
 * the raw result (which carries absolute `indexPath`) is never printed.
 */
export async function runRefresh(ctx: CommandContext): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		const result = await agent.refresh(ctx.flags.force === true);
		ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
