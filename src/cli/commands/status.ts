import { AutoRAGAgent } from "../../agent/agent.ts";
import { buildAgentOptions, resolveConfig } from "../config.ts";
import { renderError, renderStatus } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag status` — path-opaque snapshot of corpus freshness and index health.
 * Runs a cheap parse-free staleness scan plus the cached last-refresh outcome.
 * Model-free. Never emits filesystem paths; only the renderered status text.
 */
export async function runStatus(ctx: CommandContext): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		const status = await agent.getRefreshStatus();
		ctx.stdout(renderStatus(status, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
