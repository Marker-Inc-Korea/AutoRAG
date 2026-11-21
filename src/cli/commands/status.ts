import { AutoRAGAgent } from "../../agent/agent.ts";
import { buildAgentOptions, ConfigError, resolveConfig } from "../config.ts";
import { renderError, renderStatus } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag status` — snapshot of corpus freshness and index health.
 * Runs a parse-free staleness scan and inspects existing Jikji indexes without preparing them.
 * Model-free. Native doctor errors are preserved verbatim, including paths when present.
 *
 * Exit codes: 0 on success, 2 on config error, 1 on runtime error.
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
		return error instanceof ConfigError ? 2 : 1;
	}
}
