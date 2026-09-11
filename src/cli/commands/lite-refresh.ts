import { createAutoRAGLite } from "../../core.ts";
import { ConfigError } from "../config.ts";
import { renderError } from "../output.ts";
import { parseMethodFlag } from "./refresh.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag lite refresh` — model-free index refresh using AutoRAGLite.
 *
 * Delegates to the agent's refresh pipeline (parsed mirrors, MinSync,
 * datasources, Jikji) without requiring any model configuration.
 *
 * Exit codes: 0 on success, 2 on config error, 1 on runtime error.
 */
export async function runLiteRefresh(ctx: CommandContext): Promise<number> {
	try {
		const lite = createAutoRAGLite({ flags: ctx.flags, cwd: ctx.cwd });
		const methods = parseMethodFlag(ctx.flags.method);
		const force = ctx.flags.force === true || ctx.flags.full === true;
		const result = await lite.refresh(force, methods ? { methods } : undefined);
		const { renderRefresh } = await import("../output.ts");
		ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return error instanceof ConfigError ? 2 : 1;
	}
}
