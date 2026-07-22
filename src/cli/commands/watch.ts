import { AutoRAGAgent } from "../../agent/agent.ts";
import { buildAgentOptions, resolveConfig } from "../config.ts";
import { renderError, renderRefresh } from "../output.ts";
import type { CommandContext } from "./types.ts";

function parsePositiveInt(value: string | boolean | undefined, fallback: number): number {
	if (typeof value !== "string" || value.trim().length === 0) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/**
 * `autorag watch` — keep parsed/BM25/MinSync indexes current.
 *
 * - default: long-running recursive fs watch with debounced refresh
 * - `--once`: single refresh tick (for cron / launchd / Task Scheduler)
 */
export async function runWatch(ctx: CommandContext): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		const force = ctx.flags.force === true;
		const once = ctx.flags.once === true;

		if (once) {
			const result = await agent.refresh(force);
			ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
			return 0;
		}

		const debounceMs = parsePositiveInt(ctx.flags["debounce-ms"], 1500);
		const immediate = ctx.flags.immediate !== false;

		if (immediate) {
			const result = await agent.refresh(force);
			ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
		}

		const handle = agent.startWatchRefresh({ debounceMs, force });
		const started = {
			mode: "watch",
			roots: config.searchPaths.length,
			debounceMs,
			force,
			immediate,
		};
		ctx.stdout(
			ctx.json
				? JSON.stringify({
						ok: true,
						...started,
						message: "watching for source changes; send SIGINT/SIGTERM to stop",
					})
				: `AutoRAG watch started (roots=${started.roots}, debounceMs=${debounceMs}). Ctrl-C to stop.`,
		);

		await new Promise<void>((resolve) => {
			let stopped = false;
			const stop = () => {
				if (stopped) return;
				stopped = true;
				handle.stop();
				resolve();
			};
			process.once("SIGINT", stop);
			process.once("SIGTERM", stop);
		});

		if (ctx.json) {
			ctx.stdout(JSON.stringify({ ok: true, mode: "watch", stopped: true }));
		} else {
			ctx.stdout("AutoRAG watch stopped.");
		}
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
