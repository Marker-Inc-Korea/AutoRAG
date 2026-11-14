import { AutoRAGAgent } from "../../agent/agent.ts";
import { type IndexSchedule, validateIndexSchedule } from "../../agent/index-schedule.ts";
import { buildAgentOptions, ConfigError, resolveConfig } from "../config.ts";
import { renderError, renderRefresh } from "../output.ts";
import { abortOnSignal, SIGNAL_EXIT_CODES, type SignalAbort } from "../signal-abort.ts";
import type { CommandContext } from "./types.ts";

function parsePositiveInt(value: string | boolean | undefined, fallback: number): number {
	if (typeof value !== "string" || value.trim().length === 0) return fallback;
	const parsed = Number.parseInt(value, 10);
	return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/** CLI schedule flags (`--power`, `--window`, `--days`); undefined when none given so the config schedule applies. */
function scheduleFromFlags(ctx: CommandContext): IndexSchedule | undefined {
	const windows = ctx.repeatedFlags?.window ?? (typeof ctx.flags.window === "string" ? [ctx.flags.window] : undefined);
	const power = ctx.flags.power;
	const days = typeof ctx.flags.days === "string" ? ctx.flags.days.split(",").map((d) => d.trim()) : undefined;
	if (power === undefined && windows === undefined && days === undefined) return undefined;
	try {
		return validateIndexSchedule({
			...(power !== undefined ? { power } : {}),
			...(windows !== undefined ? { windows } : {}),
			...(days !== undefined ? { days } : {}),
		});
	} catch (error) {
		throw new ConfigError((error as Error).message.replace("refresh.schedule", "--power/--window/--days"));
	}
}

/**
 * `autorag watch` — keep parsed/MinSync indexes current.
 *
 * - default: long-running recursive fs watch with debounced refresh
 * - `--once`: single refresh tick (for cron / launchd / Task Scheduler)
 *
 * Exit codes: 0 on success, 2 on config error, 1 on runtime error, 130/143 when a one-shot tick is stopped by
 * SIGINT/SIGTERM. The long-running watch handles those signals itself and exits 0.
 */
export async function runWatch(ctx: CommandContext): Promise<number> {
	// Only a tick that ends on its own needs the abort: the watch loop waits on the signals it handles below.
	const stop: SignalAbort | undefined = ctx.flags.once === true ? abortOnSignal() : undefined;
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		const force = ctx.flags.force === true;
		const once = ctx.flags.once === true;
		const flagSchedule = scheduleFromFlags(ctx);
		const decision = await agent.indexScheduleDecision(flagSchedule);

		if (once) {
			if (!decision.allowed) {
				ctx.stdout(
					ctx.json
						? JSON.stringify({ ok: true, skipped: true, reason: decision.reason })
						: `Index schedule closed (${decision.reason}); skipped this tick.`,
				);
				return 0;
			}
			const result = await agent.refresh(force, stop === undefined ? undefined : { signal: stop.signal });
			ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
			return 0;
		}

		const debounceMs = parsePositiveInt(ctx.flags["debounce-ms"], 1500);
		const immediate = ctx.flags.immediate !== false;

		if (immediate && !decision.allowed) {
			ctx.stdout(
				ctx.json
					? JSON.stringify({ ok: true, skipped: true, reason: decision.reason })
					: `Index schedule closed (${decision.reason}); initial refresh deferred.`,
			);
		} else if (immediate) {
			const result = await agent.refresh(force);
			ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
		}

		const handle = agent.startWatchRefresh({
			debounceMs,
			force,
			...(flagSchedule !== undefined ? { schedule: flagSchedule } : {}),
		});
		// A refresh deferred by a closed schedule is queued; the watcher runs it once the schedule opens.
		if (immediate && !decision.allowed) handle.trigger();
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
			// Watchers and timers are unref'd, so without a ref'd handle the process would exit right after start.
			const keepAlive = setInterval(() => {}, 1 << 30);
			const stop = () => {
				if (stopped) return;
				stopped = true;
				clearInterval(keepAlive);
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
		if (stop?.received !== undefined) return SIGNAL_EXIT_CODES[stop.received];
		return error instanceof ConfigError ? 2 : 1;
	} finally {
		stop?.dispose();
	}
}
