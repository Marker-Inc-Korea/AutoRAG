import { AutoRAGAgent, type RefreshMethod } from "../../agent/agent.ts";
import { buildAgentOptions, resolveConfig } from "../config.ts";
import { renderError, renderRefresh } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag refresh` — parse configured search paths and resync every active
 * index (parsed mirror, BM25, MinSync, datasources, jikji). Model-free: no LLM
 * is constructed. Output is rendered through the path-opaque refresh renderer;
 * the raw result (which carries absolute `indexPath`) is never printed.
 *
 * `--method bm25,minsync,parsed` restricts which methods run. When omitted, all
 * methods run. Parsed mirrors are always synced when BM25 or MinSync is selected
 * (they index over the parsed mirrors).
 */
export async function runRefresh(ctx: CommandContext): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		const methods = parseMethodFlag(ctx.flags.method);
		const result = await agent.refresh(ctx.flags.force === true, methods ? { methods } : undefined);
		ctx.stdout(renderRefresh(result, { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

/** All method names accepted by `--method`. */
const ALL_METHOD_NAMES = ["all", "bm25", "minsync", "parsed", "datasources", "jikji"] as const;

/**
 * Parse a `--method` CSV flag into a sorted set of `RefreshMethod` values.
 * Returns `undefined` when the flag is missing (meaning "all methods").
 * `all` expands to every method. Unknown values throw.
 */
export function parseMethodFlag(flag: string | boolean | undefined): readonly RefreshMethod[] | undefined {
	if (flag === undefined || flag === true || flag === false) return undefined;
	const raw = flag as string;
	const entries = raw
		.split(",")
		.map((entry) => entry.trim().toLowerCase())
		.filter((entry) => entry.length > 0);
	if (entries.length === 0) return undefined;
	const valid = new Set<string>(ALL_METHOD_NAMES);
	for (const entry of entries) {
		if (!valid.has(entry)) {
			throw new Error(`Unknown --method value: ${entry}. Valid: ${ALL_METHOD_NAMES.join(", ")}`);
		}
	}
	if (entries.includes("all")) {
		return ["parsed", "bm25", "minsync", "datasources", "jikji"] as const;
	}
	// Preserve canonical order, deduplicate.
	const order: readonly RefreshMethod[] = ["parsed", "bm25", "minsync", "datasources", "jikji"];
	const seen = new Set<RefreshMethod>();
	for (const m of order) {
		if (entries.includes(m)) seen.add(m);
	}
	return [...seen];
}
