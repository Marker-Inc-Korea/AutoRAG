import { existsSync, lstatSync, rmSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import { AutoRAGAgent, type AutoRAGRefreshResult, type RefreshMethod } from "../../agent/agent.ts";
import { MINSYNC_SUBDIR } from "../../minsync/paths.ts";
import { PARSED_MIRROR_SUBDIR } from "../../mirror/paths.ts";
import { BM25_SUBDIR } from "../../retrieval/methods/bm25.ts";
import { buildAgentOptions, type CliConfig, resolveConfig } from "../config.ts";
import { renderError, renderIndex } from "../output.ts";
import { parseMethodFlag } from "./refresh.ts";
import type { CommandContext } from "./types.ts";

const ALL_RESET_TARGETS = [PARSED_MIRROR_SUBDIR, BM25_SUBDIR, MINSYNC_SUBDIR] as const;
const ALL_RESET_TARGET_NAMES = ["parsed", "bm25", "minsync"] as const;

/**
 * Run the `autorag index` command. `reset` removes the parsed mirror, BM25,
 * and MinSync index directories under `.autorag` (preserving `bin`,
 * `datasources`, and the memory file). `rebuild` resets then re-runs a forced
 * refresh. Returns 0 on success, 2 for usage/decline errors, 1 for runtime
 * errors including path-escape guard violations.
 */
export async function runIndex(ctx: CommandContext): Promise<number> {
	const sub = ctx.positionals[0];
	if (sub !== "reset" && sub !== "rebuild") {
		ctx.stderr(renderError(new Error("Usage: autorag index <reset|rebuild> [--yes] [--method]"), { json: ctx.json }));
		return 2;
	}

	let config: CliConfig;
	try {
		config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 2;
	}

	// Determine scoped reset targets from --method. Default: all three.
	const methods = parseMethodFlag(ctx.flags.method);
	const { targetNames, targetSubdirs, refreshMethods } = resolveResetScope(methods);

	const autoragDir = resolve(config.workspacePath, ".autorag");
	const targets = targetSubdirs.map((subdir) => resolve(config.workspacePath, subdir));

	// Guard 1: `.autorag` and each index dir must be real directories owned by
	// autorag. A symlink here could redirect rmSync outside the workspace, so
	// refuse rather than follow it (lexical containment alone cannot catch this).
	for (const dir of [autoragDir, ...targets]) {
		if (existsSync(dir) && lstatSync(dir).isSymbolicLink()) {
			ctx.stderr(
				renderError(new Error("Refusing to reset: an index path is a symlink, not a real directory."), {
					json: ctx.json,
				}),
			);
			return 1;
		}
	}

	// Guard 2: every target must resolve inside the .autorag directory so a
	// mis-resolved workspacePath can never delete outside it.
	for (const target of targets) {
		if (!isWithin(target, autoragDir)) {
			ctx.stderr(renderError(new Error(`Refusing to reset target outside .autorag: ${target}`), { json: ctx.json }));
			return 1;
		}
	}

	// Confirmation: --yes bypasses; otherwise require an interactive yes.
	if (!ctx.flags.yes) {
		if (ctx.promptYesNo) {
			const ok = await ctx.promptYesNo(
				`Reset the ${targetNames.join(", ")} indexes under ${join(config.workspacePath, ".autorag")}?`,
			);
			if (!ok) {
				ctx.stderr(renderError(new Error("Reset declined."), { json: ctx.json }));
				return 2;
			}
		} else {
			ctx.stderr(
				renderError(new Error("Reset requires --yes or an interactive terminal (declined)."), { json: ctx.json }),
			);
			return 2;
		}
	}

	// Remove each existing target. force:true makes this idempotent.
	for (const target of targets) {
		rmSync(target, { recursive: true, force: true });
	}

	if (sub === "reset") {
		ctx.stdout(renderIndex({ action: "reset", removed: [...targetNames] }, { json: ctx.json }));
		return 0;
	}

	// rebuild: re-run a forced refresh with a model-free agent, scoped to methods.
	let rebuilt: AutoRAGRefreshResult;
	try {
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		rebuilt = await agent.refresh(true, refreshMethods ? { methods: refreshMethods } : undefined);
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 1;
	}

	ctx.stdout(renderIndex({ action: "rebuild", removed: [...targetNames], rebuilt }, { json: ctx.json }));
	return 0;
}

/**
 * Resolve which index directories to reset and which refresh methods to run,
 * based on the parsed `--method` flag. When `methods` is undefined (no flag),
 * all three index dirs are reset and a full refresh runs.
 *
 * Mapping:
 * - `bm25` → reset BM25_SUBDIR, refresh with bm25 (+ parsed, since bm25 needs it)
 * - `minsync` → reset MINSYNC_SUBDIR, refresh with minsync (+ parsed)
 * - `parsed` → reset PARSED_MIRROR_SUBDIR, refresh with parsed only
 * - `all` or undefined → all three dirs + full refresh
 * - `datasources`/`jikji` → no dirs to reset, but included in refresh methods
 */
function resolveResetScope(methods: readonly RefreshMethod[] | undefined): {
	targetNames: readonly string[];
	targetSubdirs: readonly string[];
	refreshMethods: readonly RefreshMethod[] | undefined;
} {
	if (methods === undefined) {
		return {
			targetNames: ALL_RESET_TARGET_NAMES,
			targetSubdirs: ALL_RESET_TARGETS,
			refreshMethods: undefined,
		};
	}
	const subdirs: string[] = [];
	const names: string[] = [];
	const refresh: RefreshMethod[] = [];
	for (const m of methods) {
		refresh.push(m);
		if (m === "parsed") {
			subdirs.push(PARSED_MIRROR_SUBDIR);
			names.push("parsed");
		} else if (m === "bm25") {
			subdirs.push(BM25_SUBDIR);
			names.push("bm25");
		} else if (m === "minsync") {
			subdirs.push(MINSYNC_SUBDIR);
			names.push("minsync");
		}
		// datasources/jikji have no reset dir but are valid refresh methods.
	}
	return {
		targetNames: names.length > 0 ? names : ALL_RESET_TARGET_NAMES,
		targetSubdirs: subdirs.length > 0 ? subdirs : ALL_RESET_TARGETS,
		refreshMethods: refresh,
	};
}

function isWithin(target: string, base: string): boolean {
	const prefix = base.endsWith(sep) ? base : `${base}${sep}`;
	return target === base || target.startsWith(prefix);
}
