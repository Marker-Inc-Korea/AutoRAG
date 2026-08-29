import { existsSync, lstatSync, rmSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import { AutoRAGAgent, type AutoRAGRefreshResult, type RefreshMethod } from "../../agent/agent.ts";
import { MINSYNC_SUBDIR } from "../../minsync/paths.ts";
import { PARSED_MIRROR_SUBDIR } from "../../mirror/paths.ts";
import { acquireRefreshLock } from "../../mirror/refresh-lock.ts";
import { BM25_SUBDIR } from "../../retrieval/methods/bm25.ts";
import { buildAgentOptions, type CliConfig, resolveConfig } from "../config.ts";
import { renderError, renderIndex } from "../output.ts";
import { parseMethodFlag } from "./refresh.ts";
import type { CommandContext } from "./types.ts";

const ALL_RESET_TARGETS = [PARSED_MIRROR_SUBDIR, MINSYNC_SUBDIR, BM25_SUBDIR] as const;
const ALL_RESET_TARGET_NAMES = ["parsed", "minsync", "bm25"] as const;

/**
 * Run the `autorag index` command. `reset` removes the parsed mirror
 * MinSync, and leftover legacy BM25 index directories under `.autorag`
 * (preserving `bin`, `datasources`, and the memory file). `rebuild` resets then re-runs a forced
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

	const lock = acquireRefreshLock(config.workspacePath);
	if (!lock) {
		ctx.stderr(
			renderError(new Error("A refresh is already running for this workspace; nothing was reset."), {
				json: ctx.json,
			}),
		);
		return 1;
	}
	try {
		// Remove each existing target. force:true makes this idempotent.
		for (const target of targets) {
			rmSync(target, { recursive: true, force: true });
		}

		if (sub === "reset") {
			ctx.stdout(renderIndex({ action: "reset", removed: [...targetNames] }, { json: ctx.json }));
			return 0;
		}

		// rebuild: re-run a forced refresh with a model-free agent, scoped to methods, under the same lock.
		let rebuilt: AutoRAGRefreshResult;
		try {
			const agent = new AutoRAGAgent(buildAgentOptions(config));
			rebuilt = await agent.refresh(true, {
				lock,
				...(refreshMethods ? { methods: refreshMethods } : {}),
			});
		} catch (error) {
			ctx.stderr(renderError(error, { json: ctx.json }));
			return 1;
		}

		if (rebuilt.outcome === "busy") {
			ctx.stderr(renderError(new Error("Rebuild was refused before re-indexing ran."), { json: ctx.json }));
			return 1;
		}
		ctx.stdout(renderIndex({ action: "rebuild", removed: [...targetNames], rebuilt }, { json: ctx.json }));
		return 0;
	} finally {
		lock.release();
	}
}

/**
 * Resolve which index directories to reset and which refresh methods to run,
 * based on the parsed `--method` flag. When `methods` is undefined (no flag),
 * all three index dirs are reset and a full refresh runs.
 *
 * Mapping:
 * - `bm25` → refresh with bm25 (+ parsed, since MinSync BM25 needs it)
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
	const addTarget = (name: string, subdir: string): void => {
		if (names.includes(name)) return;
		names.push(name);
		subdirs.push(subdir);
	};
	for (const m of methods) {
		refresh.push(m);
		if (m === "parsed") {
			addTarget("parsed", PARSED_MIRROR_SUBDIR);
		} else if (m === "bm25") {
			addTarget("bm25", BM25_SUBDIR);
		} else if (m === "minsync") {
			addTarget("minsync", MINSYNC_SUBDIR);
			addTarget("bm25", BM25_SUBDIR);
		}
		// datasources/jikji have no reset dir but are valid refresh methods.
	}
	return {
		targetNames: names,
		targetSubdirs: subdirs,
		refreshMethods: refresh,
	};
}

function isWithin(target: string, base: string): boolean {
	const prefix = base.endsWith(sep) ? base : `${base}${sep}`;
	return target === base || target.startsWith(prefix);
}
