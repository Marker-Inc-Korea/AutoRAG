import { existsSync, lstatSync, rmSync } from "node:fs";
import { join, resolve, sep } from "node:path";
import { AutoRAGAgent, type AutoRAGRefreshResult } from "../../agent/agent.ts";
import { MINSYNC_SUBDIR } from "../../minsync/paths.ts";
import { PARSED_MIRROR_SUBDIR } from "../../mirror/paths.ts";
import { BM25_SUBDIR } from "../../retrieval/methods/bm25.ts";
import { buildAgentOptions, type CliConfig, resolveConfig } from "../config.ts";
import { renderError, renderIndex } from "../output.ts";
import type { CommandContext } from "./types.ts";

const RESET_TARGETS = [PARSED_MIRROR_SUBDIR, BM25_SUBDIR, MINSYNC_SUBDIR] as const;
const RESET_TARGET_NAMES = ["parsed", "bm25", "minsync"] as const;

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
		ctx.stderr(renderError(new Error("Usage: autorag index <reset|rebuild> [--yes]"), { json: ctx.json }));
		return 2;
	}

	let config: CliConfig;
	try {
		config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 2;
	}

	const autoragDir = resolve(config.workspacePath, ".autorag");
	const targets = RESET_TARGETS.map((subdir) => resolve(config.workspacePath, subdir));

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
				`Reset the ${RESET_TARGET_NAMES.join(", ")} indexes under ${join(config.workspacePath, ".autorag")}?`,
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
		ctx.stdout(renderIndex({ action: "reset", removed: [...RESET_TARGET_NAMES] }, { json: ctx.json }));
		return 0;
	}

	// rebuild: re-run a forced refresh with a model-free agent.
	let rebuilt: AutoRAGRefreshResult;
	try {
		const agent = new AutoRAGAgent(buildAgentOptions(config));
		rebuilt = await agent.refresh(true);
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json }));
		return 1;
	}

	ctx.stdout(renderIndex({ action: "rebuild", removed: [...RESET_TARGET_NAMES], rebuilt }, { json: ctx.json }));
	return 0;
}

function isWithin(target: string, base: string): boolean {
	const prefix = base.endsWith(sep) ? base : `${base}${sep}`;
	return target === base || target.startsWith(prefix);
}
