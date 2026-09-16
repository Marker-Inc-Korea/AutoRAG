import { importModel, prefetchModel, verifyModel } from "../../embedding-runtime/index.ts";
import type { ProfileId } from "../../embedding-runtime/types.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export interface ModelsCommandDeps {
	prefetchModel?: (profileId?: ProfileId) => Promise<{ profileId: ProfileId; path: string }>;
	importModel?: (
		profileId: ProfileId | undefined,
		sourcePath: string,
	) => Promise<{ profileId: ProfileId; path: string }>;
	verifyModel?: (profileId?: ProfileId) => Promise<{ profileId: ProfileId; path: string; hash: string }>;
}

function profile(ctx: CommandContext): ProfileId | undefined {
	const value = ctx.flags.profile;
	return typeof value === "string" ? (value as ProfileId) : undefined;
}
function output(ctx: CommandContext, result: unknown, action: string): void {
	ctx.stdout(
		ctx.json || ctx.flags.format === "json"
			? JSON.stringify({ ok: true, action, ...((result ?? {}) as object) }, null, 2)
			: `models: ${action}`,
	);
}
export async function runModels(ctx: CommandContext, deps: ModelsCommandDeps = {}): Promise<number> {
	const sub = ctx.positionals[0];
	try {
		if (sub === "prefetch") {
			const result = await (deps.prefetchModel ?? prefetchModel)(profile(ctx));
			output(ctx, result, "prefetch");
			return 0;
		}
		if (sub === "import") {
			const source = ctx.positionals[1];
			if (!source) throw new UsageError("Usage: autorag models import <path> [--profile <id>]");
			const result = await (deps.importModel ?? importModel)(profile(ctx), source);
			output(ctx, result, "import");
			return 0;
		}
		if (sub === "verify") {
			const result = await (deps.verifyModel ?? verifyModel)(profile(ctx));
			output(ctx, result, "verify");
			return 0;
		}
		throw new UsageError("Usage: autorag models prefetch|import <path>|verify [--profile <id>]");
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json || ctx.flags.format === "json", debug: ctx.debug }));
		return error instanceof UsageError ? 2 : 1;
	}
}
class UsageError extends Error {}
