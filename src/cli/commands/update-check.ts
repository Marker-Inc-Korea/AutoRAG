import { type AutoRAGUpdateResult, checkAutoRAGUpdate, renderAutoRAGUpdateNotice } from "../update-check.ts";
import { readPackageVersion } from "../version.ts";
import type { CommandContext } from "./types.ts";

export interface UpdateCheckDeps {
	readonly currentVersion?: string;
	readonly check?: (currentVersion: string) => Promise<AutoRAGUpdateResult>;
}

function describe(result: AutoRAGUpdateResult): string {
	const notice = renderAutoRAGUpdateNotice(result);
	if (notice !== undefined) return notice;
	switch (result.status) {
		case "up-to-date":
			return `autorag v${result.currentVersion} is up to date`;
		case "skipped":
			return "update check disabled (AUTORAG_NO_UPDATE_CHECK)";
		default:
			return `could not check for a newer autorag version (have v${result.currentVersion})`;
	}
}

/** `autorag update-check`: compare the running package against the npm registry. */
export async function runUpdateCheck(ctx: CommandContext, deps: UpdateCheckDeps = {}): Promise<number> {
	const currentVersion = deps.currentVersion ?? readPackageVersion();
	const check = deps.check ?? ((version: string) => checkAutoRAGUpdate({ currentVersion: version }));
	const result = await check(currentVersion);
	if (ctx.json) {
		ctx.stdout(JSON.stringify({ ok: true, action: "update-check", ...result }));
		return 0;
	}
	ctx.stdout(describe(result));
	return 0;
}
