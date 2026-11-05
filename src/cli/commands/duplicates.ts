import { resolve } from "node:path";
import type { DupeyScanResult } from "../../dupey/index.ts";
import { listCloudRoots, planDupeyScans, scanWithDupey } from "../../dupey/index.ts";
import { type CloudFileProbe, defaultCloudProbe } from "../../filesystem/cloud-placeholder.ts";
import { ConfigError, resolveConfig } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export type DuplicateScanner = (path: string, excludeDirs: readonly string[]) => Promise<DupeyScanResult>;

interface ExactGroup {
	readonly hash: string;
	readonly files: readonly string[];
}

/**
 * `autorag duplicates` — scan configured roots for exact and near duplicate
 * document families. Never deletes or moves source files.
 *
 * Exit codes: 0 on success, 2 on config error, 1 on runtime error.
 */
export async function runDuplicates(
	ctx: CommandContext,
	scanner: DuplicateScanner = (path, excludeDirs) => scanWithDupey(path, { excludeDirs }),
	cloudProbe: CloudFileProbe | undefined = defaultCloudProbe(),
): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const requested = ctx.positionals[0];
		const roots = requested ? [resolve(ctx.cwd, requested)] : config.searchPaths.map((path) => resolve(path));
		const plan = planDupeyScans(roots, await listCloudRoots(cloudProbe, roots));
		const scans = await Promise.all(plan.targets.map(({ path, excludeDirs }) => scanner(path, excludeDirs)));
		const exactGroups = scans.flatMap(exactGroupsFromScan);
		const extractionErrors = scans.flatMap((scan) => scan.errors);
		if (ctx.json) {
			ctx.stdout(
				JSON.stringify(
					{
						ok: true,
						roots,
						skippedRoots: plan.skipped,
						exactGroups,
						families: scans.flatMap((scan) => scan.families),
						extractionErrors,
						action: "review",
					},
					null,
					2,
				),
			);
			return 0;
		}
		const lines = [
			`duplicates: scanned ${scans.length} root(s)`,
			`  exact groups: ${exactGroups.length}`,
			`  other families: ${scans.reduce((count, scan) => count + scan.families.length, 0)}`,
			`  extraction errors: ${extractionErrors.length}`,
		];
		for (const root of plan.skipped) {
			lines.push(
				`  not scanned (inside a cloud or network folder; scanning would download or re-read every file): ${root}`,
			);
		}
		for (const [index, group] of exactGroups.entries()) {
			lines.push(`  exact group ${index + 1}:`);
			for (const file of group.files) lines.push(`    - ${file}`);
		}
		lines.push(
			"Review non-text properties, then move redundant copies to a review/archive directory before deleting anything.",
			"AutoRAG does not move or delete source files.",
		);
		ctx.stdout(lines.join("\n"));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return error instanceof ConfigError ? 2 : 1;
	}
}

function exactGroupsFromScan(scan: DupeyScanResult): ExactGroup[] {
	const groups = new Map<string, string[]>();
	for (const file of scan.files) {
		if (typeof file.content_hash !== "string" || file.content_hash.length === 0) continue;
		const files = groups.get(file.content_hash) ?? [];
		files.push(file.path);
		groups.set(file.content_hash, files);
	}
	return [...groups.entries()]
		.filter(([, files]) => files.length > 1)
		.map(([hash, files]) => ({ hash, files: [...files].sort() }));
}
