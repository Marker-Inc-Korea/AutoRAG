import { resolve } from "node:path";
import type { DupeyScanResult } from "../../dupey/index.ts";
import { scanWithDupey } from "../../dupey/index.ts";
import { resolveConfig } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

export type DuplicateScanner = (path: string) => Promise<DupeyScanResult>;

interface ExactGroup {
	readonly hash: string;
	readonly files: readonly string[];
}

export async function runDuplicates(
	ctx: CommandContext,
	scanner: DuplicateScanner = (path) => scanWithDupey(path),
): Promise<number> {
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const requested = ctx.positionals[0];
		const roots = requested ? [resolve(ctx.cwd, requested)] : config.searchPaths.map((path) => resolve(path));
		const scans = await Promise.all(roots.map((root) => scanner(root)));
		const exactGroups = scans.flatMap(exactGroupsFromScan);
		const extractionErrors = scans.flatMap((scan) => scan.errors);
		if (ctx.json) {
			ctx.stdout(
				JSON.stringify(
					{
						ok: true,
						roots,
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
			`duplicates: scanned ${roots.length} root(s)`,
			`  exact groups: ${exactGroups.length}`,
			`  other families: ${scans.reduce((count, scan) => count + scan.families.length, 0)}`,
			`  extraction errors: ${extractionErrors.length}`,
		];
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
		return 1;
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
