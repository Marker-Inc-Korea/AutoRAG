import { existsSync } from "node:fs";
import { resolveConfigPath, resolveConfigReadOnly, writeDefaultConfig } from "../config.ts";
import { renderError } from "../output.ts";
import { runSetup, type SetupReport } from "../setup.ts";
import type { CommandContext } from "./types.ts";

export async function runSetupCommand(ctx: CommandContext): Promise<number> {
	try {
		const resolved = resolveConfigPath({ flags: ctx.flags, cwd: ctx.cwd });
		if (!existsSync(resolved.configPath)) {
			writeDefaultConfig(resolved.configPath, {
				...(typeof ctx.flags.workspace === "string" ? { workspacePath: ctx.flags.workspace } : {}),
				...(typeof ctx.flags["search-paths"] === "string"
					? {
							searchPaths: ctx.flags["search-paths"]
								.split(",")
								.map((p) => p.trim())
								.filter(Boolean),
						}
					: {}),
			});
		}
		const resolvedConfig = resolveConfigReadOnly({ flags: ctx.flags, cwd: ctx.cwd });
		const report = await runSetup({
			configPath: resolved.configPath,
			workspacePath: resolvedConfig.workspacePath,
			syncHister: ctx.flags["sync-hister"] === true,
		});
		ctx.stdout(ctx.json || ctx.flags.format === "json" ? JSON.stringify(report, null, 2) : renderSetup(report));
		return report.ok ? 0 : 1;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

function renderHister(report: SetupReport): string[] {
	const reports = report.histers ?? (report.hister === undefined ? [] : [report.hister]);
	return reports.flatMap((hister) => {
		const lines: string[] = [];
		const head = `hister ${hister.datasource}:`;
		if (!hister.installed) lines.push(`${head} not installed`);
		else if (hister.running === false) lines.push(`${head} not running`);
		else lines.push(`${head} running`);
		if (hister.reason) lines.push(`  ${hister.reason}`);
		const embedding = hister.embedding;
		if (embedding !== undefined) {
			if (embedding.applied) {
				lines.push(`  embedding: now shares the model (updated ${embedding.differs.join(", ")})`);
			} else if (embedding.inSync) {
				lines.push("  embedding: shares the model");
			} else if (embedding.differs.length > 0) {
				lines.push(`  embedding: differs from the shared model (${embedding.differs.join(", ")})`);
			} else if (embedding.reason === undefined) {
				lines.push("  embedding: not checked");
			}
			if (embedding.reason) lines.push(`  embedding: ${embedding.reason}`);
		}
		if (hister.remediation) lines.push(`  next: ${hister.remediation}`);
		return lines;
	});
}

export function renderSetup(report: SetupReport): string {
	return [
		`setup: ${report.mode}`,
		report.embedding.backend === undefined
			? "embedding: blocked"
			: `embedding: ${report.embedding.backend} (${report.embedding.provider}/${report.embedding.model}, ${report.embedding.dimension}d)`,
		...(report.embedding.reason ? [`embedding: ${report.embedding.reason}`] : []),
		...report.datasources.map((d) => `  ${d.name}: ${d.state}${d.reason ? ` (${d.reason})` : ""}`),
		...renderHister(report),
		...(report.remediation ? [`remediation: ${report.remediation}`] : []),
	].join("\n");
}
