import { existsSync } from "node:fs";
import { resolveConfigPath, resolveConfigReadOnly, writeDefaultConfig } from "../config.ts";
import { renderError } from "../output.ts";
import { runSetup } from "../setup.ts";
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
			...(typeof ctx.flags.profile === "string"
				? { profileId: ctx.flags.profile as "qwen3-embedding-0.6b" | "embeddinggemma-300m" }
				: {}),
		});
		ctx.stdout(ctx.json || ctx.flags.format === "json" ? JSON.stringify(report, null, 2) : renderSetup(report));
		return report.ok ? 0 : 1;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
function renderSetup(report: Awaited<ReturnType<typeof runSetup>>): string {
	return [
		`setup: ${report.mode}`,
		`model: ${report.model.valid ? "ready" : "blocked"}`,
		...report.datasources.map((d) => `  ${d.name}: ${d.state}${d.reason ? ` (${d.reason})` : ""}`),
		...(report.remediation ? [`remediation: ${report.remediation}`] : []),
	].join("\n");
}
