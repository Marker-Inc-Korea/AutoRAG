import type { DiscordAttachmentRecord, DiscordAttachmentSyncResult } from "../../datasource/skills/discrawl/index.ts";
import { type DiscordAttachmentConnection, discordAttachmentConnections } from "../../datasource/skills/factory.ts";
import { type CliConfig, ConfigError, resolveConfig } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/**
 * `autorag attachments [status|download]` — show or fill each Discord connection's attachments folder.
 *
 * `status` re-checks every saved file (a deleted, renamed, or edited file is no longer counted as downloaded)
 * and rewrites `download-status.json`. `download` additionally saves every attachment whose signed link still
 * works, even when `connector.downloadAttachments` is `false`.
 *
 * Exit codes: 0 on success, 2 on config/usage error, 1 when a connection could not be checked.
 */
export async function runAttachments(
	ctx: CommandContext,
	connections: (config: CliConfig) => readonly DiscordAttachmentConnection[] = (config) =>
		discordAttachmentConnections(config.datasources, config.workspacePath),
): Promise<number> {
	const action = ctx.positionals[0] ?? "status";
	if (action !== "status" && action !== "download") {
		ctx.stderr(`Unknown attachments action: ${action}. Use status or download.`);
		return 2;
	}
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const wanted = typeof ctx.flags.connection === "string" ? ctx.flags.connection : undefined;
		const all = connections(config);
		const selected = wanted === undefined ? all : all.filter((connection) => connection.name === wanted);
		if (selected.length === 0) {
			ctx.stderr(
				wanted === undefined
					? "No Discord datasource is configured."
					: `No enabled Discord datasource named ${wanted}. Configured: ${all.map((c) => c.name).join(", ") || "none"}`,
			);
			return 2;
		}
		const results: { connection: DiscordAttachmentConnection; result: DiscordAttachmentSyncResult }[] = [];
		for (const connection of selected) {
			results.push({ connection, result: await connection.store.sync({ download: action === "download" }) });
		}
		if (ctx.json) {
			ctx.stdout(
				JSON.stringify(
					results.map(({ connection, result }) => ({
						connection: connection.name,
						autoDownload: connection.autoDownload,
						folder: connection.store.root,
						...result,
					})),
					null,
					2,
				),
			);
		} else {
			ctx.stdout(
				results.map(({ connection, result }) => renderConnection(connection, result, ctx.debug)).join("\n\n"),
			);
		}
		return results.every(({ result }) => result.ok) ? 0 : 1;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return error instanceof ConfigError ? 2 : 1;
	}
}

const STATE_LABEL: Readonly<Record<DiscordAttachmentRecord["state"], string>> = {
	downloaded: "downloaded",
	"not-downloaded": "not downloaded",
	changed: "changed (kept as your file)",
	missing: "missing (deleted or renamed)",
	failed: "failed",
	skipped: "skipped",
};

function renderConnection(
	connection: DiscordAttachmentConnection,
	result: DiscordAttachmentSyncResult,
	listAll: boolean,
): string {
	const header = `discord/${connection.name}: ${connection.store.root}`;
	if (!result.ok) return `${header}\n  error: ${result.message}`;
	const { summary, run } = result;
	const lines = [
		header,
		`  auto-download on refresh: ${connection.autoDownload ? "on" : "off (connector.downloadAttachments: false)"}`,
		`  ${summary.downloaded}/${summary.total} downloaded, ${summary.notDownloaded} not downloaded, ` +
			`${summary.changed} changed, ${summary.missing} missing, ${summary.failed} failed, ${summary.skipped} skipped`,
		`  downloadable now: ${summary.downloadable}; link expired: ${summary.expired}`,
	];
	if (run.attempted > 0) {
		lines.push(
			`  this run: fetched ${run.fetched}, failed ${run.failed}, skipped ${run.skipped}, ${run.bytes} bytes`,
		);
	}
	const shown = listAll ? result.records : result.records.filter((record) => record.downloadable);
	for (const record of shown) {
		const where = record.path === undefined ? "" : ` -> ${record.path}`;
		const until =
			record.downloadable && record.linkExpiresAt !== undefined ? ` (link until ${record.linkExpiresAt})` : "";
		const error = record.error === undefined ? "" : ` [${record.error}]`;
		lines.push(`  - ${STATE_LABEL[record.state]}: ${record.filename}${where}${until}${error}`);
	}
	lines.push(`  status file: ${result.statusPath}`);
	return lines.join("\n");
}
