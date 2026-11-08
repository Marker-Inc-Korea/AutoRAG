import { existsSync } from "node:fs";
import { resolve } from "node:path";
import { buildLtrDataset, writeLtrDataset } from "../../memory/ltr-export.ts";
import { RetrievalMemory } from "../../memory/memory.ts";
import { RankingLogStore, rankingLogPath } from "../../memory/ranking-log.ts";
import { resolveConfig } from "../config.ts";
import { renderError, renderMemory } from "../output.ts";
import type { CommandContext } from "./types.ts";

const USAGE = "Usage: autorag memory <inspect|export-ltr> [--output <dir>] [--valid-fraction <0..1>]";

/** Share of the newest runs held out for validation when `--valid-fraction` is omitted. */
const DEFAULT_VALID_FRACTION = 0.2;

/**
 * `autorag memory inspect` — render a path-opaque snapshot of the retrieval
 * memory (judged evidence, curated results, evidence chunks, insights).
 * Model-free. The memory file is resolved from config and loaded read-only;
 * only the rendered schema is emitted, never the storage path.
 *
 * `autorag memory export-ltr --output <dir>` — turn the learning-to-rank log
 * beside the memory file into LightGBM LambdaRank files (see
 * docs/learning-to-rank.md). Model-free and read-only on the log.
 */
export async function runMemory(ctx: CommandContext): Promise<number> {
	const subcommand = ctx.positionals[0];
	if (subcommand === "export-ltr") return exportLtr(ctx);
	if (subcommand !== "inspect") {
		ctx.stderr(renderError(new Error(USAGE), { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const memory = new RetrievalMemory({ storagePath: config.memoryPath });
		memory.load();
		ctx.stdout(renderMemory(memory.getSchema(), { json: ctx.json, debug: ctx.debug }));
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}

function exportLtr(ctx: CommandContext): number {
	const output = ctx.flags.output;
	if (typeof output !== "string" || output.trim() === "") {
		ctx.stderr(
			renderError(new Error(`export-ltr requires --output <dir>. ${USAGE}`), { json: ctx.json, debug: ctx.debug }),
		);
		return 2;
	}
	const rawFraction = ctx.flags["valid-fraction"];
	const validFraction = typeof rawFraction === "string" ? Number(rawFraction) : DEFAULT_VALID_FRACTION;
	if (!(validFraction >= 0 && validFraction < 1)) {
		ctx.stderr(
			renderError(new Error(`--valid-fraction must be a number in [0, 1), got ${String(rawFraction)}`), {
				json: ctx.json,
				debug: ctx.debug,
			}),
		);
		return 2;
	}
	try {
		const config = resolveConfig({ flags: ctx.flags, cwd: ctx.cwd });
		const logPath = rankingLogPath(config.memoryPath);
		// Exporting never creates the log: no searches yet is an empty dataset, not a new file in the user's home.
		let runs: Parameters<typeof buildLtrDataset>[0] = [];
		let skippedRecords = 0;
		if (existsSync(logPath)) {
			const store = new RankingLogStore(logPath);
			try {
				const listing = store.list();
				runs = listing.runs;
				skippedRecords = listing.skipped;
			} finally {
				store.close();
			}
		}
		const dataset = buildLtrDataset(runs, { validFraction });
		const directory = resolve(ctx.cwd, output);
		const { files } = writeLtrDataset(dataset, directory);
		const { manifest } = dataset;
		const summary = {
			output: directory,
			runs: manifest.runs,
			skippedRecords,
			skippedGroups: manifest.skippedGroups,
			train: manifest.train,
			valid: manifest.valid,
			labelCounts: manifest.labelCounts,
			files,
		};
		ctx.stdout(
			ctx.json
				? JSON.stringify(summary)
				: [
						`Exported ${manifest.train.groups + manifest.valid.groups} of ${manifest.runs} logged search runs to ${directory}`,
						`  train: ${manifest.train.groups} queries, ${manifest.train.rows} rows`,
						`  valid: ${manifest.valid.groups} queries, ${manifest.valid.rows} rows`,
						`  skipped: ${manifest.skippedGroups.singleCandidate} single-candidate, ${manifest.skippedGroups.noLabelVariance} with no label variance, ${skippedRecords} unreadable`,
						`  labels: ${JSON.stringify(manifest.labelCounts)}`,
					].join("\n"),
		);
		return 0;
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 1;
	}
}
