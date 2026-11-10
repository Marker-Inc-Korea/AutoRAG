/**
 * `autorag lite find <query>` — a model-free whole search for a Cmd+F style
 * launcher: the platform file-name index, local document content, and every
 * connected datasource, streamed as NDJSON (one line per group, in completion
 * order) so a slow datasource never holds back the rest.
 */

import { createDecisionModels } from "../../agent/decision-model-check.ts";
import { type AutoRAGLite, createAutoRAGLite } from "../../core.ts";
import {
	type FindGroupDescriptor,
	type FindGroupLine,
	type FindJudge,
	type FindLine,
	type FindRuntime,
	runFind,
} from "../../find/index.ts";
import { buildAgentOptions, type CliConfig, resolveSllmDecisionModel } from "../config.ts";
import { renderError } from "../output.ts";
import type { CommandContext } from "./types.ts";

/** Test seams: the pure search is injected here so no test touches a real backend. */
export interface LiteFindOverrides {
	readonly config?: CliConfig;
	readonly runtime?: FindRuntime;
	readonly judge?: FindJudge;
	readonly platform?: NodeJS.Platform;
	readonly now?: () => number;
	readonly emit?: (line: FindLine) => void;
}

const USAGE = "Usage: autorag lite find <query> [--query TEXT] [--top-k N] [--groups key,key]";

/**
 * The query: `--query` wins over positionals, so a query that begins with `-`
 * (`--query=--help`) is searchable instead of parsed as a flag.
 */
function resolveQuery(ctx: CommandContext): string {
	const flag = ctx.flags.query;
	if (typeof flag === "string") return flag.trim();
	return ctx.positionals.join(" ").trim();
}

function parseTopK(
	value: string | boolean | undefined,
): { readonly kind: "ok"; readonly value: number | undefined } | { readonly kind: "reject"; readonly error: string } {
	if (value === undefined || value === true || value === false) return { kind: "ok", value: undefined };
	const parsed = Number(value);
	if (!Number.isFinite(parsed) || !Number.isInteger(parsed)) {
		return { kind: "reject", error: `Invalid --top-k value: "${String(value)}". Must be a positive integer.` };
	}
	if (parsed < 1) return { kind: "reject", error: `Invalid --top-k value: ${parsed}. Must be a positive integer.` };
	return { kind: "ok", value: parsed };
}

/** `--groups a,b` (or a repeated flag) narrowed to the listed keys; undefined runs every group. */
function parseGroups(ctx: CommandContext): string[] | undefined {
	const raw: string[] = [];
	if (typeof ctx.flags.groups === "string") raw.push(ctx.flags.groups);
	for (const value of ctx.repeatedFlags?.groups ?? []) raw.push(value);
	const keys = raw
		.flatMap((value) => value.split(","))
		.map((value) => value.trim())
		.filter((value) => value.length > 0);
	return keys.length > 0 ? keys : undefined;
}

/** The configured datasources that are enabled and registered, in config order. */
function configuredDatasources(config: CliConfig, runtime: FindRuntime): FindGroupDescriptor[] {
	const registered = new Set(runtime.listDatasources().map((entry) => entry.datasourceId));
	const descriptors: FindGroupDescriptor[] = [];
	for (const [id, entry] of Object.entries(config.datasources ?? {})) {
		if (entry === false) continue;
		if (entry !== true && entry.enabled === false) continue;
		if (!registered.has(id)) continue;
		const type = entry !== true && typeof entry.type === "string" && entry.type.length > 0 ? entry.type : id;
		descriptors.push({ key: `source:${id}`, datasourceId: id, datasourceType: type });
	}
	return descriptors;
}

/** Whether the platform file-name index is enabled in config (default enabled). */
function filenameIndexEnabled(config: CliConfig, platform: NodeJS.Platform): boolean {
	if (platform === "win32") return config.everything !== false && config.everything?.enabled !== false;
	return config.fsearch !== false && config.fsearch?.enabled !== false;
}

function renderHuman(
	groupsLine: readonly FindGroupDescriptor[],
	groups: readonly FindGroupLine[],
	elapsedMs: number,
): string {
	const lines: string[] = [];
	for (const group of groups) {
		const summary = group.status === "ready" ? `${group.hits.length} hits` : group.status;
		lines.push(`${group.key}: ${summary}${group.error === null ? "" : ` (${group.error})`}`);
	}
	lines.push(`done in ${elapsedMs}ms (${groupsLine.length} groups)`);
	return lines.join("\n");
}

/**
 * Run the command with injectable seams; {@link runLiteFind} is the dispatch
 * entry point. Returns 2 for a usage error, config-resolution failure, or an
 * unusable runtime; 0 otherwise (individual group failures stay on the stream).
 */
export async function runLiteFindWith(ctx: CommandContext, overrides: LiteFindOverrides = {}): Promise<number> {
	const query = resolveQuery(ctx);
	if (query.length === 0) {
		ctx.stderr(renderError(new Error(USAGE), { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const topK = parseTopK(ctx.flags["top-k"]);
	if (topK.kind === "reject") {
		ctx.stderr(ctx.json ? JSON.stringify({ ok: false, error: topK.error }) : topK.error);
		return 2;
	}

	let lite: AutoRAGLite | undefined;
	let config: CliConfig;
	try {
		if (overrides.config !== undefined) {
			config = overrides.config;
		} else {
			lite = createAutoRAGLite({ flags: ctx.flags, cwd: ctx.cwd });
			config = lite.config;
		}
	} catch (error) {
		ctx.stderr(renderError(error, { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const runtime = overrides.runtime ?? lite;
	if (runtime === undefined) {
		ctx.stderr(renderError(new Error("lite find needs a runtime"), { json: ctx.json, debug: ctx.debug }));
		return 2;
	}

	const platform = overrides.platform ?? process.platform;
	const sourceDescriptors = configuredDatasources(config, runtime);
	const wantedGroups = parseGroups(ctx);
	const allGroups: readonly FindGroupDescriptor[] = [{ key: "files" }, { key: "content" }, ...sourceDescriptors];
	const groupsLine =
		wantedGroups === undefined ? allGroups : allGroups.filter((descriptor) => wantedGroups.includes(descriptor.key));

	const groupLines: FindGroupLine[] = [];
	let doneElapsed = 0;
	const emit: (line: FindLine) => void =
		overrides.emit ??
		((line) => {
			if (ctx.json) {
				ctx.stdout(JSON.stringify(line));
			} else if (line.type === "group") {
				groupLines.push(line);
			} else if (line.type === "done") {
				doneElapsed = line.elapsedMs;
			}
		});

	// Announce the group list before any slow work (judge resolution, datasource
	// searches) so the UI can paint its tabs immediately.
	emit({ type: "groups", groups: groupsLine });

	let judge: FindJudge | undefined = overrides.judge;
	if (judge === undefined) {
		try {
			const sllm = await resolveSllmDecisionModel(config).catch(() => undefined);
			judge = createDecisionModels({
				...buildAgentOptions(config).jev,
				...(sllm === undefined ? {} : { sllm }),
			}).judge;
		} catch {
			// No decision model can be built: the gate fails open and keeps every hit.
			judge = undefined;
		}
	}

	await runFind({
		query,
		topK: topK.value ?? 10,
		runtime,
		...(judge === undefined ? {} : { judge }),
		platform,
		fileSearchEnabled: filenameIndexEnabled(config, platform),
		semanticEnabled: true,
		datasources: sourceDescriptors,
		...(wantedGroups === undefined ? {} : { groups: wantedGroups }),
		announce: false,
		...(overrides.now === undefined ? {} : { now: overrides.now }),
		emit,
	});

	if (overrides.emit === undefined && !ctx.json) {
		ctx.stdout(renderHuman(groupsLine, groupLines, doneElapsed));
	}
	return 0;
}

/** Dispatch entry point for `autorag lite find`. */
export async function runLiteFind(ctx: CommandContext): Promise<number> {
	return runLiteFindWith(ctx);
}
