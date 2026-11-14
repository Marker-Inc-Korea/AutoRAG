import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";
import type { RefreshMethod } from "../agent/agent.ts";
import { evidenceFor } from "../cli/commands/evidence.ts";
import { validateReport } from "../cli/commands/report.ts";
import type { ConvertAndRelateResult } from "../convert/index.ts";
import type { AutoRAGLite } from "../core.ts";
import { datasourceSearchToolName } from "../datasource/tool-naming.ts";
import { type DupeyScanResult, scanWithDupey } from "../dupey/index.ts";
import {
	filterFileNameSearchMatches,
	resolveConfiguredFileSearchRoot,
	searchFileNames,
} from "../filesystem/name-search.ts";
import { isParsedRefreshComplete } from "../mirror/paths.ts";
import { RetrievalSelectionError } from "../retrieval/index.ts";
import { parseTimeWindow, TimeWindowError } from "../retrieval/time-window.ts";

/** ISO-8601 window bounds, parsed once in the handler so a bad value is returned verbatim, not as a schema error. */
const windowFields = {
	since: z
		.string()
		.optional()
		.describe(
			"ISO-8601 lower bound, inclusive; a date-only value is local midnight. Only evidence dated at or after it.",
		),
	until: z
		.string()
		.optional()
		.describe(
			"ISO-8601 upper bound, EXCLUSIVE; a date-only value is local midnight. 'All of September' is since 2026-09-01, until 2026-10-01.",
		),
};

const datasourceSearchInput = z.strictObject({
	query: z.string().trim().min(1),
	topK: z.number().int().positive().max(100).optional(),
	scope: z.string().trim().min(1).optional(),
	...windowFields,
});

const emptyInput = z.strictObject({});
const objectOutput = z.looseObject({});

const searchInput = z
	.strictObject({
		query: z.string().trim().min(1),
		topK: z.number().int().positive().max(100).optional(),
		scope: z.string().trim().min(1).optional(),
		...windowFields,
		strict: z.boolean().optional(),
		datasourceIds: z.array(z.string().trim().min(1)).optional(),
		methods: z.array(z.string().trim().min(1)).optional(),
		local: z.boolean().optional(),
	})
	.strict();

const fileSearchInput = z
	.strictObject({
		query: z.string(),
		root: z.string().trim().min(1).optional(),
		matchPath: z.boolean().optional(),
		matchCase: z.boolean().optional(),
		kind: z.enum(["files", "folders"]).optional(),
		maxResults: z.number().int().positive().max(1000).optional(),
		offset: z.number().int().nonnegative().optional(),
	})
	.strict();

const refreshInput = z
	.strictObject({
		force: z.boolean().optional(),
		methods: z
			.array(z.enum(["parsed", "minsync", "datasources", "jikji", "everything", "fsearch"]))
			.min(1)
			.optional(),
	})
	.strict();

const datasourceGetInput = z.strictObject({ datasourceId: z.string().trim().min(1) });
const reportInput = z.strictObject({
	query: z.string().trim().min(1),
	// Detailed report validation stays in validateReport so malformed reports
	// return a stable tool error rather than protocol-level invalid params.
	report: z.unknown(),
});
const evidenceInput = z.strictObject({
	sessionId: z.string().trim().min(1),
	resultNumber: z.number().int().positive().optional(),
});

// `overwrite`/`skipExisting` are deliberately not offered: through MCP a conversion never replaces a file.
// Filee numbers a colliding output name instead, so nothing is lost.
const convertInput = z.strictObject({
	inputs: z.array(z.string().trim().min(1)).min(1).max(500),
	to: z.string().trim().min(1),
	outputDir: z.string().trim().min(1).optional(),
	recursive: z.boolean().optional(),
	quality: z.number().int().min(1).max(100).optional(),
	keepDates: z.boolean().optional(),
});
const convertFormatsInput = z.strictObject({ from: z.string().trim().min(1).optional() });

/** Per-file rows past this are summarised, so a huge batch cannot flood the model's context. */
const MAX_CONVERT_FILES = 200;

interface ExactDuplicateGroup {
	readonly hash: string;
	readonly files: readonly string[];
}

function exactGroupsFromScan(scan: DupeyScanResult): ExactDuplicateGroup[] {
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

export interface AutoRAGMcpServerOptions {
	readonly readOnly?: boolean;
	readonly tools?: readonly string[];
	/** Test/embedded override; defaults to the current Node platform. */
	readonly platform?: NodeJS.Platform;
}

function jsonResult<T extends object>(value: T, isError = false) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
		structuredContent: value,
		...(isError ? { isError: true } : {}),
	};
}

/** The parsed window, or the tool error carrying the parse message verbatim. */
function windowOrError(
	input: { readonly since?: string | undefined; readonly until?: string | undefined },
	extra: Record<string, unknown>,
):
	| { readonly ok: true; readonly window: { readonly since?: number; readonly until?: number } }
	| { readonly ok: false; readonly error: ReturnType<typeof toolError> } {
	try {
		return { ok: true, window: parseTimeWindow(input) ?? {} };
	} catch (error) {
		if (!(error instanceof TimeWindowError)) throw error;
		return { ok: false, error: toolError("invalid-time-window", error.message, extra) };
	}
}

function toolError(code: string, message: string, extra: Record<string, unknown> = {}) {
	return jsonResult({ ok: false, errorCode: code, message, ...extra }, true);
}

/** Entries of one code kept verbatim; the rest of that code are summarized in one entry. */
const DIAGNOSTIC_EXAMPLES_PER_CODE = 3;

interface DiagnosticLike {
	readonly code: string;
	readonly severity: string;
	readonly message: string;
	readonly source?: string;
	readonly reason?: string;
}

/**
 * Keep an MCP response small when one code repeats once per file. A corpus whose parser changed reports one
 * `stale-index` entry per source (1,100 of them, about 750 KB, on a real 160k-file index), which the model then has to
 * read through on every search and status call. Each code keeps its first few entries verbatim and one summary entry
 * says how many more there were and how they split by `reason`, so "why is it stale" survives and a different code is
 * never hidden behind a flood of this one. Short lists pass through unchanged.
 */
export function capDiagnostics<T extends DiagnosticLike>(diagnostics: readonly T[]): (T | DiagnosticLike)[] {
	const byCode = new Map<string, T[]>();
	for (const entry of diagnostics) {
		const list = byCode.get(entry.code);
		if (list === undefined) byCode.set(entry.code, [entry]);
		else list.push(entry);
	}
	const out: (T | DiagnosticLike)[] = [];
	for (const [code, entries] of byCode) {
		out.push(...entries.slice(0, DIAGNOSTIC_EXAMPLES_PER_CODE));
		const rest = entries.slice(DIAGNOSTIC_EXAMPLES_PER_CODE);
		if (rest.length === 0) continue;
		const reasons = new Map<string, number>();
		for (const entry of rest) {
			const key = entry.reason ?? "unspecified";
			reasons.set(key, (reasons.get(key) ?? 0) + 1);
		}
		const split = [...reasons].map(([reason, count]) => `${reason}: ${count}`).join(", ");
		const severity = entries.some((entry) => entry.severity === "error")
			? "error"
			: entries.some((entry) => entry.severity === "warning")
				? "warning"
				: "info";
		out.push({
			code,
			severity,
			message: `${rest.length} more ${code} entries omitted (${split}); the first ${DIAGNOSTIC_EXAMPLES_PER_CODE} are shown above.`,
		});
	}
	return out;
}

/**
 * Shape a conversion for the model: one row per source with its outputs and error, plus the Dupey verdict per
 * output. Nothing converted is an error that still carries the per-file reasons, since "why" is what the model
 * needs next.
 */
function convertResult(details: ConvertAndRelateResult) {
	const { result, links } = details;
	const files = result.files.slice(0, MAX_CONVERT_FILES).map((file) => ({
		source: file.source,
		state: file.state,
		outputs: file.outputs,
		...(file.error !== undefined ? { error: file.error } : {}),
		...(file.notice !== undefined ? { notice: file.notice } : {}),
	}));
	const body = {
		target: result.target,
		converted: result.converted,
		skipped: result.skipped,
		failed: result.failed,
		files,
		...(result.files.length > files.length ? { omittedFiles: result.files.length - files.length } : {}),
		links,
	};
	return result.converted === 0 && result.failed > 0
		? toolError("convert-failed", "No file was converted.", body)
		: jsonResult({ ok: true, ...body });
}

function isToolEnabled(name: string, options: AutoRAGMcpServerOptions): boolean {
	return options.tools === undefined || options.tools.includes(name);
}

function readPackageVersion(): string {
	try {
		const manifest = JSON.parse(readFileSync(fileURLToPath(new URL("../../package.json", import.meta.url)), "utf8"));
		if (
			typeof manifest === "object" &&
			manifest !== null &&
			"version" in manifest &&
			typeof manifest.version === "string"
		) {
			return manifest.version;
		}
	} catch {
		// Development fallback when the package manifest is unavailable.
	}
	return "0.0.0-dev";
}

function escapeEverythingRegexLiteral(value: string): string {
	let escaped = "";
	for (const character of value) {
		if ("\\^$.*+?()[]{}|".includes(character)) escaped += "\\";
		escaped += character;
	}
	return escaped;
}

async function searchConfiguredFileNames(
	lite: AutoRAGLite,
	request: {
		readonly query: string;
		readonly root?: string;
		readonly matchPath?: boolean;
		readonly matchCase?: boolean;
		readonly kind?: "files" | "folders";
		readonly maxResults?: number;
		readonly offset?: number;
	},
	platform: NodeJS.Platform,
): Promise<object> {
	const scopedRoot =
		request.root === undefined
			? undefined
			: await resolveConfiguredFileSearchRoot(lite.config.searchPaths, request.root);
	if (request.root !== undefined && scopedRoot === undefined) {
		return {
			ok: true,
			backend: platform === "win32" ? "everything" : "fsearch-cli",
			results: [],
			truncated: false,
			diagnostics: [
				{
					code: "root-out-of-scope",
					severity: "warning",
					source: request.root,
					message: "The requested search root is not inside any configured search root.",
				},
			],
		};
	}
	if (platform === "darwin" || platform === "linux") {
		const maxResults = Math.min(request.maxResults ?? 100, 1000);
		const requestedEnd = (request.offset ?? 0) + maxResults + 1;
		const filtered: { path: string; type: "file" | "folder" }[] = [];
		let providerOffset = 0;
		let providerHasMore = true;
		let backend: "fsearch-cli" | "walk" = "fsearch-cli";
		let note: string | undefined;
		while (providerHasMore && filtered.length < requestedEnd) {
			const result = await lite.searchFsearch({
				...request,
				query: escapeEverythingRegexLiteral(request.query),
				regex: true,
				path: scopedRoot,
				offset: providerOffset,
				maxResults: maxResults + 1,
			});
			if (!result.ok) return { ...result, backend: "fsearch-cli" };
			backend = result.backend;
			note = result.note;
			const batch = result.results.map(({ path, type }) => ({ path, type }));
			const allowed = await filterFileNameSearchMatches(
				scopedRoot === undefined ? lite.config.searchPaths : [scopedRoot],
				batch,
				lite.config.excludePaths ?? [],
			);
			filtered.push(...allowed);
			providerOffset += batch.length;
			providerHasMore = batch.length === Math.min(maxResults + 1, 1000);
			if (batch.length === 0) break;
		}
		const offset = request.offset ?? 0;
		return {
			ok: true,
			backend,
			results: filtered.slice(offset, offset + maxResults),
			truncated: filtered.length > offset + maxResults || providerHasMore,
			diagnostics:
				note === undefined
					? []
					: [{ code: "fsearch-degraded", severity: "warning", source: "fsearch", message: note }],
		};
	}

	if (platform !== "win32") return searchFileNames(lite.config.searchPaths, request, lite.config.excludePaths ?? []);
	const maxResults = Math.min(request.maxResults ?? 100, 1000);
	const requestedEnd = (request.offset ?? 0) + maxResults + 1;
	const filtered: { path: string; type: "file" | "folder" }[] = [];
	let providerOffset = 0;
	let providerHasMore = true;
	while (providerHasMore && filtered.length < requestedEnd) {
		const result = await lite.searchEverything({
			query: escapeEverythingRegexLiteral(request.query),
			regex: true,
			matchPath: request.matchPath,
			matchCase: request.matchCase,
			kind: request.kind,
			path: scopedRoot,
			offset: providerOffset,
			maxResults: maxResults + 1,
		});
		if (!result.ok) return { ...result, backend: "everything" };
		const batch = result.results.map(({ path: resultPath, type }) => ({ path: resultPath, type }));
		const allowed = await filterFileNameSearchMatches(
			scopedRoot === undefined ? lite.config.searchPaths : [scopedRoot],
			batch,
			lite.config.excludePaths ?? [],
		);
		filtered.push(...allowed);
		providerOffset += batch.length;
		providerHasMore = batch.length === maxResults + 1;
		if (batch.length === 0) break;
	}
	const offset = request.offset ?? 0;
	return {
		ok: true,
		backend: "everything",
		results: filtered.slice(offset, offset + maxResults),
		truncated: filtered.length > offset + maxResults || providerHasMore,
		diagnostics: [],
	};
}

export function createAutoRAGMcpServer(lite: AutoRAGLite, options: AutoRAGMcpServerOptions = {}): McpServer {
	const server = new McpServer({ name: "autorag-lite", version: readPackageVersion() });
	const readOnly = options.readOnly === true;
	const platform = options.platform ?? process.platform;

	if (isToolEnabled("autorag.status", options)) {
		server.registerTool(
			"autorag.status",
			{
				title: "AutoRAG Index Status",
				description: "Return AutoRAG Lite index freshness and health without modifying the corpus.",
				inputSchema: emptyInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async () => {
				const status = await lite.getRefreshStatus();
				return jsonResult({ ...status, diagnostics: capDiagnostics(status.diagnostics) });
			},
		);
	}

	if (isToolEnabled("autorag.search", options)) {
		server.registerTool(
			"autorag.search",
			{
				title: "Search AutoRAG Lite Corpus",
				description:
					"Search selected configured retrieval methods and datasources. Selection narrows execution before any backend is invoked.",
				inputSchema: searchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, topK, scope, since, until, strict, datasourceIds, methods, local }) => {
				const window = windowOrError({ since, until }, { query });
				if (!window.ok) return window.error;
				const status = await lite.getRefreshStatus();
				if (!isParsedRefreshComplete(lite.config.workspacePath)) {
					return toolError(
						"index-not-ready",
						"Index has not been refreshed. Call autorag.refresh before searching.",
						{
							action: "autorag.refresh",
							query,
						},
					);
				}
				if (strict === true && status.stale) {
					return toolError(
						"stale-index",
						"The configured corpus is stale and strict search forbids answering from it.",
						{
							action: "autorag.refresh",
							query,
						},
					);
				}
				try {
					const retrieved = await lite.searchSelected(
						query,
						{ datasourceIds, methods, local },
						{ topK, scope, ...window.window },
					);
					return jsonResult({
						ok: true,
						query,
						stale: status.stale,
						selection: { datasourceIds, methods, local },
						results: retrieved.results.map((result, index) => ({
							number: index + 1,
							source: result.source,
							method: typeof result.metadata.method === "string" ? result.metadata.method : "unknown",
							score: result.score,
							metadata: result.metadata,
							content: result.content,
						})),
						unsearched: retrieved.unsearched,
						...(retrieved.window === undefined ? {} : { window: retrieved.window }),
						diagnostics: capDiagnostics([...status.diagnostics, ...retrieved.diagnostics]),
					});
				} catch (error) {
					return toolError(
						error instanceof RetrievalSelectionError ? "invalid-selection" : "search-failed",
						error instanceof Error ? error.message : String(error),
						{ retryable: !(error instanceof RetrievalSelectionError) },
					);
				}
			},
		);
	}

	const integratedDatasourceIds = new Set<string>();
	try {
		for (const method of lite.getRetrievalEngine().getMethodRegistry().list()) {
			const datasourceId = method.describe().datasourceId;
			if (datasourceId !== undefined) integratedDatasourceIds.add(datasourceId);
		}
	} catch {
		for (const entry of lite.listDatasources()) integratedDatasourceIds.add(entry.datasourceId);
	}
	for (const entry of lite.listDatasources()) {
		if (!integratedDatasourceIds.has(entry.datasourceId)) continue;
		const toolName = `autorag.${datasourceSearchToolName(entry.datasourceId)}`;
		if (!isToolEnabled(toolName, options)) continue;
		server.registerTool(
			toolName,
			{
				title: `Search ${entry.name}`,
				description:
					`Search only the integrated ${entry.name} datasource (${entry.datasourceId}): ${entry.description}. ` +
					"The datasource is server-configured; query, topK, scope, and since/until only narrow the search.",
				inputSchema: datasourceSearchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, topK, scope, since, until }) => {
				const window = windowOrError({ since, until }, { query, datasourceId: entry.datasourceId });
				if (!window.ok) return window.error;
				const status = await lite.getRefreshStatus();
				if (!isParsedRefreshComplete(lite.config.workspacePath)) {
					return toolError(
						"index-not-ready",
						"Index has not been refreshed. Call autorag.refresh before searching.",
						{
							action: "autorag.refresh",
							query,
							datasourceId: entry.datasourceId,
						},
					);
				}
				try {
					const retrieved = await lite.searchSelected(
						query,
						{ datasourceIds: [entry.datasourceId], local: false },
						{ topK, scope, ...window.window },
					);
					return jsonResult({
						ok: true,
						query,
						datasourceId: entry.datasourceId,
						stale: status.stale,
						results: retrieved.results.map((result, index) => ({
							number: index + 1,
							source: result.source,
							method: typeof result.metadata.method === "string" ? result.metadata.method : "unknown",
							score: result.score,
							metadata: result.metadata,
							content: result.content,
						})),
						unsearched: retrieved.unsearched,
						...(retrieved.window === undefined ? {} : { window: retrieved.window }),
						diagnostics: capDiagnostics([...status.diagnostics, ...retrieved.diagnostics]),
					});
				} catch (error) {
					return toolError(
						error instanceof RetrievalSelectionError ? "invalid-selection" : "search-failed",
						error instanceof Error ? error.message : String(error),
						{ datasourceId: entry.datasourceId, retryable: !(error instanceof RetrievalSelectionError) },
					);
				}
			},
		);
	}

	if (isToolEnabled("autorag.duplicates", options)) {
		server.registerTool(
			"autorag.duplicates",
			{
				title: "Scan Duplicate Documents",
				description:
					"Scan configured search roots with Dupey for exact duplicate files and near-duplicate document families. Review results before moving or deleting anything.",
				inputSchema: emptyInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async () => {
				try {
					if (lite.config.dupey?.enabled === false) {
						return toolError("duplicates-disabled", "Dupey duplicate scanning is disabled by configuration.");
					}
					const configuredDupey = lite.config.dupey as
						| (NonNullable<AutoRAGLite["config"]["dupey"]> & {
								readonly run?: (args: readonly string[]) => Promise<string>;
						  })
						| undefined;
					const dupeyOptions =
						configuredDupey === undefined
							? {}
							: {
									...(configuredDupey.binaryPath !== undefined
										? { executable: configuredDupey.binaryPath }
										: {}),
									...(configuredDupey.timeoutMs !== undefined ? { timeoutMs: configuredDupey.timeoutMs } : {}),
									...(configuredDupey.run !== undefined ? { run: configuredDupey.run } : {}),
								};
					const roots = lite.config.searchPaths.map((path: string) => resolve(path));
					const scans = await Promise.all(roots.map((root: string) => scanWithDupey(root, dupeyOptions)));
					return jsonResult({
						ok: true,
						roots,
						exactGroups: scans.flatMap(exactGroupsFromScan),
						families: scans.flatMap((scan) => scan.families),
						extractionErrors: scans.flatMap((scan) => scan.errors),
						action: "review",
					});
				} catch (error) {
					return toolError("duplicates-failed", error instanceof Error ? error.message : String(error), {
						retryable: true,
					});
				}
			},
		);
	}

	if (isToolEnabled("autorag.search.files", options)) {
		server.registerTool(
			"autorag.search.files",
			{
				title: "Search Configured File Names",
				description:
					"Search file and folder names under configured roots. Uses Windows Everything on Windows and the existing fsearch-cli/FSearch index on macOS/Linux; only FSearch binary absence degrades to a bounded filesystem walk. Never reads file contents.",
				inputSchema: fileSearchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, root, matchPath, matchCase, kind, maxResults, offset }) => {
				const result = await searchConfiguredFileNames(
					lite,
					{ query, root, matchPath, matchCase, kind, maxResults, offset },
					platform,
				);
				return jsonResult(result, Reflect.get(result, "ok") === false);
			},
		);
	}

	if (isToolEnabled("autorag.datasources.list", options)) {
		server.registerTool(
			"autorag.datasources.list",
			{
				title: "List Configured Datasources",
				description: "List the datasources connected under the server's configuration.",
				inputSchema: emptyInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async () => jsonResult({ ok: true, datasources: lite.listDatasources() }),
		);
	}

	if (isToolEnabled("autorag.datasources.get", options)) {
		server.registerTool(
			"autorag.datasources.get",
			{
				title: "Get Configured Datasource",
				description:
					"Return one configured datasource descriptor without credentials or private configuration metadata.",
				inputSchema: datasourceGetInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ datasourceId }) => {
				const datasource = lite.listDatasources().find((entry) => entry.datasourceId === datasourceId);
				return datasource === undefined
					? toolError("datasource-not-found", `No configured datasource named ${datasourceId}.`, { datasourceId })
					: jsonResult({ ok: true, datasource });
			},
		);
	}

	// Conversion is on unless the config turns Filee off (the same switch that removes the agent's convert_files).
	const convertEnabled = lite.config.filee?.enabled !== false;

	if (convertEnabled && !readOnly && isToolEnabled("autorag.convert", options)) {
		server.registerTool(
			"autorag.convert",
			{
				title: "Convert Files",
				description:
					"Convert local files to another format offline with Filee: documents, spreadsheets, slides, HWP/HWPX, images, e-books, archives, audio, video, data. Writes new files next to each source (or into outputDir) and never replaces an existing file: a colliding name gets a number added. Each output is compared with its source by Dupey, so the result says whether the copy would count as a duplicate. Use autorag.convert.formats when unsure what a source can become.",
				inputSchema: convertInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: false, idempotentHint: false, destructiveHint: false, openWorldHint: false },
			},
			async ({ inputs, to, outputDir, recursive, quality, keepDates }, ctx) => {
				try {
					const details = await lite.convertFiles(
						{
							inputs,
							to,
							...(outputDir !== undefined ? { outputDir } : {}),
							...(recursive !== undefined ? { recursive } : {}),
							...(quality !== undefined ? { quality } : {}),
							...(keepDates !== undefined ? { keepDates } : {}),
						},
						ctx.mcpReq.signal,
					);
					return convertResult(details);
				} catch (error) {
					return toolError("convert-failed", error instanceof Error ? error.message : String(error));
				}
			},
		);
	}

	if (convertEnabled && isToolEnabled("autorag.convert.formats", options)) {
		server.registerTool(
			"autorag.convert.formats",
			{
				title: "List Convertible Formats",
				description:
					"List the file formats Filee knows, or what one source format (heic, docx, hwp, mkv) can be converted to, including targets that need an optional engine. Read-only.",
				inputSchema: convertFormatsInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ from }, ctx) => {
				try {
					const formats = await lite.listFileConversions(from, ctx.mcpReq.signal);
					return jsonResult({ ok: true, ...(from !== undefined ? { from } : {}), formats });
				} catch (error) {
					return toolError("convert-failed", error instanceof Error ? error.message : String(error));
				}
			},
		);
	}

	if (!readOnly && isToolEnabled("autorag.report", options)) {
		server.registerTool(
			"autorag.report",
			{
				title: "Persist AutoRAG Curated Report",
				description:
					"Persist an externally curated Lite search report with opaque source mappings for later evidence inspection.",
				inputSchema: reportInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: false, idempotentHint: false, destructiveHint: false, openWorldHint: false },
			},
			async ({ query, report }) => {
				let details: ReturnType<typeof validateReport>;
				try {
					details = validateReport(report);
				} catch (error) {
					return toolError("invalid-report", error instanceof Error ? error.message : String(error));
				}
				try {
					const response = lite.recordReport(query, details);
					return jsonResult({
						ok: true,
						sessionId: response.sessionId,
						query: response.query,
						answer: response.answer,
						resultCount: response.results.length,
						results: response.results,
						...(response.diagnostics !== undefined && response.diagnostics.length > 0
							? { diagnostics: response.diagnostics }
							: {}),
					});
				} catch (error) {
					return toolError("report-failed", error instanceof Error ? error.message : String(error));
				}
			},
		);
	}

	if (isToolEnabled("autorag.evidence", options)) {
		server.registerTool(
			"autorag.evidence",
			{
				title: "Inspect AutoRAG Evidence",
				description: "Return persisted opaque sources and evidence chunks for a curated Lite report session.",
				inputSchema: evidenceInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ sessionId, resultNumber }) => {
				const evidence = evidenceFor(lite.getMemorySchema(), sessionId, resultNumber);
				return evidence.results.length === 0
					? toolError("evidence-not-found", `No evidence found for session ${sessionId}.`, { sessionId })
					: jsonResult({ ok: true, ...evidence });
			},
		);
	}

	if (!readOnly && isToolEnabled("autorag.refresh", options)) {
		server.registerTool(
			"autorag.refresh",
			{
				title: "Refresh AutoRAG Lite Index",
				description: "Refresh configured AutoRAG indexes. Refreshing never modifies source documents.",
				inputSchema: refreshInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: false, idempotentHint: true, destructiveHint: false, openWorldHint: false },
			},
			async ({ force, methods }) => {
				try {
					const result = await lite.refresh(
						force === true,
						methods === undefined ? undefined : { methods: methods as RefreshMethod[] },
					);
					return jsonResult({ ok: true, ...result });
				} catch (error) {
					return toolError("refresh-failed", error instanceof Error ? error.message : String(error), {
						retryable: true,
					});
				}
			},
		);
	}

	return server;
}
