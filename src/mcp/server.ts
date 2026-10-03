import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";
import type { RefreshMethod } from "../agent/agent.ts";
import type { AutoRAGResultsDetails } from "../agent/emit-results-tool.ts";
import { validateReport } from "../cli/commands/report.ts";
import type { AutoRAGLite } from "../core.ts";
import { type DupeyScanResult, scanWithDupey } from "../dupey/index.ts";
import { isParsedRefreshComplete } from "../mirror/paths.ts";

const emptyInput = z.strictObject({});
const objectOutput = z.looseObject({});

const searchInput = z
	.object({
		query: z.string().trim().min(1),
		topK: z.number().int().positive().max(100).optional(),
		scope: z.string().trim().min(1).optional(),
		tags: z.array(z.string().trim().min(1)).optional(),
		strict: z.boolean().optional(),
	})
	.strict();

const refreshInput = z
	.object({
		force: z.boolean().optional(),
		methods: z
			.array(z.enum(["parsed", "minsync", "datasources", "jikji", "everything"]))
			.min(1)
			.optional(),
	})
	.strict();

const reportInput = z
	.strictObject({
		query: z.string().trim().min(1),
		report: z
			.looseObject({
				answer: z.string(),
				results: z.array(z.record(z.string(), z.unknown())),
				mapping: z.array(z.record(z.string(), z.unknown())),
				warnings: z.array(z.string()).optional(),
			})
			.passthrough(),
	})
	.strict();

const evidenceInput = z
	.object({
		sessionId: z.string().trim().min(1),
		resultNumber: z.number().int().positive().optional(),
	})
	.strict();

const feedbackInput = z
	.object({
		sessionId: z.string().trim().min(1),
		usefulNumbers: z.array(z.number().int().positive()).default([]),
		notUsefulNumbers: z.array(z.number().int().positive()).default([]),
	})
	.strict()
	.refine((value) => value.usefulNumbers.length > 0 || value.notUsefulNumbers.length > 0, {
		message: "At least one usefulNumbers or notUsefulNumbers item is required",
	});

export interface AutoRAGMcpServerOptions {
	readonly readOnly?: boolean;
	readonly tools?: readonly string[];
}

function jsonResult<T extends object>(value: T, isError = false) {
	return {
		content: [{ type: "text" as const, text: JSON.stringify(value, null, 2) }],
		structuredContent: value,
		...(isError ? { isError: true } : {}),
	};
}

function toolError(code: string, message: string, extra: Record<string, unknown> = {}) {
	return jsonResult(
		{
			ok: false,
			errorCode: code,
			message,
			...extra,
		},
		true,
	);
}

function isToolEnabled(name: string, options: AutoRAGMcpServerOptions): boolean {
	return options.tools === undefined || options.tools.includes(name);
}

function evidenceFor(
	schema: ReturnType<AutoRAGLite["getMemorySchema"]>,
	sessionId: string,
	resultNumber?: number,
): Record<string, unknown> {
	const results = schema.curatedResults
		.filter((result) => result.sessionId === sessionId)
		.filter((result) => resultNumber === undefined || result.number === resultNumber)
		.sort((a, b) => a.number - b.number)
		.map((result) => ({
			number: result.number,
			query: result.query,
			confidence: result.confidence,
			chunks: result.evidenceIds
				.map((id) => schema.evidenceChunks.find((chunk) => chunk.stableEvidenceId === id))
				.filter((chunk) => chunk !== undefined),
		}));
	return { sessionId, results };
}

interface ExactGroup {
	readonly hash: string;
	readonly files: readonly string[];
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
/**
 * The installed package version. Mirrors the CLI entrypoint's resolution:
 * `src/mcp/server.ts` and the published `dist/mcp/server.js` both resolve
 * `../../package.json` to the package root.
 */
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
		// Fall through to the dev fallback below.
	}
	return "0.0.0-dev";
}

export function createAutoRAGMcpServer(lite: AutoRAGLite, options: AutoRAGMcpServerOptions = {}): McpServer {
	const server = new McpServer({ name: "autorag-lite", version: readPackageVersion() });
	const readOnly = options.readOnly === true;

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
			async () => jsonResult(await lite.getRefreshStatus()),
		);
	}

	if (isToolEnabled("autorag.search", options)) {
		server.registerTool(
			"autorag.search",
			{
				title: "Search AutoRAG Lite Corpus",
				description:
					"Search the configured AutoRAG Lite corpus without refreshing or modifying indexes. Check stale, diagnostics, and unsearched before trusting the result set.",
				inputSchema: searchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, topK, scope, tags, strict }) => {
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
					const retrieved = await lite.retrieve(query, {
						topK,
						scope,
						allowedTags: tags,
					});
					return jsonResult({
						ok: true,
						query,
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
						diagnostics: [...status.diagnostics, ...retrieved.diagnostics],
					});
				} catch (error) {
					return toolError("search-failed", error instanceof Error ? error.message : String(error), {
						retryable: true,
					});
				}
			},
		);
	}

	if (!readOnly && isToolEnabled("autorag.refresh", options)) {
		server.registerTool(
			"autorag.refresh",
			{
				title: "Refresh AutoRAG Lite Index",
				description: "Refresh configured AutoRAG indexes. Source documents are never modified.",
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

	if (!readOnly && isToolEnabled("autorag.report", options)) {
		server.registerTool(
			"autorag.report",
			{
				title: "Persist AutoRAG Curated Report",
				description:
					"Persist an externally curated report with exact source and retrieval-method mappings for later evidence and feedback.",
				inputSchema: reportInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: false, idempotentHint: false, destructiveHint: false, openWorldHint: false },
			},
			async ({ query, report }) => {
				try {
					const details = validateReport(report) as AutoRAGResultsDetails;
					const response = lite.recordReport(query, details);
					return jsonResult({
						ok: true,
						sessionId: response.sessionId,
						query: response.query,
						answer: response.answer,
						resultCount: response.results.length,
					});
				} catch (error) {
					return toolError("invalid-report", error instanceof Error ? error.message : String(error));
				}
			},
		);
	}

	if (isToolEnabled("autorag.evidence", options)) {
		server.registerTool(
			"autorag.evidence",
			{
				title: "Read AutoRAG Evidence",
				description:
					"Read persisted evidence attached to a report session. This does not read arbitrary filesystem paths.",
				inputSchema: evidenceInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ sessionId, resultNumber }) => {
				const view = evidenceFor(lite.getMemorySchema(), sessionId, resultNumber);
				if ((view.results as unknown[]).length === 0) {
					return toolError("session-not-found", `No evidence found for session ${sessionId}.`, { sessionId });
				}
				return jsonResult(view);
			},
		);
	}

	if (!readOnly && isToolEnabled("autorag.feedback", options)) {
		server.registerTool(
			"autorag.feedback",
			{
				title: "Record AutoRAG Feedback",
				description:
					"Record useful or not-useful signals for numbered results in a persisted AutoRAG report session.",
				inputSchema: feedbackInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: false, idempotentHint: true, destructiveHint: false, openWorldHint: false },
			},
			async ({ sessionId, usefulNumbers, notUsefulNumbers }) => {
				if (usefulNumbers.some((number) => notUsefulNumbers.includes(number))) {
					return toolError("invalid-feedback", "A result number cannot be both useful and not useful.");
				}
				const applied = lite.recordPersistedFeedbackByNumbers(sessionId, usefulNumbers, notUsefulNumbers);
				return applied
					? jsonResult({ ok: true, applied: true, sessionId, usefulNumbers, notUsefulNumbers })
					: toolError("session-not-found", `No matching report results found for session ${sessionId}.`, {
							sessionId,
						});
			},
		);
	}

	if (isToolEnabled("autorag.duplicates", options)) {
		server.registerTool(
			"autorag.duplicates",
			{
				title: "Scan AutoRAG Duplicate Documents",
				description:
					"Scan configured search roots for duplicate document families. This tool never moves, renames, or deletes source files.",
				inputSchema: emptyInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async () => {
				try {
					const roots = lite.config.searchPaths.map((path) => resolve(path));
					const scans = await Promise.all(roots.map((root) => scanWithDupey(root)));
					return jsonResult({
						ok: true,
						roots,
						exactGroups: scans.flatMap(exactGroupsFromScan),
						families: scans.flatMap((scan) => scan.families),
						extractionErrors: scans.flatMap((scan) => scan.errors),
						action: "review",
					});
				} catch (error) {
					return toolError("duplicates-failed", error instanceof Error ? error.message : String(error));
				}
			},
		);
	}

	return server;
}
