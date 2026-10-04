import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { McpServer } from "@modelcontextprotocol/server";
import { z } from "zod";
import type { RefreshMethod } from "../agent/agent.ts";
import type { AutoRAGLite } from "../core.ts";
import type { EverythingSearchRequest } from "../everything/index.ts";
import { searchFileNames } from "../filesystem/name-search.ts";
import { isParsedRefreshComplete } from "../mirror/paths.ts";
import { RetrievalSelectionError } from "../retrieval/index.ts";

const emptyInput = z.strictObject({});
const objectOutput = z.looseObject({});

const searchInput = z
	.strictObject({
		query: z.string().trim().min(1),
		topK: z.number().int().positive().max(100).optional(),
		scope: z.string().trim().min(1).optional(),
		tags: z.array(z.string().trim().min(1)).optional(),
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

const everythingSearchInput = z
	.strictObject({
		query: z.string().trim().min(1),
		regex: z.boolean().optional(),
		matchCase: z.boolean().optional(),
		matchPath: z.boolean().optional(),
		wholeWord: z.boolean().optional(),
		kind: z.enum(["files", "folders"]).optional(),
		path: z.string().trim().min(1).optional(),
		sort: z
			.enum([
				"name-ascending",
				"name-descending",
				"path-ascending",
				"path-descending",
				"size-ascending",
				"size-descending",
				"date-modified-ascending",
				"date-modified-descending",
			])
			.optional(),
		offset: z.number().int().nonnegative().optional(),
		maxResults: z.number().int().positive().max(1000).optional(),
	})
	.strict();

const refreshInput = z
	.strictObject({
		force: z.boolean().optional(),
		methods: z
			.array(z.enum(["parsed", "minsync", "datasources", "jikji", "everything"]))
			.min(1)
			.optional(),
	})
	.strict();

const datasourceGetInput = z.strictObject({ datasourceId: z.string().trim().min(1) });

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
	return jsonResult({ ok: false, errorCode: code, message, ...extra }, true);
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
					"Search selected configured retrieval methods and datasources. Selection narrows execution before any backend is invoked.",
				inputSchema: searchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, topK, scope, tags, strict, datasourceIds, methods, local }) => {
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
						{ topK, scope, allowedTags: tags },
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
						diagnostics: [...status.diagnostics, ...retrieved.diagnostics],
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

	if (isToolEnabled("autorag.search.files", options)) {
		server.registerTool(
			"autorag.search.files",
			{
				title: "Search Configured File Names",
				description:
					"Search file and folder names under configured search roots without reading file contents. The optional root must remain inside a configured root.",
				inputSchema: fileSearchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ query, root, matchPath, matchCase, kind, maxResults, offset }) => {
				const result = await searchFileNames(
					lite.config.searchPaths,
					{ query, root, matchPath, matchCase, kind, maxResults, offset },
					lite.config.excludePaths ?? [],
				);
				return jsonResult(result, false);
			},
		);
	}

	if (isToolEnabled("autorag.search.everything", options)) {
		server.registerTool(
			"autorag.search.everything",
			{
				title: "Search Everything File Index",
				description:
					"Search the Windows-only user-level Everything index. Returns an unsupported-platform result elsewhere.",
				inputSchema: everythingSearchInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async (request) => jsonResult(await lite.searchEverything(request as EverythingSearchRequest)),
		);
	}

	if (isToolEnabled("autorag.datasources.list", options)) {
		server.registerTool(
			"autorag.datasources.list",
			{
				title: "List Authorized Datasources",
				description: "List configured datasources visible under the server's default-deny authorization policy.",
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
				title: "Get Authorized Datasource",
				description:
					"Return one authorized datasource descriptor without credentials or private configuration metadata.",
				inputSchema: datasourceGetInput,
				outputSchema: objectOutput,
				annotations: { readOnlyHint: true, openWorldHint: false },
			},
			async ({ datasourceId }) => {
				const datasource = lite.listDatasources().find((entry) => entry.datasourceId === datasourceId);
				return datasource === undefined
					? toolError("datasource-not-found", `No authorized datasource named ${datasourceId}.`, { datasourceId })
					: jsonResult({ ok: true, datasource });
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

	return server;
}
