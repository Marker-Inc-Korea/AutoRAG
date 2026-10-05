import { mkdirSync, mkdtempSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { createAutoRAGMcpServer } from "../../src/mcp/server.ts";

const roots: string[] = [];

afterEach(() => {
	while (roots.length > 0) {
		const root = roots.pop();
		if (root !== undefined) rmSync(root, { recursive: true, force: true });
	}
});

/** A workspace whose parsed refresh has (not) completed, per the readiness marker. */
function workspace(ready: boolean): string {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-server-"));
	roots.push(root);
	if (ready) {
		mkdirSync(join(root, ".autorag"), { recursive: true });
		writeFileSync(
			join(root, ".autorag", "refresh-complete.json"),
			JSON.stringify({ version: 1, completed: true, parsed: true }),
		);
	}
	return root;
}

const datasource = {
	datasourceId: "kakao",
	name: "kakao",
	type: "chat",
	description: "카카오톡 대화",
	tags: ["kakao"],
	capabilities: ["keyword"],
	status: "active",
	sourceScopes: ["/kakao/default"],
} as const;

interface SearchCall {
	readonly query: string;
	readonly selection: unknown;
	readonly options: unknown;
}

interface EverythingRequest {
	readonly query: string;
	readonly regex?: boolean;
	readonly matchCase?: boolean;
	readonly matchPath?: boolean;
	readonly kind?: "files" | "folders";
	readonly path?: string;
	readonly offset?: number;
	readonly maxResults?: number;
}

interface FSearchRequest {
	readonly query: string;
	readonly path?: string;
	readonly matchPath?: boolean;
	readonly matchCase?: boolean;
	readonly kind?: "files" | "folders";
	readonly offset?: number;
	readonly maxResults?: number;
}

interface FSearchEntry {
	readonly path: string;
	readonly type: "file" | "folder";
}

interface FSearchCall {
	readonly request: FSearchRequest;
}

interface EverythingEntry {
	readonly path: string;
	readonly type: "file" | "folder";
}

interface EverythingCall {
	readonly request: EverythingRequest;
}

interface FakeOptions {
	readonly workspacePath: string;
	readonly searchPaths?: readonly string[];
	readonly excludePaths?: readonly string[];
	/** Fixed Everything provider result (success or failure). */
	readonly everything?: unknown;
	/** Dynamic Everything provider seam; wins over `everything` when both are set. */
	readonly everythingProvider?: (request: EverythingRequest) => unknown | Promise<unknown>;
	/** Dynamic FSearch provider seam; wins over the default result. */
	readonly fsearchProvider?: (request: FSearchRequest) => unknown | Promise<unknown>;
	/** Injected Dupey scanner seam; the server forwards `config.dupey` verbatim. */
	readonly dupey?: unknown;
	readonly datasources?: readonly unknown[];
	/** Datasource ids with a registered retrieval method (dynamic tool exposure). */
	readonly retrievalDatasourceIds?: readonly string[];
}

function fakeLite(options: FakeOptions): {
	lite: AutoRAGLite;
	searchCalls: SearchCall[];
	everythingCalls: EverythingCall[];
	fsearchCalls: FSearchCall[];
} {
	const searchCalls: SearchCall[] = [];
	const everythingCalls: EverythingCall[] = [];
	const fsearchCalls: FSearchCall[] = [];
	// Unchecked cast: the fake implements only the methods the MCP surface calls.
	const lite = {
		config: {
			searchPaths: options.searchPaths ?? [],
			excludePaths: options.excludePaths ?? [],
			workspacePath: options.workspacePath,
			memoryPath: join(options.workspacePath, "memory.json"),
			dupey: options.dupey ?? {
				run: async () => JSON.stringify({ dir: "", files: [], families: [], errors: [] }),
			},
		},
		getRefreshStatus: async () => ({
			state: "success",
			inFlight: false,
			stale: false,
			diagnostics: [],
			components: {},
		}),
		getRetrievalEngine: () => ({
			getMethodRegistry: () => ({
				list: () =>
					(options.retrievalDatasourceIds ?? ["kakao"]).map((datasourceId) => ({
						describe: () => ({
							name: `${datasourceId}-lexical`,
							type: "posix",
							description: "stub datasource retrieval method",
							status: "active",
							capabilities: [],
							datasourceId,
						}),
					})),
			}),
		}),
		listDatasources: () => options.datasources ?? [datasource],
		searchSelected: async (query: string, selection: unknown, retrievalOptions: unknown) => {
			searchCalls.push({ query, selection, options: retrievalOptions });
			return {
				results: [
					{
						id: "minsync:1",
						source: "docs/refund-policy.md",
						score: 0.9,
						metadata: { method: "minsync" },
						content: "Refund exceptions require director approval before payout.",
					},
				],
				diagnostics: [],
				unsearched: [],
			};
		},
		searchEverything: async (request: unknown) => {
			const typed = request as EverythingRequest;
			everythingCalls.push({ request: typed });
			if (options.everythingProvider !== undefined) return await options.everythingProvider(typed);
			return options.everything ?? { ok: true, results: [] };
		},
		searchFsearch: async (request: unknown) => {
			const typed = request as FSearchRequest;
			fsearchCalls.push({ request: typed });
			if (options.fsearchProvider !== undefined) return await options.fsearchProvider(typed);
			return { ok: true, backend: "fsearch-cli", results: [] };
		},
		refresh: async () => ({ ok: true }),
	} as unknown as AutoRAGLite;
	return { lite, searchCalls, everythingCalls, fsearchCalls };
}

/**
 * A fake Everything provider that serves a fixed universe lazily: it slices by
 * the request's own `offset`/`maxResults`, so the server must page correctly to
 * take a stable page after its own filtering.
 */
function pagingEverything(universe: readonly EverythingEntry[]) {
	return (request: EverythingRequest) => {
		const offset = request.offset ?? 0;
		const max = request.maxResults ?? universe.length;
		return { ok: true, results: universe.slice(offset, offset + max) };
	};
}

function pagingFsearch(universe: readonly FSearchEntry[]) {
	return (request: FSearchRequest) => {
		const offset = request.offset ?? 0;
		const max = request.maxResults ?? universe.length;
		return {
			ok: true,
			backend: "fsearch-cli",
			total: universe.length,
			results: universe.slice(offset, offset + max),
		};
	};
}

async function connectedServer(lite: AutoRAGLite, serverOptions = {}) {
	const server = createAutoRAGMcpServer(lite, serverOptions);
	const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
	await server.connect(serverTransport);
	const client = new Client({ name: "autorag-mcp-test", version: "1.0.0" });
	await client.connect(clientTransport);
	return { client, server };
}

function field(value: unknown, key: string): unknown {
	if (typeof value !== "object" || value === null) return undefined;
	return Reflect.get(value, key);
}

/** Recursively search a structured result for a `key === expected` pair, body-shape agnostic. */
function hasFieldValue(value: unknown, key: string, expected: unknown, seen = new Set<unknown>()): boolean {
	if (typeof value !== "object" || value === null) return false;
	if (seen.has(value)) return false;
	seen.add(value);
	if (field(value, key) === expected) return true;
	return Object.values(value).some((item) => hasFieldValue(item, key, expected, seen));
}

/** Extract `{ path, type }` entries from a tool result's `results` array, ignoring other fields. */
function resultEntries(value: unknown): { path: string; type: string }[] {
	const results = field(value, "results");
	if (!Array.isArray(results)) return [];
	return results
		.filter((entry): entry is Record<string, unknown> => typeof entry === "object" && entry !== null)
		.map((entry) => ({ path: String(entry.path), type: String(entry.type) }));
}

describe("AutoRAG Lite MCP server", () => {
	it("exposes the Lite tools with stable names", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect([...names].sort()).toEqual(
			[
				"autorag.status",
				"autorag.search",
				"autorag.search.files",
				"autorag.datasources.list",
				"autorag.datasources.get",
				"autorag.duplicates",
				"autorag.search_datasource_kakao",
				"autorag.refresh",
				"autorag.report",
				"autorag.evidence",
				"autorag.feedback",
			].sort(),
		);
		await client.close();
		await server.close();
	});

	it("publishes a description on every tool", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const { tools } = await client.listTools();
		for (const tool of tools) {
			expect(typeof tool.description).toBe("string");
			expect(tool.description?.length ?? 0).toBeGreaterThan(0);
		}
		const duplicates = tools.find((tool) => tool.name === "autorag.duplicates");
		expect(duplicates?.description).toContain("Dupey");
		await client.close();
		await server.close();
	});

	it("omits mutating tools in read-only mode", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite, { readOnly: true });
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect(names).not.toContain("autorag.refresh");
		expect(names).not.toContain("autorag.report");
		expect(names).not.toContain("autorag.feedback");
		expect(names).toContain("autorag.search");
		expect(names).toContain("autorag.evidence");
		expect(names).toContain("autorag.search.files");
		expect(names).toContain("autorag.datasources.list");
		expect(names).toContain("autorag.datasources.get");
		expect(names).toContain("autorag.duplicates");
		expect(names).toContain("autorag.search_datasource_kakao");
		await server.close();
	});

	it("returns structured status output", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.status", arguments: {} });
		expect(result.isError).not.toBe(true);
		expect(result.structuredContent).toMatchObject({ state: "success", stale: false });
		await client.close();
		await server.close();
	});

	it("maps search selection and retrieval options onto searchSelected", async () => {
		const { lite, searchCalls } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({
			name: "autorag.search",
			arguments: {
				query: "refund approval",
				topK: 5,
				scope: "/kakao/default",
				tags: ["kakao"],
				strict: true,
				datasourceIds: ["kakao"],
				methods: ["kakao.keyword"],
				local: false,
			},
		});
		expect(result.isError).not.toBe(true);
		expect(searchCalls).toHaveLength(1);
		expect(searchCalls[0]).toMatchObject({
			query: "refund approval",
			selection: { datasourceIds: ["kakao"], methods: ["kakao.keyword"], local: false },
			options: { topK: 5, scope: "/kakao/default", allowedTags: ["kakao"] },
		});
		expect(hasFieldValue(result.structuredContent, "source", "docs/refund-policy.md")).toBe(true);
		await client.close();
		await server.close();
	});

	it("returns an actionable index-not-ready error before refreshing", async () => {
		const { lite, searchCalls } = fakeLite({ workspacePath: workspace(false) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.search", arguments: { query: "hello" } });
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({
			errorCode: "index-not-ready",
			action: "autorag.refresh",
		});
		expect(searchCalls).toHaveLength(0);
		await client.close();
		await server.close();
	});

	it("routes macOS file search through FSearch", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-docs-")));
		roots.push(docs);
		const refund = join(docs, "refund-policy.md");
		writeFileSync(refund, "Director approval is required.\n");
		const { lite, fsearchCalls } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			fsearchProvider: async () => ({
				ok: true,
				backend: "fsearch-cli",
				results: [{ path: refund, type: "file" }],
				total: 1,
			}),
		});
		const { client, server } = await connectedServer(lite, { platform: "darwin" });
		const result = await client.callTool({ name: "autorag.search.files", arguments: { query: "refund" } });
		expect(result.isError).not.toBe(true);
		expect(result.structuredContent).toMatchObject({ ok: true, backend: "fsearch-cli" });
		expect(hasFieldValue(result.structuredContent, "path", refund)).toBe(true);
		expect(fsearchCalls).toHaveLength(1);
		expect(fsearchCalls[0]?.request).toMatchObject({ query: "refund" });
		await client.close();
		await server.close();
	});

	it("escapes literal query punctuation into an Everything regex on Windows", async () => {
		const { lite, everythingCalls } = fakeLite({
			workspacePath: workspace(true),
			everything: { ok: true, results: [] },
		});
		const { client, server } = await connectedServer(lite, { platform: "win32" });
		const result = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "a.b", matchPath: true, matchCase: true },
		});
		expect(result.isError).not.toBe(true);
		expect(result.structuredContent).toMatchObject({ ok: true, backend: "everything" });
		expect(everythingCalls).toHaveLength(1);
		const request = everythingCalls[0]?.request;
		expect(request).toMatchObject({ regex: true, matchPath: true, matchCase: true });
		expect(typeof request?.query).toBe("string");
		// Proves literal semantics: the escaped pattern matches the literal ".", not a wildcard.
		const pattern = new RegExp(request?.query ?? "");
		expect(pattern.test("a.b")).toBe(true);
		expect(pattern.test("axb")).toBe(false);
		await client.close();
		await server.close();
	});

	it("routes Windows file search through Everything and drops denied results", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-win-")));
		roots.push(docs);
		const keep = join(docs, "keep.txt");
		const privateDir = join(docs, "private");
		const secret = join(privateDir, "secret.txt");
		const folder = join(docs, "keep-folder");
		mkdirSync(privateDir, { recursive: true });
		mkdirSync(folder, { recursive: true });
		writeFileSync(keep, "keep\n");
		writeFileSync(secret, "secret\n");
		const { lite, everythingCalls } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			excludePaths: [privateDir],
			everything: {
				ok: true,
				results: [
					{ path: keep, type: "file" },
					{ path: secret, type: "file" },
					{ path: folder, type: "folder" },
				],
			},
		});
		const { client, server } = await connectedServer(lite, { platform: "win32" });
		const result = await client.callTool({ name: "autorag.search.files", arguments: { query: "keep" } });
		expect(result.isError).not.toBe(true);
		expect(result.structuredContent).toMatchObject({ ok: true, backend: "everything" });
		expect(everythingCalls).toHaveLength(1);
		const entries = resultEntries(result.structuredContent).sort((a, b) => a.path.localeCompare(b.path));
		expect(entries).toEqual(
			[
				{ path: keep, type: "file" },
				{ path: folder, type: "folder" },
			].sort((a, b) => a.path.localeCompare(b.path)),
		);
		expect(entries.some((entry) => entry.path === secret)).toBe(false);
		await client.close();
		await server.close();
	});

	it("rejects a Windows search root outside the configured roots without invoking the provider", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-win-")));
		const outside = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-out-")));
		roots.push(docs, outside);
		const { lite, everythingCalls } = fakeLite({ workspacePath: workspace(true), searchPaths: [docs] });
		const { client, server } = await connectedServer(lite, { platform: "win32" });
		const result = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "keep", root: outside },
		});
		expect(everythingCalls).toHaveLength(0);
		expect(resultEntries(result.structuredContent)).toEqual([]);
		expect(result.structuredContent).toMatchObject({ backend: "everything" });
		expect(hasFieldValue(result.structuredContent, "code", "root-out-of-scope")).toBe(true);
		await client.close();
		await server.close();
	});

	it("surfaces an Everything backend failure as an MCP error without filesystem fallback", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-win-")));
		roots.push(docs);
		const { lite, everythingCalls } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			everything: { ok: false, reason: "search-failed", message: "Everything exploded" },
		});
		const { client, server } = await connectedServer(lite, { platform: "win32" });
		const result = await client.callTool({ name: "autorag.search.files", arguments: { query: "refund" } });
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({
			ok: false,
			backend: "everything",
			reason: "search-failed",
			message: "Everything exploded",
		});
		expect(everythingCalls).toHaveLength(1);
		await client.close();
		await server.close();
	});

	it("paginates FSearch results with a lookahead slot", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-page-")));
		roots.push(docs);
		for (const name of ["report-1.txt", "report-2.txt", "report-3.txt"]) writeFileSync(join(docs, name), "x\n");
		const { lite } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			fsearchProvider: pagingFsearch(
				["report-1.txt", "report-2.txt", "report-3.txt"].map((name) => ({
					path: join(docs, name),
					type: "file" as const,
				})),
			),
		});
		const { client, server } = await connectedServer(lite, { platform: "darwin" });

		const first = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "report", maxResults: 2 },
		});
		expect(resultEntries(first.structuredContent).map((entry) => entry.path)).toEqual([
			join(docs, "report-1.txt"),
			join(docs, "report-2.txt"),
		]);
		expect(field(first.structuredContent, "truncated")).toBe(true);

		const rest = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "report", maxResults: 2, offset: 2 },
		});
		expect(resultEntries(rest.structuredContent).map((entry) => entry.path)).toEqual([join(docs, "report-3.txt")]);
		expect(field(rest.structuredContent, "truncated")).toBe(false);

		const pastEnd = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "report", maxResults: 2, offset: 3 },
		});
		expect(resultEntries(pastEnd.structuredContent)).toEqual([]);
		expect(field(pastEnd.structuredContent, "truncated")).toBe(false);
		await client.close();
		await server.close();
	});

	it("excludes configured paths, internal directories, and symlinks from FSearch results", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-excl-")));
		roots.push(docs);
		const keep = join(docs, "report-keep.txt");
		writeFileSync(keep, "keep\n");
		const privateDir = join(docs, "private");
		mkdirSync(privateDir, { recursive: true });
		writeFileSync(join(privateDir, "report-secret.txt"), "secret\n");
		const gitDir = join(docs, ".git");
		mkdirSync(gitDir, { recursive: true });
		writeFileSync(join(gitDir, "report-git.txt"), "internal\n");
		let symlinked = false;
		try {
			symlinkSync(keep, join(docs, "report-link.txt"));
			symlinked = true;
		} catch {
			// Symlink creation needs privileges on Windows; the other exclusions still apply.
		}
		const { lite } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			excludePaths: [privateDir],
			fsearchProvider: async () => ({
				ok: true,
				backend: "fsearch-cli",
				results: [
					{ path: keep, type: "file" },
					{ path: join(privateDir, "report-secret.txt"), type: "file" },
					{ path: join(gitDir, "report-git.txt"), type: "file" },
					...(symlinked ? [{ path: join(docs, "report-link.txt"), type: "file" as const }] : []),
				],
			}),
		});
		const { client, server } = await connectedServer(lite, { platform: "darwin" });
		const result = await client.callTool({ name: "autorag.search.files", arguments: { query: "report" } });
		const paths = resultEntries(result.structuredContent).map((entry) => entry.path);
		expect(paths).toEqual([keep]);
		if (symlinked) expect(paths).not.toContain(join(docs, "report-link.txt"));
		await client.close();
		await server.close();
	});

	it("pages Windows Everything results after dropping denied entries", async () => {
		const docs = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-winpage-")));
		roots.push(docs);
		const privateDir = join(docs, "private");
		mkdirSync(privateDir, { recursive: true });
		const denied = join(privateDir, "denied.txt");
		writeFileSync(denied, "denied\n");
		const keep1 = join(docs, "keep-1.txt");
		const keep2 = join(docs, "keep-2.txt");
		const keep3 = join(docs, "keep-3.txt");
		for (const path of [keep1, keep2, keep3]) writeFileSync(path, "keep\n");
		const { lite } = fakeLite({
			workspacePath: workspace(true),
			searchPaths: [docs],
			excludePaths: [privateDir],
			everythingProvider: pagingEverything([
				{ path: denied, type: "file" },
				{ path: keep1, type: "file" },
				{ path: keep2, type: "file" },
				{ path: keep3, type: "file" },
			]),
		});
		const { client, server } = await connectedServer(lite, { platform: "win32" });

		const first = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "keep", maxResults: 2 },
		});
		expect(resultEntries(first.structuredContent).map((entry) => entry.path)).toEqual([keep1, keep2]);
		expect(field(first.structuredContent, "truncated")).toBe(true);

		const rest = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "keep", maxResults: 2, offset: 2 },
		});
		expect(resultEntries(rest.structuredContent).map((entry) => entry.path)).toEqual([keep3]);
		expect(field(rest.structuredContent, "truncated")).toBe(false);
		await client.close();
		await server.close();
	});

	it("lists the authorized datasource catalog", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.datasources.list", arguments: {} });
		expect(result.isError).not.toBe(true);
		expect(hasFieldValue(result.structuredContent, "description", datasource.description)).toBe(true);
		await client.close();
		await server.close();
	});

	it("gets one datasource and rejects unknown ids", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const found = await client.callTool({
			name: "autorag.datasources.get",
			arguments: { datasourceId: "kakao" },
		});
		expect(found.isError).not.toBe(true);
		expect(hasFieldValue(found.structuredContent, "datasourceId", "kakao")).toBe(true);

		const missing = await client.callTool({
			name: "autorag.datasources.get",
			arguments: { datasourceId: "missing" },
		});
		expect(missing.isError).toBe(true);
		await client.close();
		await server.close();
	});

	it("validates search input before execution", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.search", arguments: {} });
		expect(result.isError).toBe(true);
		expect(result.content[0]).toMatchObject({ type: "text" });
		await client.close();
		await server.close();
	});

	it("groups exact duplicates and reports families from the configured Dupey scanner", async () => {
		const docs = mkdtempSync(join(tmpdir(), "autorag-mcp-dupes-"));
		roots.push(docs);
		const runCalls: string[][] = [];
		const scan = {
			dir: resolve(docs),
			files: [
				{ path: join(docs, "a.md"), content_hash: "hash-a" },
				{ path: join(docs, "b.md"), content_hash: "hash-a" },
				{ path: join(docs, "c.md"), content_hash: "hash-c" },
			],
			families: [{ id: 1, relation: "contains", files: [join(docs, "a.md")], members: [], edges: [] }],
			errors: [{ path: join(docs, "broken.pdf"), message: "extract failed" }],
		};
		const { lite } = fakeLite({
			workspacePath: docs,
			searchPaths: [docs],
			dupey: {
				run: async (args: readonly string[]) => {
					runCalls.push([...args]);
					return JSON.stringify(scan);
				},
			},
		});
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(result.isError).not.toBe(true);
		const output = result.structuredContent as Record<string, unknown>;
		expect(output.ok).toBe(true);
		expect(output.action).toBe("review");
		expect(output.roots).toEqual([resolve(docs)]);
		expect(output.exactGroups).toEqual([{ hash: "hash-a", files: [join(docs, "a.md"), join(docs, "b.md")].sort() }]);
		expect(hasFieldValue(output, "relation", "contains")).toBe(true);
		expect(hasFieldValue(output, "message", "extract failed")).toBe(true);
		expect(runCalls).toEqual([["scan", resolve(docs), "--json"]]);
		await client.close();
		await server.close();
	});

	it("returns duplicates-failed when the Dupey scanner fails", async () => {
		const docs = mkdtempSync(join(tmpdir(), "autorag-mcp-dupes-"));
		roots.push(docs);
		const { lite } = fakeLite({
			workspacePath: docs,
			searchPaths: [docs],
			dupey: {
				run: async () => {
					throw new Error("dupey exploded");
				},
			},
		});
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({ errorCode: "duplicates-failed" });
		await client.close();
		await server.close();
	});

	it("exposes dynamic datasource tools only for datasources with a registered retrieval method", async () => {
		const slack = { ...datasource, datasourceId: "slack", name: "slack", description: "Slack workspace" };
		const { lite } = fakeLite({
			workspacePath: workspace(true),
			datasources: [datasource, slack],
			retrievalDatasourceIds: ["kakao"],
		});
		const { client, server } = await connectedServer(lite);
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect(names).toContain("autorag.search_datasource_kakao");
		expect(names).not.toContain("autorag.search_datasource_slack");
		const dynamic = tools.find((tool) => tool.name === "autorag.search_datasource_kakao");
		expect(dynamic?.description).toContain(datasource.description);
		await client.close();
		await server.close();
	});

	it("routes a dynamic datasource search with only that datasource", async () => {
		const { lite, searchCalls } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({
			name: "autorag.search_datasource_kakao",
			arguments: { query: "refund", topK: 4, scope: "/kakao/default" },
		});
		expect(result.isError).not.toBe(true);
		expect(searchCalls).toHaveLength(1);
		expect(searchCalls[0]).toMatchObject({
			query: "refund",
			selection: { datasourceIds: ["kakao"], local: false },
			options: { topK: 4, scope: "/kakao/default" },
		});
		expect(result.structuredContent).toMatchObject({ ok: true, datasourceId: "kakao" });
		await client.close();
		await server.close();
	});

	it("reports index-not-ready from a dynamic datasource tool before refresh", async () => {
		const { lite, searchCalls } = fakeLite({ workspacePath: workspace(false) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({
			name: "autorag.search_datasource_kakao",
			arguments: { query: "refund" },
		});
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({
			errorCode: "index-not-ready",
			datasourceId: "kakao",
		});
		expect(searchCalls).toHaveLength(0);
		await client.close();
		await server.close();
	});
});
