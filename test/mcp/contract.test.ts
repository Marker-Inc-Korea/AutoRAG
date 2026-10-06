import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { FSearchClient } from "../../src/fsearch/client.ts";
import { createAutoRAGMcpServer } from "../../src/mcp/server.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { RetrievalSelectionError } from "../../src/retrieval/selection.ts";

const cleanup: (() => Promise<void> | void)[] = [];
afterEach(async () => {
	while (cleanup.length > 0) await cleanup.pop()?.();
});

const datasource = {
	datasourceId: "docs",
	name: "docs",
	type: "local",
	description: "Local documents",
	tags: ["local"],
	capabilities: ["keyword"],
	status: "active",
	sourceScopes: ["/docs/default"],
} as const;

function markReady(root: string): void {
	mkdirSync(join(root, ".autorag"), { recursive: true });
	writeFileSync(
		join(root, ".autorag", "refresh-complete.json"),
		JSON.stringify({ version: 1, completed: true, parsed: true }),
	);
}

function fixture(ready = true) {
	const root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-mcp-contract-")));
	cleanup.push(() => rmSync(root, { recursive: true, force: true }));
	if (ready) markReady(root);
	const state = { stale: false, diagnostics: [] as { code: string; severity: string; message: string }[] };
	const retrieved = {
		results: [] as {
			id: string;
			source: string;
			score: number;
			metadata: Record<string, unknown>;
			content: string;
		}[],
		diagnostics: [] as object[],
		unsearched: [] as object[],
	};
	const search = vi.fn(async () => retrieved);
	const refresh = vi.fn(async () => {
		markReady(root);
		state.stale = false;
		return { parsed: { ok: true } };
	});
	const fsearch = vi.fn(
		async (_request: object): Promise<object> => ({ ok: true, backend: "fsearch-cli", results: [] }),
	);
	const everything = vi.fn(async (_request: object): Promise<object> => ({ ok: true, results: [] }));
	const dupey = vi.fn(async (): Promise<string> => JSON.stringify({ dir: root, files: [], families: [], errors: [] }));
	const config = {
		searchPaths: [root],
		excludePaths: [] as string[],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		dupey: { run: dupey, enabled: true },
	};
	// Only external runtime dependencies are fake: Client, server factory,
	// input/output validation, protocol and transport remain the shipped code.
	const lite = {
		config,
		getRefreshStatus: async () => ({
			state: "success",
			inFlight: false,
			stale: state.stale,
			diagnostics: state.diagnostics,
			components: {},
		}),
		getRetrievalEngine: () => ({
			getMethodRegistry: () => ({
				list: () => [{ describe: () => ({ name: "docs.keyword", datasourceId: "docs" }) }],
			}),
		}),
		listDatasources: () => [datasource],
		searchSelected: search,
		refresh,
		searchEverything: everything,
		searchFsearch: fsearch,
	} as unknown as AutoRAGLite;
	return { root, lite, config, state, retrieved, search, refresh, fsearch, everything, dupey };
}

async function connect(lite: AutoRAGLite, options: Parameters<typeof createAutoRAGMcpServer>[1] = {}) {
	const server = createAutoRAGMcpServer(lite, options);
	const client = new Client({ name: "autorag-contract-client", version: "1.0.0" });
	// Register teardown before initialization; failed assertions/handshakes
	// cannot leave a transport live. Client closes before server.
	cleanup.push(async () => {
		try {
			await client.close();
		} finally {
			await server.close();
		}
	});
	const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
	await server.connect(serverTransport);
	await client.connect(clientTransport);
	return client;
}

type ToolResponse = Awaited<ReturnType<Client["callTool"]>>;
function payload(result: ToolResponse): Record<string, unknown> {
	const text = result.content.find((item) => item.type === "text");
	if (text?.type !== "text") throw new Error("Missing MCP text content");
	const json = JSON.parse(text.text);
	expect(json).toEqual(result.structuredContent);
	return json;
}

const invalidArguments: [string, Record<string, unknown>][] = [
	["autorag.status", { arbitrary: true }],
	["autorag.search", {}],
	["autorag.search", { query: "   " }],
	["autorag.search", { query: "hello", topK: 0 }],
	["autorag.search", { query: "hello", topK: 101 }],
	["autorag.search", { query: "hello", topK: 1.5 }],
	["autorag.search", { query: "hello", strict: "true" }],
	["autorag.search", { query: "hello", workspacePath: "/" }],
	["autorag.search", { query: "hello", tags: [""] }],
	["autorag.search.files", { query: "hello", maxResults: 0 }],
	["autorag.search.files", { query: "hello", maxResults: 1001 }],
	["autorag.search.files", { query: "hello", offset: -1 }],
	["autorag.search.files", { query: "hello", offset: 0.5 }],
	["autorag.search.files", { query: "hello", kind: "all" }],
	["autorag.search.files", { query: "hello", regex: true }],
	["autorag.datasources.list", { scope: "/" }],
	["autorag.datasources.get", {}],
	["autorag.datasources.get", { datasourceId: "   " }],
	["autorag.duplicates", { root: "/" }],
	["autorag.refresh", { force: "true" }],
	["autorag.refresh", { methods: [] }],
	["autorag.refresh", { methods: ["unknown"] }],
	["autorag.search_datasource_docs", { query: "hello", datasourceIds: ["secret"] }],
	["autorag.search_datasource_docs", { query: "hello", local: true }],
	["autorag.search_datasource_docs", { query: "hello", topK: 101 }],
];

describe("MCP Client contract boundaries", () => {
	it.each(invalidArguments)("rejects invalid %s arguments %j before backend execution", async (name, args) => {
		const f = fixture();
		const client = await connect(f.lite, { platform: "linux" });
		const result = await client.callTool({ name, arguments: args });
		expect(result.isError).toBe(true);
		expect(f.search).not.toHaveBeenCalled();
		expect(f.refresh).not.toHaveBeenCalled();
		expect(f.fsearch).not.toHaveBeenCalled();
		expect(f.everything).not.toHaveBeenCalled();
		expect(f.dupey).not.toHaveBeenCalled();
		// A validation failure must not poison the connection.
		expect(payload(await client.callTool({ name: "autorag.status", arguments: {} }))).toMatchObject({ stale: false });
	});

	it.each([
		{ options: { readOnly: true }, name: "autorag.refresh", args: {} },
		{ options: { readOnly: true }, name: "autorag.report", args: { query: "hello", report: {} } },
		{ options: { readOnly: true }, name: "autorag.feedback", args: { sessionId: "session", useful: [1] } },
		{ options: { tools: ["autorag.status"] }, name: "autorag.search", args: { query: "hello" } },
		{ options: { tools: ["autorag.status"] }, name: "autorag.search_datasource_docs", args: { query: "hello" } },
		{ options: { tools: ["autorag.status"] }, name: "autorag.evidence", args: { sessionId: "session" } },
		{ options: { tools: [] }, name: "autorag.status", args: {} },
		{ options: {}, name: "autorag.not_registered", args: {} },
	])("cannot directly call unavailable $name with $options", async ({ options, name, args }) => {
		const f = fixture();
		const client = await connect(f.lite, options);
		expect((await client.listTools()).tools.map((tool) => tool.name)).not.toContain(name);
		let error: unknown;
		try {
			await client.callTool({ name, arguments: args });
		} catch (caught) {
			error = caught;
		}
		expect(error).toBeDefined();
		expect([-32601, -32602]).toContain((error as { code?: number }).code);
		expect(f.search).not.toHaveBeenCalled();
		expect(f.refresh).not.toHaveBeenCalled();
	});

	it("transitions from not-ready to refreshed and distinguishes strict/non-strict stale results", async () => {
		const f = fixture(false);
		const client = await connect(f.lite);
		const before = await client.callTool({ name: "autorag.search", arguments: { query: "policy" } });
		expect(before.isError).toBe(true);
		expect(payload(before)).toMatchObject({ errorCode: "index-not-ready", action: "autorag.refresh" });
		expect(f.search).not.toHaveBeenCalled();
		const refreshed = await client.callTool({ name: "autorag.refresh", arguments: { methods: ["fsearch"] } });
		expect(refreshed.isError).not.toBe(true);
		expect(payload(refreshed)).toMatchObject({ ok: true, parsed: { ok: true } });
		f.state.stale = true;
		const strict = await client.callTool({ name: "autorag.search", arguments: { query: "policy", strict: true } });
		expect(strict.isError).toBe(true);
		expect(payload(strict)).toMatchObject({ errorCode: "stale-index", action: "autorag.refresh" });
		expect(f.search).not.toHaveBeenCalled();
		const permissive = await client.callTool({
			name: "autorag.search",
			arguments: { query: "policy", strict: false, topK: 100 },
		});
		expect(permissive.isError).not.toBe(true);
		expect(payload(permissive)).toMatchObject({ ok: true, stale: true, results: [] });
	});

	it.each(["autorag.search", "autorag.search_datasource_docs"])(
		"retains numbering, evidence and incomplete-coverage diagnostics for %s",
		async (name) => {
			const f = fixture();
			f.state.diagnostics.push({ code: "stale-index", severity: "warning", message: "Index predates source" });
			f.retrieved.results.push(
				{
					id: "1",
					source: "/docs/a.md",
					score: 0.8,
					metadata: { method: "docs.keyword", page: 3 },
					content: "Director approval required.",
				},
				{ id: "2", source: "opaque://docs/b", score: 0.5, metadata: {}, content: "A second source." },
			);
			f.retrieved.diagnostics.push({
				code: "retrieval-method-failed",
				severity: "error",
				message: "Archive offline",
			});
			f.retrieved.unsearched.push({ source: "/docs/archive", reason: "Archive offline" });
			const client = await connect(f.lite);
			const result = await client.callTool({ name, arguments: { query: "policy" } });
			expect(result.isError).not.toBe(true);
			expect(payload(result)).toMatchObject({
				results: [
					{
						number: 1,
						source: "/docs/a.md",
						method: "docs.keyword",
						score: 0.8,
						metadata: { page: 3 },
						content: "Director approval required.",
					},
					{ number: 2, source: "opaque://docs/b", method: "unknown", score: 0.5, content: "A second source." },
				],
				diagnostics: [f.state.diagnostics[0], f.retrieved.diagnostics[0]],
				unsearched: f.retrieved.unsearched,
			});
		},
	);

	it.each(["autorag.search", "autorag.search_datasource_docs"])(
		"classifies selection failures separately from runtime failures in %s",
		async (name) => {
			const f = fixture();
			f.search.mockRejectedValueOnce(
				new RetrievalSelectionError("unauthorized-datasource", "Datasource access denied"),
			);
			f.search.mockRejectedValueOnce(new Error("Archive unavailable"));
			const client = await connect(f.lite);
			const invalid = await client.callTool({ name, arguments: { query: "policy" } });
			expect(invalid.isError).toBe(true);
			expect(payload(invalid)).toMatchObject({
				errorCode: "invalid-selection",
				retryable: false,
				message: "Datasource access denied",
			});
			const failed = await client.callTool({ name, arguments: { query: "policy" } });
			expect(failed.isError).toBe(true);
			expect(payload(failed)).toMatchObject({
				errorCode: "search-failed",
				retryable: true,
				message: "Archive unavailable",
			});
			const recovered = await client.callTool({ name, arguments: { query: "policy" } });
			expect(recovered.isError).not.toBe(true);
			expect(payload(recovered)).toMatchObject({ ok: true, results: [] });
		},
	);

	it("reports refresh failures without destroying the session", async () => {
		const f = fixture();
		f.refresh.mockRejectedValueOnce(new Error("Index locked"));
		const client = await connect(f.lite);
		const failed = await client.callTool({ name: "autorag.refresh", arguments: { force: true } });
		expect(failed.isError).toBe(true);
		expect(payload(failed)).toMatchObject({ errorCode: "refresh-failed", retryable: true, message: "Index locked" });
		expect(payload(await client.callTool({ name: "autorag.refresh", arguments: {} }))).toMatchObject({ ok: true });
	});

	it("distinguishes disabled Dupey from a malformed scanner response", async () => {
		const f = fixture();
		f.config.dupey.enabled = false;
		const client = await connect(f.lite);
		const disabled = await client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(disabled.isError).toBe(true);
		expect(payload(disabled)).toMatchObject({ errorCode: "duplicates-disabled" });
		expect(f.dupey).not.toHaveBeenCalled();
		f.config.dupey.enabled = true;
		f.dupey.mockResolvedValueOnce("not JSON");
		const malformed = await client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(malformed.isError).toBe(true);
		expect(payload(malformed)).toMatchObject({ errorCode: "duplicates-failed", retryable: true });
	});

	it.each(["darwin", "linux"] as const)(
		"rejects out-of-root file searches and exposes backend failures on %s",
		async (platform) => {
			const f = fixture();
			const other = fixture();
			const client = await connect(f.lite, { platform });
			const denied = await client.callTool({
				name: "autorag.search.files",
				arguments: { query: "report", root: other.root },
			});
			expect(payload(denied)).toMatchObject({ results: [], diagnostics: [{ code: "root-out-of-scope" }] });
			expect(f.fsearch).not.toHaveBeenCalled();
			f.fsearch.mockResolvedValueOnce({ ok: false, reason: "search-failed", message: "FSearch database corrupt" });
			const failed = await client.callTool({ name: "autorag.search.files", arguments: { query: "report" } });
			expect(failed.isError).toBe(true);
			expect(payload(failed)).toMatchObject({
				ok: false,
				backend: "fsearch-cli",
				reason: "search-failed",
				message: "FSearch database corrupt",
			});
			expect(f.everything).not.toHaveBeenCalled();
		},
	);

	it("enforces trusted datasource authorization through the real engine", async () => {
		const f = fixture();
		const engine = new RetrievalEngine({
			datasourceAccess: { allowedTags: ["local"], allowedScopes: ["/docs/default/**"] },
		});
		const authorized = vi.fn(async () => [
			{ id: "allowed", source: "/docs/default/allowed", score: 1, metadata: {}, content: "Allowed evidence" },
			{ id: "secret-scope", source: "/docs/private/secret", score: 1, metadata: {}, content: "Must not leak" },
		]);
		const denied = vi.fn(async () => [
			{ id: "secret", source: "/secret/a", score: 1, metadata: {}, content: "Forbidden backend" },
		]);
		engine.register({
			describe: () => ({
				name: "docs.keyword",
				datasourceId: "docs",
				type: "bm25",
				description: "docs",
				status: "active",
				capabilities: ["scoped"],
				tags: ["local"],
			}),
			retrieve: authorized,
		});
		engine.register({
			describe: () => ({
				name: "secret.keyword",
				datasourceId: "secret",
				type: "bm25",
				description: "secret",
				status: "active",
				capabilities: ["scoped"],
				tags: ["secret"],
			}),
			retrieve: denied,
		});
		f.lite.getRetrievalEngine = () => engine;
		f.lite.searchSelected = (query, selection, options) => engine.retrieveSelected(query, selection, options);
		const client = await connect(f.lite);
		expect((await client.listTools()).tools.map((tool) => tool.name)).not.toContain(
			"autorag.search_datasource_secret",
		);
		const success = await client.callTool({ name: "autorag.search_datasource_docs", arguments: { query: "policy" } });
		expect(payload(success)).toMatchObject({
			results: [{ source: "/docs/default/allowed", content: "Allowed evidence" }],
		});
		expect(payload(success).results as unknown[]).toHaveLength(1);
		const refused = await client.callTool({
			name: "autorag.search",
			arguments: { query: "policy", datasourceIds: ["secret"] },
		});
		expect(refused.isError).toBe(true);
		expect(payload(refused)).toMatchObject({ errorCode: "invalid-selection", retryable: false });
		const widened = await client.callTool({
			name: "autorag.search",
			arguments: { query: "policy", tags: ["secret"], scope: "/secret/**" },
		});
		expect(payload(widened)).toMatchObject({ results: [] });
		expect(denied).not.toHaveBeenCalled();
		expect(authorized).toHaveBeenCalledTimes(1);
	});
	it("uses the real FSearchClient missing-binary fallback without claiming an indexed search", async () => {
		const f = fixture(false);
		writeFileSync(join(f.root, "report.a.txt"), "original content");
		writeFileSync(join(f.root, "reportXa.txt"), "other content");
		const backend = new FSearchClient({
			root: f.root,
			folders: [f.root],
			platform: "linux",
			binaryPath: "missing-fsearch",
			run: async () => ({ code: null, stdout: "", stderr: "ENOENT" }),
		});
		f.lite.searchFsearch = (request) => backend.search(request);
		const client = await connect(f.lite, { platform: "linux" });
		const result = await client.callTool({
			name: "autorag.search.files",
			arguments: { query: "report.a", kind: "files" },
		});
		expect(result.isError).not.toBe(true);
		expect(payload(result)).toMatchObject({
			ok: true,
			backend: "walk",
			results: [{ path: join(f.root, "report.a.txt"), type: "file" }],
			diagnostics: [{ code: "fsearch-degraded" }],
		});
	});
});
