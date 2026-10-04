import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
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

interface EverythingCall {
	readonly request: unknown;
}

interface FakeOptions {
	readonly workspacePath: string;
	readonly searchPaths?: readonly string[];
	readonly everything?: unknown;
}

function fakeLite(options: FakeOptions): {
	lite: AutoRAGLite;
	searchCalls: SearchCall[];
	everythingCalls: EverythingCall[];
} {
	const searchCalls: SearchCall[] = [];
	const everythingCalls: EverythingCall[] = [];
	// Unchecked cast: the fake implements only the methods the MCP surface calls.
	const lite = {
		config: {
			searchPaths: options.searchPaths ?? [],
			workspacePath: options.workspacePath,
			memoryPath: join(options.workspacePath, "memory.json"),
		},
		getRefreshStatus: async () => ({
			state: "success",
			inFlight: false,
			stale: false,
			diagnostics: [],
			components: {},
		}),
		listDatasources: () => [datasource],
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
			everythingCalls.push({ request });
			return (
				options.everything ?? {
					ok: true,
					results: [{ path: "C:\\docs\\refund.txt", type: "file", size: 12, dateModified: undefined }],
				}
			);
		},
		refresh: async () => ({ ok: true }),
	} as unknown as AutoRAGLite;
	return { lite, searchCalls, everythingCalls };
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

describe("AutoRAG Lite MCP server", () => {
	it("exposes the search-only tools with stable names", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect([...names].sort()).toEqual(
			[
				"autorag.status",
				"autorag.search",
				"autorag.search.files",
				"autorag.search.everything",
				"autorag.datasources.list",
				"autorag.datasources.get",
				"autorag.refresh",
			].sort(),
		);
		await client.close();
		await server.close();
	});

	it("omits only refresh in read-only mode", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite, { readOnly: true });
		const { tools } = await client.listTools();
		const names = tools.map((tool) => tool.name);
		expect(names).not.toContain("autorag.refresh");
		expect(names).toContain("autorag.search");
		expect(names).toContain("autorag.search.files");
		expect(names).toContain("autorag.search.everything");
		expect(names).toContain("autorag.datasources.list");
		expect(names).toContain("autorag.datasources.get");
		await client.close();
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

	it("searches configured file roots by name without reading contents", async () => {
		const docs = mkdtempSync(join(tmpdir(), "autorag-mcp-docs-"));
		roots.push(docs);
		writeFileSync(join(docs, "refund-policy.md"), "Director approval is required.\n");
		const { lite } = fakeLite({ workspacePath: workspace(true), searchPaths: [docs] });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.search.files", arguments: { query: "refund" } });
		expect(result.isError).not.toBe(true);
		expect(hasFieldValue(result.structuredContent, "path", realpathSync(join(docs, "refund-policy.md")))).toBe(true);
		await client.close();
		await server.close();
	});

	it("forwards Everything search fields to the provider", async () => {
		const { lite, everythingCalls } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({
			name: "autorag.search.everything",
			arguments: { query: "ext:txt refund", regex: true, kind: "files", sort: "name-ascending" },
		});
		expect(result.isError).not.toBe(true);
		expect(everythingCalls).toHaveLength(1);
		expect(everythingCalls[0]?.request).toMatchObject({
			query: "ext:txt refund",
			regex: true,
			kind: "files",
			sort: "name-ascending",
		});
		await client.close();
		await server.close();
	});

	it("surfaces an Everything backend failure as an error", async () => {
		const { lite } = fakeLite({
			workspacePath: workspace(true),
			everything: { ok: false, reason: "unsupported-platform", message: "Everything is not enabled on this host." },
		});
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.search.everything", arguments: { query: "refund" } });
		expect(result.isError === true || field(result.structuredContent, "ok") === false).toBe(true);
		await client.close();
		await server.close();
	});

	it("lists the authorized datasource catalog", async () => {
		const { lite } = fakeLite({ workspacePath: workspace(true) });
		const { client, server } = await connectedServer(lite);
		const result = await client.callTool({ name: "autorag.datasources.list", arguments: {} });
		expect(result.isError).not.toBe(true);
		expect(hasFieldValue(result.structuredContent, "datasourceId", "kakao")).toBe(true);
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
});
