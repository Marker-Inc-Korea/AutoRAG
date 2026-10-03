import { randomUUID } from "node:crypto";
import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import { describe, expect, it } from "vitest";
import type { AutoRAGLite } from "../../src/core.ts";
import { createAutoRAGMcpServer } from "../../src/mcp/server.ts";

function fakeLite(overrides: Partial<AutoRAGLite> = {}): AutoRAGLite {
	return {
		config: {
			searchPaths: [],
			workspacePath: `/tmp/autorag-mcp-${randomUUID()}`,
			memoryPath: `/tmp/autorag-mcp-${randomUUID()}.json`,
		},
		getRefreshStatus: async () => ({
			state: "success",
			inFlight: false,
			stale: false,
			diagnostics: [],
			components: {},
		}),
		getMemorySchema: () => ({
			version: 4,
			curatedResults: [],
			evidenceChunks: [],
			feedbackSignals: [],
			signalDefaults: { explicitWeight: 1, followupWeight: 1, retryWeight: 1, implicitCap: 1 },
			warnings: [],
			insights: [],
			pendingInsightSignals: [],
		}),
		recordPersistedFeedbackByNumbers: () => false,
		...overrides,
	} as unknown as AutoRAGLite;
}

async function connectedServer(lite: AutoRAGLite, options = {}) {
	const server = createAutoRAGMcpServer(lite, options);
	const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
	await server.connect(serverTransport);
	const client = new Client({ name: "autorag-mcp-test", version: "1.0.0" });
	await client.connect(clientTransport);
	return { client, server };
}

describe("AutoRAG Lite MCP server", () => {
	it("exposes the MVP tools with stable names", async () => {
		const { client, server } = await connectedServer(fakeLite());
		const { tools } = await client.listTools();
		expect(tools.map((tool) => tool.name)).toEqual([
			"autorag.status",
			"autorag.search",
			"autorag.refresh",
			"autorag.report",
			"autorag.evidence",
			"autorag.feedback",
			"autorag.duplicates",
		]);
		await client.close();
		await server.close();
	});

	it("omits write tools in read-only mode", async () => {
		const { client, server } = await connectedServer(fakeLite(), { readOnly: true });
		const { tools } = await client.listTools();
		expect(tools.map((tool) => tool.name)).toEqual([
			"autorag.status",
			"autorag.search",
			"autorag.evidence",
			"autorag.duplicates",
		]);
		await client.close();
		await server.close();
	});

	it("returns structured status output", async () => {
		const { client, server } = await connectedServer(fakeLite());
		const result = await client.callTool({ name: "autorag.status", arguments: {} });
		expect(result.isError).not.toBe(true);
		expect(result.structuredContent).toMatchObject({ state: "success", stale: false });
		await client.close();
		await server.close();
	});

	it("returns an actionable index-not-ready execution error", async () => {
		const { client, server } = await connectedServer(fakeLite());
		const result = await client.callTool({ name: "autorag.search", arguments: { query: "hello" } });
		expect(result.isError).toBe(true);
		expect(result.structuredContent).toMatchObject({
			errorCode: "index-not-ready",
			action: "autorag.refresh",
		});
		await client.close();
		await server.close();
	});

	it("validates feedback input before execution", async () => {
		const { client, server } = await connectedServer(fakeLite());
		const result = await client.callTool({ name: "autorag.feedback", arguments: { sessionId: "session" } });
		expect(result.isError).toBe(true);
		expect(result.content[0]).toMatchObject({ type: "text" });
		await client.close();
		await server.close();
	});
});
