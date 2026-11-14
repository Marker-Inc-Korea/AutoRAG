import { Client, InMemoryTransport } from "@modelcontextprotocol/client";
import type { AutoRAGLite } from "../../src/core.ts";
import { createAutoRAGMcpServer } from "../../src/mcp/server.ts";

/**
 * Every fixed tool the MCP server registers, read through a real client, so docs and agent packages can be checked
 * against what an agent will actually see rather than against source text. Datasource tools are dynamic (one per
 * configured archive) and are not included.
 */
export async function registeredMcpTools(options: { readOnly: boolean }) {
	// The server only reads these members at registration time.
	const lite = {
		config: {
			searchPaths: [],
			excludePaths: [],
			workspacePath: "/nonexistent",
			memoryPath: "/nonexistent/memory.json",
		},
		getRetrievalEngine: () => ({ getMethodRegistry: () => ({ list: () => [] }) }),
		listDatasources: () => [],
	} as unknown as AutoRAGLite;
	const server = createAutoRAGMcpServer(lite, { readOnly: options.readOnly });
	const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
	await server.connect(serverTransport);
	const client = new Client({ name: "tool-names", version: "1.0.0" });
	await client.connect(clientTransport);
	try {
		return (await client.listTools()).tools;
	} finally {
		await client.close();
	}
}

export async function registeredMcpToolNames(options: { readOnly: boolean }): Promise<readonly string[]> {
	return (await registeredMcpTools(options)).map((tool) => tool.name);
}
