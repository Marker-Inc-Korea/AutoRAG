#!/usr/bin/env node
import { realpathSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { createAutoRAGLite } from "../core.ts";
import { configSignature, createReloadingLite } from "./reloading-lite.ts";
import { createAutoRAGMcpServer } from "./server.ts";

function parseBoolean(value: string | undefined, fallback = false): boolean {
	if (value === undefined) return fallback;
	return value === "1" || value.toLowerCase() === "true";
}

function parseTools(value: string | undefined): readonly string[] | undefined {
	if (value === undefined || value.trim() === "") return undefined;
	return value
		.split(",")
		.map((tool) => tool.trim())
		.filter((tool) => tool.length > 0);
}

function isInvokedDirectly(): boolean {
	const entry = process.argv[1];
	if (entry === undefined) return false;
	try {
		return realpathSync(entry) === realpathSync(fileURLToPath(import.meta.url));
	} catch {
		return false;
	}
}

export async function main(): Promise<void> {
	const readOnly = parseBoolean(process.env.AUTORAG_MCP_READ_ONLY);
	const tools = parseTools(process.env.AUTORAG_MCP_TOOLS);
	const lite = createReloadingLite({
		create: () => createAutoRAGLite({ env: process.env, readOnly }),
		signature: () => configSignature(process.env),
		// stdout is the protocol channel; a config that stopped loading is only worth a stderr line.
		onError: (error) =>
			process.stderr.write(`autorag-mcp: keeping the previous config, the new one failed: ${error.message}\n`),
	});
	const server = createAutoRAGMcpServer(lite, { readOnly, tools });
	await server.connect(new StdioServerTransport());
}

if (isInvokedDirectly()) {
	main().catch((error: unknown) => {
		process.stderr.write(`${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`);
		process.exitCode = 1;
	});
}
