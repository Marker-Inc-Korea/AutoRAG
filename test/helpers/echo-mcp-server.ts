/**
 * A real stdio MCP server for tests of AutoRAG's MCP *client* wiring. Run it with
 * `process.execPath <this file>`; it serves one `echo` tool. stdout belongs to the
 * protocol, so nothing else is written there. When `ECHO_MCP_PID_FILE` is set the
 * server records its pid there, so a test can prove the process was stopped.
 */
import { writeFileSync } from "node:fs";
import { McpServer } from "@modelcontextprotocol/server";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { z } from "zod";

const pidFile = process.env.ECHO_MCP_PID_FILE;
if (pidFile !== undefined) writeFileSync(pidFile, String(process.pid));

const server = new McpServer({ name: "echo-fixture", version: "1.0.0" });
server.registerTool(
	"echo",
	{ description: "Echo the given text back, prefixed with the marker.", inputSchema: z.object({ text: z.string() }) },
	async ({ text }) => ({ content: [{ type: "text", text: `echo-fixture:${text}` }] }),
);
await server.connect(new StdioServerTransport());
