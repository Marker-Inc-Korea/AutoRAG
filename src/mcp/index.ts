#!/usr/bin/env node
import { randomBytes } from "node:crypto";
import { chmodSync, mkdirSync, readFileSync, realpathSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { parseArgs } from "node:util";
import { StdioServerTransport } from "@modelcontextprotocol/server/stdio";
import { resolveAutoRAGHome } from "../config/home.ts";
import { createAutoRAGLite } from "../core.ts";
import { startMcpHttpServer } from "./http.ts";
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

const TOKEN_FILE = "mcp-http-token";

const USAGE = `autorag-mcp - AutoRAG Lite MCP server

  autorag-mcp                       stdio (default; what Hermes, Claude Code and Codex spawn)
  autorag-mcp --http [options]      streamable HTTP, for agents that run in a cloud and reach you through a URL

HTTP options:
  --host HOST        interface to bind (default 127.0.0.1)
  --port PORT        port (default 8765; 0 picks a free one)
  --public-url URL   https origin the agent reaches this server at (tunnel/proxy); enables the OAuth endpoints
                     connector-only clients need. Also AUTORAG_MCP_PUBLIC_URL.
  --allow-write      also offer autorag.refresh, autorag.report and autorag.convert (default: read-only)

Auth: every request needs "Authorization: Bearer <owner token>". The token is AUTORAG_MCP_TOKEN (at least 32
characters) or, when unset, a random one kept at $AUTORAG_HOME/${TOKEN_FILE} (owner-only).
Env: AUTORAG_MCP_TOOLS, AUTORAG_MCP_READ_ONLY (stdio), AUTORAG_CONFIG, AUTORAG_HOME.`;

/** The owner token from the environment, or a random one persisted owner-only so restarts keep issued tokens valid. */
function resolveOwnerToken(env: NodeJS.ProcessEnv): { token: string; file?: string } {
	const fromEnv = env.AUTORAG_MCP_TOKEN?.trim();
	if (fromEnv !== undefined && fromEnv !== "") return { token: fromEnv };
	const home = resolveAutoRAGHome(env);
	const file = join(home, TOKEN_FILE);
	try {
		const existing = readFileSync(file, "utf8").trim();
		if (existing !== "") return { token: existing, file };
	} catch {
		// First run: no token yet.
	}
	mkdirSync(home, { recursive: true });
	const token = randomBytes(32).toString("base64url");
	writeFileSync(file, `${token}\n`, { mode: 0o600, flag: "wx" });
	chmodSync(file, 0o600);
	return { token, file };
}

export async function main(argv: readonly string[] = process.argv.slice(2)): Promise<void> {
	const { values } = parseArgs({
		args: [...argv],
		strict: true,
		options: {
			http: { type: "boolean" },
			host: { type: "string" },
			port: { type: "string" },
			"public-url": { type: "string" },
			"allow-write": { type: "boolean" },
			help: { type: "boolean", short: "h" },
		},
	});
	if (values.help === true) {
		process.stdout.write(`${USAGE}\n`);
		return;
	}
	const tools = parseTools(process.env.AUTORAG_MCP_TOOLS);
	if (values.http !== true) {
		for (const flag of ["host", "port", "public-url", "allow-write"] as const) {
			if (values[flag] !== undefined) throw new Error(`--${flag} only applies with --http.`);
		}
	}
	const readOnly =
		values.http === true ? values["allow-write"] !== true : parseBoolean(process.env.AUTORAG_MCP_READ_ONLY);
	const lite = createReloadingLite({
		create: () => createAutoRAGLite({ env: process.env, readOnly }),
		signature: () => configSignature(process.env),
		// stdout is the protocol channel; a config that stopped loading is only worth a stderr line.
		onError: (error) =>
			process.stderr.write(`autorag-mcp: keeping the previous config, the new one failed: ${error.message}\n`),
	});
	if (values.http !== true) {
		const server = createAutoRAGMcpServer(lite, { readOnly, tools });
		await server.connect(new StdioServerTransport());
		return;
	}
	const port = values.port === undefined ? 8765 : Number(values.port);
	if (!Number.isInteger(port) || port < 0 || port > 65535) throw new Error(`Invalid --port: ${values.port}`);
	const owner = resolveOwnerToken(process.env);
	process.stderr.write(
		owner.file === undefined
			? "autorag-mcp: owner token from AUTORAG_MCP_TOKEN\n"
			: `autorag-mcp: owner token stored at ${owner.file}\n`,
	);
	const http = await startMcpHttpServer({
		lite,
		host: values.host,
		port,
		publicUrl: values["public-url"] ?? process.env.AUTORAG_MCP_PUBLIC_URL?.trim() ?? undefined,
		ownerToken: owner.token,
		readOnly,
		tools,
		onError: (error) => process.stderr.write(`autorag-mcp: ${error.message}\n`),
	});
	if (http.publicMcpUrl !== undefined) process.stderr.write(`autorag-mcp: public endpoint ${http.publicMcpUrl}\n`);
	process.stderr.write(`autorag-mcp: listening on ${http.url} (${readOnly ? "read-only" : "read-write"})\n`);
	const stop = (): void => {
		http.close().finally(() => process.exit(0));
	};
	process.once("SIGTERM", stop);
	process.once("SIGINT", stop);
}

if (isInvokedDirectly()) {
	main().catch((error: unknown) => {
		process.stderr.write(`${error instanceof Error ? (error.stack ?? error.message) : String(error)}\n`);
		process.exitCode = 1;
	});
}
