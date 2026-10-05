import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { Client, ProtocolError, ProtocolErrorCode } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { describe, expect, it, onTestFinished } from "vitest";
import { writeFakeDupeyExecutable } from "../helpers/fake-dupey.ts";
import { writeFakeFSearchExecutable } from "../helpers/fake-fsearch.ts";
import { writeFakeMinSyncExecutable } from "../helpers/fake-minsync.ts";

const REFUND_FILE = "refund.md";
const REFUND_TEXT = "Refund exceptions require director approval before payout.\n";

function makeFixture() {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-stdio-"));
	// Registered before any connection: even if the handshake or an assertion
	// throws, the workspace and any spawned children are torn down.
	onTestFinished(() => rmSync(root, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 }));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, REFUND_FILE), REFUND_TEXT);
	const stagedFiles = join(root, ".autorag", "minsync", "files");
	mkdirSync(stagedFiles, { recursive: true });
	writeFileSync(join(stagedFiles, REFUND_FILE), REFUND_TEXT);
	// `AUTORAG_CONFIG` deliberately ignores `minSync.binaryPath` (MinSync resolves
	// from PATH/the workspace cache), so the fake must sit on PATH under the
	// resolver's platform name to make the e2e independent of an installed minsync.
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	writeFakeFSearchExecutable(binDir, join(docs, REFUND_FILE));
	writeFakeMinSyncExecutable(binDir);
	const dupeyBinary = writeFakeDupeyExecutable(binDir);
	const config = join(root, "config.json");
	writeFileSync(
		config,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: {
				workspacePath: join(root, ".autorag", "minsync"),
				autoInstall: false,
			},
			jikji: false,
			everything: false,
			// `watch: false` keeps the fake from becoming a live daemon: the fixture
			// must never leave a `fsearch-cli watch` process (or its timer) behind.
			fsearch: {
				binaryPath: join(binDir, process.platform === "win32" ? "fsearch-cli.exe" : "fsearch-cli"),
				watch: false,
			},
			// Pass the shim explicitly: `portableSpawnCommand` only rewrites a shebang
			// script to its interpreter when it can see the file at that exact path.
			dupey: { binaryPath: dupeyBinary },
			// Spotlight needs no external install, so a configured skill yields a
			// dynamic datasource tool without touching the network or a binary.
			datasources: { spotlight: { enabled: true } },
			datasourceAccess: { allowedTags: ["spotlight"] },
		}),
	);
	return { root, docs, config, binDir };
}

interface StdioConnection {
	readonly client: Client;
	readonly close: () => Promise<void>;
}

async function closeQuietly(target: { close(): Promise<void> }): Promise<void> {
	try {
		await target.close();
	} catch {
		// Already closed or never fully connected; takeover teardown still ran.
	}
}

async function connect(
	config: string,
	binDir: string,
	envOverrides: Record<string, string> = {},
): Promise<StdioConnection> {
	// Windows spells the variable `Path`; reuse the existing key so the child
	// env has no case-duplicate entry and the fake stays first on the lookup path.
	const pathKey = Object.keys(process.env).find((key) => key.toLowerCase() === "path") ?? "PATH";
	const env = Object.fromEntries(
		Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	) as Record<string, string>;
	env.AUTORAG_CONFIG = config;
	env[pathKey] = `${binDir}${delimiter}${env[pathKey] ?? ""}`;
	for (const [key, value] of Object.entries(envOverrides)) env[key] = value;
	const transport = new StdioClientTransport({
		command: process.execPath,
		args: [join(process.cwd(), "src/mcp/index.ts")],
		env,
		stderr: "pipe",
	});
	const client = new Client({ name: "autorag-stdio-qa", version: "1.0.0" });
	let closed: Promise<void> | undefined;
	const close = (): Promise<void> => {
		closed ??= (async () => {
			await closeQuietly(client);
			await closeQuietly(transport);
		})();
		return closed;
	};
	// Register teardown BEFORE connecting so a failed handshake or assertion
	// cannot leak the stdio child process.
	onTestFinished(close);
	try {
		await client.connect(transport);
	} catch (error) {
		await close();
		throw error;
	}
	return { client, close };
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

/** The tool-authored text block must be the pretty-printed form of `structuredContent`. */
function expectStructuredMatchesText(result: { content: unknown; structuredContent?: unknown }): void {
	const content = result.content;
	if (!Array.isArray(content) || content.length === 0) throw new Error("MCP result had no content blocks");
	const text = field(content[0], "text");
	expect(typeof text).toBe("string");
	expect(JSON.parse(text as string)).toEqual(result.structuredContent);
}

/** Extract a tool's `results` array, or fail loudly when the payload shape drifted. */
function resultsArray(structured: unknown): readonly unknown[] {
	const results = field(structured, "results");
	if (!Array.isArray(results)) throw new Error("MCP result did not contain a results array");
	return results;
}

/** Calls a tool that must fail at the SDK/protocol layer and returns the thrown error. */
async function expectProtocolFailure(call: Promise<unknown>): Promise<ProtocolError> {
	let error: unknown;
	try {
		await call;
	} catch (caught) {
		error = caught;
	}
	expect(error).toBeInstanceOf(ProtocolError);
	const protocolError = error as ProtocolError;
	expect(protocolError.code).toBe(ProtocolErrorCode.InvalidParams);
	return protocolError;
}

/** Assert the exact minsync evidence for the fixture corpus. */
function expectRefundEvidence(results: readonly unknown[], docs: string): void {
	expect(results).toHaveLength(1);
	const hit = results[0];
	expect(field(hit, "source")).toBe(realpathSync(join(docs, REFUND_FILE)));
	expect(field(hit, "content")).toBe(REFUND_TEXT);
}

describe("AutoRAG Lite MCP stdio", () => {
	it("refreshes, searches exact fixture evidence, routes filename search to fsearch-cli, and preserves results across a restart", async () => {
		const { docs, config, binDir } = makeFixture();
		const first = await connect(config, binDir);
		const listed = await first.client.listTools();
		const names = listed.tools.map((tool) => tool.name);
		for (const tool of [
			"autorag.status",
			"autorag.search",
			"autorag.search.files",
			"autorag.datasources.list",
			"autorag.datasources.get",
			"autorag.duplicates",
			"autorag.search_datasource_spotlight",
			"autorag.refresh",
		]) {
			expect(names).toContain(tool);
		}

		const duplicatesTool = listed.tools.find((tool) => tool.name === "autorag.duplicates");
		expect(duplicatesTool?.description).toContain("Dupey");
		const dynamicTool = listed.tools.find((tool) => tool.name === "autorag.search_datasource_spotlight");
		expect(dynamicTool?.description).toContain("Spotlight");

		// Before refresh the dynamic tool must fail ready-gated without touching Spotlight.
		const dynamicNotReady = await first.client.callTool({
			name: "autorag.search_datasource_spotlight",
			arguments: { query: "refund" },
		});
		expect(dynamicNotReady.isError).toBe(true);
		expect(hasFieldValue(dynamicNotReady.structuredContent, "errorCode", "index-not-ready")).toBe(true);
		expect(hasFieldValue(dynamicNotReady.structuredContent, "datasourceId", "spotlight")).toBe(true);

		const duplicates = await first.client.callTool({ name: "autorag.duplicates", arguments: {} });
		expect(duplicates.isError).not.toBe(true);
		expectStructuredMatchesText(duplicates);
		expect(hasFieldValue(duplicates.structuredContent, "action", "review")).toBe(true);
		expect(hasFieldValue(duplicates.structuredContent, "hash", "dupey-fixture-hash")).toBe(true);

		const refresh = await first.client.callTool({ name: "autorag.refresh", arguments: {} });
		expect(refresh.isError).not.toBe(true);
		expectStructuredMatchesText(refresh);

		const search = await first.client.callTool({
			name: "autorag.search",
			arguments: { query: "Who approves refund exceptions?", topK: 3 },
		});
		expect(search.isError).not.toBe(true);
		expectStructuredMatchesText(search);
		expectRefundEvidence(resultsArray(search.structuredContent), docs);

		const files = await first.client.callTool({
			name: "autorag.search.files",
			arguments: { query: "refund" },
		});
		if (process.platform === "win32") {
			// The fixture disables Everything, so Windows routing must surface the
			// provider failure as an MCP error instead of silently using the walker.
			expect(files.isError).toBe(true);
			expect(hasFieldValue(files.structuredContent, "backend", "everything")).toBe(true);
		} else {
			expect(files.isError).not.toBe(true);
			expectStructuredMatchesText(files);
			expect(field(files.structuredContent, "backend")).toBe("fsearch-cli");
			expect(hasFieldValue(files.structuredContent, "path", realpathSync(join(docs, REFUND_FILE)))).toBe(true);
		}
		await first.close();

		// A fresh process must serve the same corpus evidence from the persisted index.
		const second = await connect(config, binDir);
		const status = await second.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
		expect(field(status.structuredContent, "state")).toBe("success");
		expect(field(status.structuredContent, "stale")).toBe(false);

		const restarted = await second.client.callTool({
			name: "autorag.search",
			arguments: { query: "Who approves refund exceptions?", topK: 3 },
		});
		expect(restarted.isError).not.toBe(true);
		expectStructuredMatchesText(restarted);
		expectRefundEvidence(resultsArray(restarted.structuredContent), docs);

		const list = await second.client.callTool({ name: "autorag.datasources.list", arguments: {} });
		expect(list.isError).not.toBe(true);
		expectStructuredMatchesText(list);

		const missing = await second.client.callTool({
			name: "autorag.datasources.get",
			arguments: { datasourceId: "missing" },
		});
		expect(missing.isError === true || field(missing.structuredContent, "ok") === false).toBe(true);
		await second.close();
	}, 120000);

	it("omits refresh under the read-only env and rejects a direct call as a protocol error", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir, { AUTORAG_MCP_READ_ONLY: "1" });
		const names = (await connection.client.listTools()).tools.map((tool) => tool.name);
		expect(names).toContain("autorag.status");
		expect(names).toContain("autorag.search");
		expect(names).not.toContain("autorag.refresh");

		const error = await expectProtocolFailure(connection.client.callTool({ name: "autorag.refresh", arguments: {} }));
		expect(error.message).toContain("autorag.refresh");

		// The read-only server still serves the tools it does expose.
		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
	}, 60000);

	it("lists only allowlisted tools and rejects an omitted tool as a protocol error", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir, { AUTORAG_MCP_TOOLS: "autorag.status,autorag.search.files" });
		const names = (await connection.client.listTools()).tools.map((tool) => tool.name);
		expect(names).toHaveLength(2);
		expect(names).toContain("autorag.status");
		expect(names).toContain("autorag.search.files");

		const error = await expectProtocolFailure(
			connection.client.callTool({ name: "autorag.duplicates", arguments: {} }),
		);
		expect(error.message).toContain("autorag.duplicates");

		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
	}, 60000);

	it("reports invalid input as a tool error without poisoning the connection", async () => {
		const { config, binDir } = makeFixture();
		const connection = await connect(config, binDir);

		const invalid = await connection.client.callTool({
			name: "autorag.search.files",
			arguments: { query: "refund", maxResults: -1 },
		});
		expect(invalid.isError).toBe(true);
		const invalidText = field(Array.isArray(invalid.content) ? invalid.content[0] : undefined, "text");
		expect(typeof invalidText).toBe("string");
		expect(invalidText).toContain("Input validation error");

		// The same connection must still answer a valid call with real state.
		const status = await connection.client.callTool({ name: "autorag.status", arguments: {} });
		expect(status.isError).not.toBe(true);
		expectStructuredMatchesText(status);
		expect(field(status.structuredContent, "state")).toBe("idle");
		expect(field(status.structuredContent, "inFlight")).toBe(false);
	}, 60000);
});
