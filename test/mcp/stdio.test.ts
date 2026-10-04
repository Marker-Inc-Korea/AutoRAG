import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { describe, expect, it } from "vitest";
import { writeFakeMinSyncExecutable } from "../helpers/fake-minsync.ts";

function makeFixture() {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-stdio-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "refund.md"), "Refund exceptions require director approval before payout.\n");
	const stagedFiles = join(root, ".autorag", "minsync", "files");
	mkdirSync(stagedFiles, { recursive: true });
	writeFileSync(join(stagedFiles, "refund.md"), "Refund exceptions require director approval before payout.\n");
	// `AUTORAG_CONFIG` deliberately ignores `minSync.binaryPath` (MinSync resolves
	// from PATH/the workspace cache), so the fake must sit on PATH under the
	// resolver's platform name to make the e2e independent of an installed minsync.
	const binDir = join(root, "bin");
	mkdirSync(binDir, { recursive: true });
	writeFakeMinSyncExecutable(binDir);
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
			fsearch: false,
		}),
	);
	return { root, docs, config, binDir };
}

async function connect(config: string, binDir: string) {
	// Windows spells the variable `Path`; reuse the existing key so the child
	// env has no case-duplicate entry and the fake stays first on the lookup path.
	const pathKey = Object.keys(process.env).find((key) => key.toLowerCase() === "path") ?? "PATH";
	const env = Object.fromEntries(
		Object.entries(process.env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	) as Record<string, string>;
	env.AUTORAG_CONFIG = config;
	env[pathKey] = `${binDir}${delimiter}${env[pathKey] ?? ""}`;
	const transport = new StdioClientTransport({
		command: process.execPath,
		args: [join(process.cwd(), "src/mcp/index.ts")],
		env,
		stderr: "pipe",
	});
	const client = new Client({ name: "autorag-stdio-qa", version: "1.0.0" });
	await client.connect(transport);
	return { client, transport };
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

describe("AutoRAG Lite MCP stdio", () => {
	it("runs refresh, search, file-name search, and datasource list/get across a restart", async () => {
		const { root, docs, config, binDir } = makeFixture();
		try {
			const first = await connect(config, binDir);
			const listed = await first.client.listTools();
			const names = listed.tools.map((tool) => tool.name);
			for (const tool of [
				"autorag.status",
				"autorag.search",
				"autorag.search.files",
				"autorag.search.everything",
				"autorag.datasources.list",
				"autorag.datasources.get",
				"autorag.refresh",
			]) {
				expect(names).toContain(tool);
			}

			const refresh = await first.client.callTool({ name: "autorag.refresh", arguments: {} });
			expect(refresh.isError).not.toBe(true);

			const search = await first.client.callTool({
				name: "autorag.search",
				arguments: { query: "Who approves refund exceptions?", topK: 3 },
			});
			expect(search.isError).not.toBe(true);
			const searchOutput = search.structuredContent;
			if (
				searchOutput === null ||
				typeof searchOutput !== "object" ||
				!("results" in searchOutput) ||
				!Array.isArray(searchOutput.results)
			) {
				throw new Error("MCP search did not return a results array");
			}
			expect(searchOutput.results.length).toBeGreaterThan(0);

			const files = await first.client.callTool({
				name: "autorag.search.files",
				arguments: { query: "refund" },
			});
			expect(files.isError).not.toBe(true);
			expect(hasFieldValue(files.structuredContent, "path", realpathSync(join(docs, "refund.md")))).toBe(true);
			await first.client.close();

			const second = await connect(config, binDir);
			const list = await second.client.callTool({ name: "autorag.datasources.list", arguments: {} });
			expect(list.isError).not.toBe(true);

			const missing = await second.client.callTool({
				name: "autorag.datasources.get",
				arguments: { datasourceId: "missing" },
			});
			expect(missing.isError === true || field(missing.structuredContent, "ok") === false).toBe(true);
			await second.client.close();
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	}, 120000);
});
