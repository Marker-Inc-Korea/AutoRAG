import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Client } from "@modelcontextprotocol/client";
import { StdioClientTransport } from "@modelcontextprotocol/client/stdio";
import { describe, expect, it } from "vitest";
import { writeFakeMinSync } from "../helpers/fake-minsync.ts";

function makeFixture() {
	const root = mkdtempSync(join(tmpdir(), "autorag-mcp-stdio-"));
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "refund.md"), "Refund exceptions require director approval before payout.\n");
	const stagedFiles = join(root, ".autorag", "minsync", "files");
	mkdirSync(stagedFiles, { recursive: true });
	writeFileSync(join(stagedFiles, "refund.md"), "Refund exceptions require director approval before payout.\n");
	writeFakeMinSync(join(root, "fake-minsync.mjs"));
	const config = join(root, "config.json");
	writeFileSync(
		config,
		JSON.stringify({
			searchPaths: [docs],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			minSync: {
				binaryPath: join(root, "fake-minsync.mjs"),
				workspacePath: join(root, ".autorag", "minsync"),
				autoInstall: false,
			},
			jikji: false,
			everything: false,
		}),
	);
	return { root, config };
}

async function connect(config: string) {
	const transport = new StdioClientTransport({
		command: process.execPath,
		args: [join(process.cwd(), "src/mcp/index.ts")],
		env: { ...process.env, AUTORAG_CONFIG: config },
		stderr: "pipe",
	});
	const client = new Client({ name: "autorag-stdio-qa", version: "1.0.0" });
	await client.connect(transport);
	return { client, transport };
}

describe("AutoRAG Lite MCP stdio", () => {
	it("runs refresh, search, report, evidence, and feedback across a restart", async () => {
		const { root, config } = makeFixture();
		try {
			const first = await connect(config);
			const listed = await first.client.listTools();
			expect(listed.tools.map((tool) => tool.name)).toContain("autorag.search");

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

			const report = await first.client.callTool({
				name: "autorag.report",
				arguments: {
					query: "Who approves refund exceptions?",
					report: {
						answer: "[1] Director approval is required before payout.",
						results: [
							{
								number: 1,
								title: "Refund exception approval",
								summary: "Director approval is required before payout.",
								evidence: [{ excerpt: "Refund exceptions require director approval before payout." }],
								confidence: 0.9,
							},
						],
						mapping: [
							{
								number: 1,
								source: join(root, "docs", "refund.md"),
								method: "manual-qa",
								content: "Refund exceptions require director approval before payout.",
							},
						],
					},
				},
			});
			expect(report.isError).not.toBe(true);
			const reportContent = report.structuredContent;
			if (
				reportContent === null ||
				typeof reportContent !== "object" ||
				!("sessionId" in reportContent) ||
				typeof reportContent.sessionId !== "string"
			) {
				throw new Error("MCP report did not return a sessionId");
			}
			const sessionId = reportContent.sessionId;
			await first.client.close();

			const second = await connect(config);
			const evidence = await second.client.callTool({
				name: "autorag.evidence",
				arguments: { sessionId },
			});
			expect(evidence.isError).not.toBe(true);
			expect(evidence.structuredContent).toMatchObject({ sessionId });

			const feedback = await second.client.callTool({
				name: "autorag.feedback",
				arguments: { sessionId, usefulNumbers: [1] },
			});
			expect(feedback.isError).not.toBe(true);
			expect(feedback.structuredContent).toMatchObject({ applied: true, sessionId });
			await second.client.close();

			const memory = JSON.parse(readFileSync(join(root, "memory.json"), "utf8")) as {
				feedbackSignals: unknown[];
			};
			expect(memory.feedbackSignals.length).toBeGreaterThan(0);
		} finally {
			rmSync(root, { recursive: true, force: true });
		}
	}, 120000);
});
