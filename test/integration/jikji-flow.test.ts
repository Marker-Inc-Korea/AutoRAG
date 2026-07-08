import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

let root: string;
let docs: string;
let binaryPath: string;
let logPath: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-flow-"));
	docs = join(root, "docs");
	binaryPath = join(root, "fake-jikji.mjs");
	logPath = join(root, "jikji-calls.jsonl");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "q3-report.txt"), "Q3 document exists for Jikji lookup.\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeJikji(exitCode = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
console.log(JSON.stringify({ prepared: true }));
process.exit(${exitCode});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function methodNames(agent: AutoRAGAgent): string[] {
	return agent
		.getMethodRegistry()
		.list()
		.map((method) => method.describe().name);
}

function loggedArgs(): readonly string[][] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => (JSON.parse(line) as { readonly args: string[] }).args);
}

describe("AutoRAGAgent Jikji indexing integration", () => {
	it("does not register Jikji as a retrieval method by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		expect(methodNames(agent)).toEqual([]);
	});

	it("keeps Jikji out of the retrieval registry when configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		expect(methodNames(agent)).toEqual([]);
		expect(agent.getMethodRegistry().get("jikji")).toBeUndefined();
	});

	it("can configure MinSync and Jikji indexing independently", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
			jikji: { binaryPath },
		});

		expect(methodNames(agent)).toEqual(["minsync"]);
	});

	it("prepares Jikji maps without contributing retrieval results", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		const prepareResults = await agent.prepareJikji();
		const retrievalResults = await agent.retrieve("document", { topK: 1 });

		expect(prepareResults?.[0]).toMatchObject({ ok: true });
		expect(loggedArgs()).toEqual([["prepare", docs, "--json"]]);
		expect(retrievalResults).toEqual([]);
	});

	it("adds prompt guidance that Jikji is indexing context, not a retrieval backend", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("## Jikji File Map");
		expect(prompt).toContain("navigation hint");
		expect(prompt).toContain("Do not treat Jikji as an answer-producing retrieval backend");
	});
	it("only ever invokes `jikji prepare ... --json`, never `jikji find`", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		await agent.prepareJikji();

		const flatArgs = loggedArgs().flat();
		expect(flatArgs).not.toContain("find");
		for (const call of loggedArgs()) {
			expect(call[0]).toBe("prepare");
			expect(call).toContain("--json");
		}
	});

	it("constructs and retrieves without Jikji configured and without any Python runtime", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		await expect(agent.prepareJikji()).resolves.toBeUndefined();
		const results = await agent.retrieve("document", { topK: 1 });
		expect(results).toEqual([]);
	});

	it("surfaces a path-free degraded diagnostic when the configured Jikji binary is missing", async () => {
		const missingBinary = join(root, "does-not-exist-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: missingBinary },
		});

		const results = await agent.prepareJikji();
		const first = results?.[0];
		expect(first).toMatchObject({ ok: false, reason: "spawn-error" });
		expect(JSON.stringify(first)).not.toContain(missingBinary);
		expect(JSON.stringify(first)).not.toContain(root);
		await agent.refresh(true);
		const diag = (await agent.getRefreshStatus()).diagnostics.find((item) => item.source === "jikji");
		expect(diag?.code).toBe("jikji-unavailable");
		expect(diag?.message).not.toContain(missingBinary);
		expect(diag?.message).not.toContain(root);
	});
});
