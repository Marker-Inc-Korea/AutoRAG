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

function writeFakeJikji(payload: unknown, exitCode = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
console.log(JSON.stringify(${JSON.stringify(payload)}));
process.exit(${exitCode});
`,
	);
	chmodSync(binaryPath, 0o755);
}

function payload(): unknown {
	return {
		query_type: "single_file",
		confidence: "high",
		handoff_action: "direct_use",
		index_status: "ready",
		paths: ["q3-report.txt"],
		answer_paths: ["q3-report.txt"],
		evidence_pack: [
			{
				path: "q3-report.txt",
				why: ["body"],
				matched_terms: ["enterprise"],
				evidence: ["Jikji says Q3 enterprise revenue increased"],
			},
		],
		judge_candidate_slate: [],
		candidates: [],
	};
}

function methodNames(agent: AutoRAGAgent): string[] {
	return agent
		.getMethodRegistry()
		.list()
		.map((method) => method.describe().name);
}

describe("AutoRAGAgent Jikji integration", () => {
	it("does not register Jikji by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		expect(methodNames(agent)).toEqual(["posix"]);
	});

	it("includes Jikji results in retrieve when configured", async () => {
		writeFakeJikji(payload());
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		const results = await agent.retrieve("enterprise revenue", { topK: 1 });

		expect(methodNames(agent)).toEqual(["posix", "jikji"]);
		expect(results).toHaveLength(1);
		expect(results[0]?.source).toBe("/docs/q3-report.txt");
		expect(results[0]?.metadata.method).toBe("jikji");
		expect(JSON.stringify(results)).not.toContain(root);
	});

	it("can configure MinSync and Jikji independently", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			minSync: { binaryPath: join(root, "missing-minsync"), workspacePath: join(root, ".autorag", "minsync") },
			jikji: { binaryPath },
		});

		expect(methodNames(agent)).toEqual(["posix", "minsync", "jikji"]);
	});

	it("searchDocuments hides paths and numbered feedback records Jikji usefulness", async () => {
		writeFakeJikji(payload());
		const memoryPath = join(root, "memory.json");
		const agent = new AutoRAGAgent({ searchPaths: [docs], memoryPath, workspacePath: root, jikji: { binaryPath } });

		const response = await agent.searchDocuments("enterprise revenue", { topK: 1 });
		agent.recordFeedbackByNumbers(response.sessionId, [1]);

		expect(response.answer).toContain("[1]");
		expect(response.answer).toContain("Jikji says Q3 enterprise revenue increased");
		expect(response.answer).not.toContain(root);
		const memory = JSON.parse(readFileSync(memoryPath, "utf8")) as {
			entries: Array<{ method: string; outcome: string }>;
		};
		expect(memory.entries.some((entry) => entry.method === "jikji" && entry.outcome === "useful")).toBe(true);
	});

	it("continues merging Posix results when Jikji fails", async () => {
		writeFakeJikji({ paths: [123] });
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		const results = await agent.retrieve("document", { topK: 1 });

		expect(results[0]?.metadata.method).toBe("posix");
		expect(results[0]?.source).toBe("/docs/q3-report.txt");
	});
});
