import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JIKJI_FIND_TOOL_NAME } from "../../src/agent/jikji-find-tool.ts";

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

/**
 * Fake jikji binary that branches on argv[0] (the subcommand):
 * - `find`: prints a hand-authored answer-pack JSON
 * - `prepare`: prints a prepare summary {root,files,folders,deleted,agent_map}
 */
function writeFakeJikji(answerPack?: Record<string, unknown>): void {
	const pack = answerPack ?? {
		answer_paths: ["q3-report.txt"],
		paths: ["q3-report.txt"],
		candidates: [{ path: "q3-report.txt", next_read: "original", label: "Q3 report" }],
		evidence_pack: [{ path: "q3-report.txt", next_read: "original" }],
		handoff_action: "direct_use",
		tool_call_policy: {
			stop_after_find: true,
			forbidden_tools: ["bash"],
			allowed_followups: [],
		},
		agent_should_not_rerank: true,
	};
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
if (args[0] === "find") {
	console.log(JSON.stringify(${JSON.stringify(pack)}));
} else {
	console.log(JSON.stringify({ prepared: true, root: args[1], files: 1, folders: 0, deleted: 0, agent_map: false }));
}
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

function toolNames(agent: AutoRAGAgent): string[] {
	return (agent as unknown as { innerAgent: { state: { tools: AgentTool[] } } }).innerAgent.state.tools.map(
		(tool) => tool.name,
	);
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

		expect(methodNames(agent)).not.toContain("jikji");
	});

	it("keeps Jikji out of the retrieval registry when configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		expect(methodNames(agent)).not.toContain("jikji");
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

		expect(methodNames(agent)).toEqual(["minsync", "bm25"]);
	});

	it("registers jikji_find tool when jikji is configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		expect(toolNames(agent)).toContain(JIKJI_FIND_TOOL_NAME);
	});

	it("does NOT register jikji_find when jikji is not configured", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		expect(toolNames(agent)).not.toContain(JIKJI_FIND_TOOL_NAME);
	});

	it("runs prepare --json during refresh and uses the new local-discovery prompt section", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		await agent.refresh(true);

		const prepareCalls = loggedArgs().filter((args) => args[0] === "prepare");
		expect(prepareCalls.length).toBeGreaterThanOrEqual(1);
		for (const call of prepareCalls) {
			expect(call).toContain("--json");
			// --no-agent-rules is emitted by default (writeAgentRules !== true)
			expect(call).toContain("--no-agent-rules");
		}

		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("## Jikji Local Discovery");
	});

	it("adds prompt guidance for Jikji local discovery section", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		const prompt = agent.getSystemPrompt();

		expect(prompt).toContain("## Jikji Local Discovery");
		expect(prompt).toContain("jikji_find");
	});

	it("emits ['find', root, query, '--json'] when findJikji is called and returns answer_paths", async () => {
		writeFakeJikji();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath },
		});

		await agent.refresh(true);
		// Clear logs from prepare so we can isolate the find call
		writeFileSync(logPath, "");

		const result = await agent.findJikji("Q3 report", { topK: 5 });

		const calls = loggedArgs();
		expect(calls.length).toBeGreaterThanOrEqual(1);
		const findCall = calls.find((args) => args[0] === "find");
		expect(findCall).toBeDefined();
		expect(findCall?.[1]).toBe(realpathSync(docs));
		expect(findCall?.[2]).toBe("Q3 report");
		expect(findCall).toContain("--json");

		expect(result.answerPack).toBeDefined();
		expect(result.answerPack?.answerPaths.length).toBeGreaterThan(0);
		expect(result.policy).toBeDefined();
		expect(result.policy?.handoffAction).toBe("direct_use");
		expect(result.policy?.stopAfterFind).toBe(true);
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

	it("returns jikji-unavailable result and leaves policy undefined when binary is missing", async () => {
		const missingBinary = join(root, "does-not-exist-jikji");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: { binaryPath: missingBinary },
		});

		const result = await agent.findJikji("anything");
		expect(result.answerPack).toBeUndefined();
		expect(result.policy).toBeUndefined();
		expect(result.diagnostics.length).toBeGreaterThan(0);
		expect(result.diagnostics[0]?.code).toBe("jikji-unavailable");
	});
});
