import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-tool-merge-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

function makeTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} tool`,
		parameters: Type.Object({ query: Type.String() }),
		async execute() {
			return { content: [{ type: "text", text: "ok" }], details: { resultCount: 1, method: name, sources: [] } };
		},
	};
}

interface AgentInternals {
	innerAgent: {
		state: {
			tools: AgentTool[];
		};
	};
}

function toolNames(agent: AutoRAGAgent): string[] {
	const inner = (agent as unknown as AgentInternals).innerAgent;
	return inner.state.tools.map((tool) => tool.name);
}

describe("AutoRAGAgent built-in tool merge", () => {
	it("default prompt includes grep/find/read/ls/stat built-ins", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const prompt = agent.getSystemPrompt();
		for (const name of ["grep", "find", "read", "ls", "stat"]) {
			expect(prompt).toContain(name);
		}
		expect(prompt).toContain("AutoRAG-owned");
		expect(prompt).toContain("path-opaque");
	});

	it("default tool set contains the built-in tools exactly once", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const names = toolNames(agent);
		for (const name of ["grep", "find", "read", "ls", "stat"]) {
			expect(names.filter((n) => n === name)).toHaveLength(1);
		}
	});

	it("caller grep cannot shadow or remove the built-in", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("grep")],
		});
		const names = toolNames(agent);
		// caller grep dropped, built-in grep present exactly once
		expect(names.filter((n) => n === "grep")).toHaveLength(1);
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("content search (regex/literal)");
	});

	it("caller bash and read_file are dropped from tool names and prompt", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("bash"), makeTool("read_file")],
		});
		const names = toolNames(agent);
		expect(names).not.toContain("bash");
		expect(names).not.toContain("read_file");
		const prompt = agent.getSystemPrompt();
		expect(prompt).not.toContain("**bash**");
		expect(prompt).not.toContain("real-path search/navigation fallback");
	});

	it("preserves a non-reserved caller search tool while dropping bash/read_file", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("search_custom"), makeTool("bash"), makeTool("read_file"), makeTool("grep")],
		});
		const names = toolNames(agent);
		expect(names).toContain("search_custom");
		expect(names).not.toContain("bash");
		expect(names).not.toContain("read_file");
		// built-in grep wins; caller grep dropped
		expect(names.filter((n) => n === "grep")).toHaveLength(1);
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("search_custom");
		expect(prompt).toContain("caller-provided retrieval tool");
	});

	it("tool names are unique across the full merged set", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [
				makeTool("grep"),
				makeTool("find"),
				makeTool("check_memory"),
				makeTool("emit_autorag_results"),
				makeTool("search_posix_documents"),
				makeTool("search_minsync_documents"),
				makeTool("search_all_documents"),
				makeTool("search_custom"),
			],
		});
		const names = toolNames(agent);
		expect(new Set(names).size).toBe(names.length);
		expect(names).toContain("search_custom");
		// reserved names appear exactly once (built-in/agent-owned wins)
		for (const reserved of [
			"grep",
			"find",
			"check_memory",
			"emit_autorag_results",
			"search_posix_documents",
			"search_minsync_documents",
			"search_all_documents",
		]) {
			expect(names.filter((n) => n === reserved)).toHaveLength(1);
		}
	});
});
