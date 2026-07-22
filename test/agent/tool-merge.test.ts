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

describe("AutoRAGAgent bash-based tool surface", () => {
	it("default tool set contains bash exactly once and no deleted builtins", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const names = toolNames(agent);
		expect(names.filter((n) => n === "bash")).toHaveLength(1);
		// the old builtin/posix tools were deleted; they must never appear
		for (const name of ["grep", "find", "read", "ls", "stat", "search_posix_documents"]) {
			expect(names).not.toContain(name);
		}
		// the always-present search_* + structural tools are registered
		for (const name of [
			"check_memory",
			"search_bm25_documents",
			"search_minsync_documents",
			"search_all_documents",
			"search_datasource_documents",
			"emit_autorag_results",
		]) {
			expect(names).toContain(name);
		}
	});

	it("a caller-provided bash tool is dropped while AutoRAG's own bash remains once", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("bash")],
		});
		const names = toolNames(agent);
		// caller bash dropped (reserved); AutoRAG bash present exactly once
		expect(names.filter((n) => n === "bash")).toHaveLength(1);
	});

	it("a caller-provided read_file tool is preserved (no longer dropped)", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("read_file")],
		});
		const names = toolNames(agent);
		expect(names).toContain("read_file");
		expect(names.filter((n) => n === "read_file")).toHaveLength(1);
	});

	it("preserves a non-reserved caller search tool", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("search_custom")],
		});
		const names = toolNames(agent);
		expect(names).toContain("search_custom");
		const prompt = agent.getSystemPrompt();
		expect(prompt).toContain("search_custom");
		expect(prompt).toContain("caller-provided retrieval tool");
	});

	it("a caller-provided reserved search tool (search_all_documents) is dropped", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool("search_all_documents")],
		});
		const names = toolNames(agent);
		// caller copy dropped; AutoRAG's own search_all_documents present exactly once
		expect(names.filter((n) => n === "search_all_documents")).toHaveLength(1);
	});

	it("tool names are unique across the full merged set", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [
				makeTool("bash"),
				makeTool("read_file"),
				makeTool("check_memory"),
				makeTool("emit_autorag_results"),
				makeTool("search_all_documents"),
				makeTool("search_minsync_documents"),
				makeTool("search_custom"),
			],
		});
		const names = toolNames(agent);
		expect(new Set(names).size).toBe(names.length);
		// non-reserved caller tool survives
		expect(names).toContain("search_custom");
		// read_file is now preserved alongside the reserved AutoRAG tools
		expect(names).toContain("read_file");
		// reserved names appear exactly once (AutoRAG-owned wins over any caller copy)
		for (const reserved of [
			"bash",
			"check_memory",
			"emit_autorag_results",
			"search_all_documents",
			"search_minsync_documents",
		]) {
			expect(names.filter((n) => n === reserved)).toHaveLength(1);
		}
	});
});
