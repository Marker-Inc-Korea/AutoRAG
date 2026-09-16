import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { buildSystemPrompt } from "../../src/agent/system-prompt.ts";
import { WEB_FETCH_TOOL_NAME } from "../../src/agent/web-fetch-tool.ts";
import { WEB_SEARCH_TOOL_NAME } from "../../src/agent/web-search-tool.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-web-wiring-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

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

function makeTool(name: string): AgentTool {
	return {
		name,
		label: name,
		description: `${name} tool`,
		parameters: Type.Object({ query: Type.String() }),
		async execute() {
			return { content: [{ type: "text", text: "ok" }], details: {} };
		},
	};
}

describe("AutoRAGAgent web tool surface", () => {
	it("registers web_search and web_fetch exactly once by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		const names = toolNames(agent);
		expect(names.filter((n) => n === WEB_SEARCH_TOOL_NAME)).toHaveLength(1);
		expect(names.filter((n) => n === WEB_FETCH_TOOL_NAME)).toHaveLength(1);
	});

	it("drops caller-provided tools that collide with the reserved web tool names", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			tools: [makeTool(WEB_SEARCH_TOOL_NAME), makeTool(WEB_FETCH_TOOL_NAME)],
		});
		const names = toolNames(agent);
		expect(names.filter((n) => n === WEB_SEARCH_TOOL_NAME)).toHaveLength(1);
		expect(names.filter((n) => n === WEB_FETCH_TOOL_NAME)).toHaveLength(1);
		// AutoRAG's own descriptions win, not the caller-provided stub.
		const inner = (agent as unknown as AgentInternals).innerAgent;
		const searchTool = inner.state.tools.find((tool) => tool.name === WEB_SEARCH_TOOL_NAME);
		expect(searchTool?.description).toContain("Web search");
	});

	it("omits both web tools when webSearch is disabled", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			webSearch: false,
		});
		const names = toolNames(agent);
		expect(names).not.toContain(WEB_SEARCH_TOOL_NAME);
		expect(names).not.toContain(WEB_FETCH_TOOL_NAME);
	});

	it("omits both web tools for remote P2P sessions", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			remoteSession: true,
		});
		const names = toolNames(agent);
		expect(names).not.toContain(WEB_SEARCH_TOOL_NAME);
		expect(names).not.toContain(WEB_FETCH_TOOL_NAME);
	});
});

describe("system prompt web guidance", () => {
	const baseConfig = { toolNames: [], manifests: [] };

	it("mentions web_search and web_fetch when the tools are registered", () => {
		const prompt = buildSystemPrompt({
			...baseConfig,
			toolNames: [WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME],
		});
		expect(prompt).toContain(`- **${WEB_SEARCH_TOOL_NAME}**`);
		expect(prompt).toContain(`- **${WEB_FETCH_TOOL_NAME}**`);
		expect(prompt).toContain("Web Research");
	});

	it("omits web guidance when the tools are absent", () => {
		const prompt = buildSystemPrompt(baseConfig);
		expect(prompt).not.toContain(`- **${WEB_SEARCH_TOOL_NAME}**`);
		expect(prompt).not.toContain("Web Research");
	});

	it("forbids passing local paths and datasource virtual ids to web_fetch", () => {
		const prompt = buildSystemPrompt({
			...baseConfig,
			toolNames: [WEB_SEARCH_TOOL_NAME, WEB_FETCH_TOOL_NAME],
		});
		expect(prompt).toContain("http");
		expect(prompt).toMatch(/web_fetch[^\n]*\n([\s\S]*?)(local|virtual)/i);
	});
});
