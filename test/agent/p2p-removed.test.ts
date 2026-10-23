import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-agent-p2p-removed-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

interface AgentInternals {
	innerAgent: { state: { tools: AgentTool[] } };
}

function makeAgent(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [FIXTURE_DIR],
		workspacePath: tmpDir,
		memoryPath: join(tmpDir, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		webSearch: false,
	});
}

function toolNames(agent: AutoRAGAgent): string[] {
	// Test-only: the registered tool list lives on the private pi agent state.
	const internals = agent as unknown as AgentInternals;
	return internals.innerAgent.state.tools.map((tool) => tool.name);
}

describe("agent has no P2P surface", () => {
	it("registers no peer tools", () => {
		const names = toolNames(makeAgent());
		expect(names).not.toContain("recommend_peer_targets");
		expect(names).not.toContain("query_peer_agent");
	});

	it("keeps the core retrieval and memory tools", () => {
		const names = toolNames(makeAgent());
		expect(names).toContain("search_all_documents");
		expect(names).toContain("check_memory");
	});

	it("does not mention peers or SimpleX in the system prompt and keeps local file organization", () => {
		const prompt = makeAgent().getSystemPrompt();
		expect(prompt).not.toMatch(/simplex|query_peer_agent|recommend_peer_targets|remote\/P2P/i);
		expect(prompt).toContain("## Local File Organization");
	});

	it("is not a remote-session agent", () => {
		expect(makeAgent()).not.toHaveProperty("remoteSession");
	});
});
