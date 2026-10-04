import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { JEV_TOOL_NAME } from "../../src/jev/index.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-jev-wiring-"));
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

describe("AutoRAGAgent jev tool surface", () => {
	it("omits the jev tool by default", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(toolNames(agent)).not.toContain(JEV_TOOL_NAME);
	});

	it("registers the jev tool exactly once when enabled", () => {
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: { backend: "openrouter" },
		});
		const names = toolNames(agent);
		expect(names.filter((name) => name === JEV_TOOL_NAME)).toHaveLength(1);
		const inner = (agent as unknown as AgentInternals).innerAgent;
		const jevTool = inner.state.tools.find((tool) => tool.name === JEV_TOOL_NAME);
		expect(jevTool?.description).toContain("Jev");
	});

	it("omits the jev tool when disabled and for remote P2P sessions", () => {
		const disabled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: false,
		});
		expect(toolNames(disabled)).not.toContain(JEV_TOOL_NAME);

		const remote = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: {},
			remoteSession: true,
		});
		expect(toolNames(remote)).not.toContain(JEV_TOOL_NAME);
	});

	it("drops a caller-provided tool that collides with the reserved jev name", () => {
		const callerTool: AgentTool = {
			name: JEV_TOOL_NAME,
			label: "caller jev",
			description: "caller stub",
			parameters: Type.Object({ state: Type.String() }),
			async execute() {
				return { content: [{ type: "text", text: "caller" }], details: {} };
			},
		};
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: { backend: "openrouter" },
			tools: [callerTool],
		});
		const names = toolNames(agent);
		expect(names.filter((name) => name === JEV_TOOL_NAME)).toHaveLength(1);
		const inner = (agent as unknown as AgentInternals).innerAgent;
		const jevTool = inner.state.tools.find((tool) => tool.name === JEV_TOOL_NAME);
		expect(jevTool?.description).toContain("Jev");
	});
});
