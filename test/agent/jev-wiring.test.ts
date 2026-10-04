import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createJevExtension } from "../../src/agent/jev-extension.ts";
import { JEV_TOOL_NAME } from "../../src/jev/index.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string;

beforeEach(() => {
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-jev-wiring-"));
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

/** Runs an extension factory against a minimal ExtensionAPI and returns registered tools. */
function registeredTools(factory: ExtensionFactory): ToolDefinition[] {
	const tools: ToolDefinition[] = [];
	const api = { registerTool: (tool: ToolDefinition) => tools.push(tool) } as unknown as ExtensionAPI;
	factory(api);
	return tools;
}

describe("jev pi extension", () => {
	it("registers exactly one tool under the reserved jev name", () => {
		const tools = registeredTools(createJevExtension({ backend: "openrouter" }));
		expect(tools).toHaveLength(1);
		expect(tools[0]?.name).toBe(JEV_TOOL_NAME);
		expect(tools[0]?.description).toContain("Jev");
		// pi rejects non-object parameter schemas, so the object schema is load-critical.
		expect(typeof tools[0]?.parameters).toBe("object");
	});

	it("advertises the jev tool in the agent system prompt only when enabled", () => {
		const enabled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: { backend: "openrouter" },
		});
		expect(enabled.getSystemPrompt()).toContain("**jev**");

		const disabled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: false,
		});
		expect(disabled.getSystemPrompt()).not.toContain("**jev**");

		const off = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
		});
		expect(off.getSystemPrompt()).not.toContain("**jev**");
	});

	it("omits the jev prompt line for remote P2P sessions even when enabled", () => {
		const remote = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			jev: {},
			remoteSession: true,
		});
		expect(remote.getSystemPrompt()).not.toContain("**jev**");
	});
});
