import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI, ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { createJevExtension, JEV_TOOL_NAME } from "../../src/agent/jev-extension.ts";

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

/** Invokes a registered pi tool without the TUI-only execute arguments. */
async function runTool(tool: ToolDefinition, params: unknown): Promise<{ details: Record<string, unknown> }> {
	const execute = tool.execute as unknown as (
		id: string,
		params: unknown,
	) => Promise<{ details: Record<string, unknown> }>;
	return execute("test-call", params);
}

describe("jev pi extension", () => {
	it("registers exactly one tool under the reserved jev name", () => {
		const tools = registeredTools(createJevExtension());
		expect(tools).toHaveLength(1);
		expect(tools[0]?.name).toBe(JEV_TOOL_NAME);
		expect(tools[0]?.description).toContain("Jev");
		// pi rejects non-object parameter schemas, so the object schema is load-critical.
		expect(typeof tools[0]?.parameters).toBe("object");
	});

	it("runs a batched judgment through jev-use's keyless mock backend", async () => {
		const previous = process.env.JEV_BACKEND;
		process.env.JEV_BACKEND = "mock";
		try {
			const [tool] = registeredTools(createJevExtension());
			if (tool === undefined) throw new Error("expected the jev tool to register");
			const { details } = await runTool(tool, {
				state: "Production checkout is down and customers cannot pay.",
				questions: [{ id: "urgent", type: "noul", question: "Does this describe an urgent production incident?" }],
			});
			const verdicts = details.verdicts as { type: string; answer: unknown }[];
			expect(verdicts).toHaveLength(1);
			expect(verdicts[0]?.type).toBe("noul");
			expect(typeof verdicts[0]?.answer).toBe("number");
			expect(typeof details.escalated).toBe("boolean");
		} finally {
			if (previous === undefined) {
				delete process.env.JEV_BACKEND;
			} else {
				process.env.JEV_BACKEND = previous;
			}
		}
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
