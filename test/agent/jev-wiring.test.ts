import type { ExtensionAPI, ExtensionFactory, ToolDefinition } from "@earendil-works/pi-coding-agent";
import { MockBackend } from "jev-use";
import { describe, expect, it } from "vitest";
import { createJevExtension, JEV_TOOL_NAME } from "../../src/agent/jev-extension.ts";

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

	it("runs a batched judgment through the backend it is given", async () => {
		const [tool] = registeredTools(createJevExtension({ backend: new MockBackend() }));
		if (tool === undefined) throw new Error("expected the jev tool to register");
		const { details } = await runTool(tool, {
			state: "Production checkout is down and customers cannot pay.",
			questions: [{ id: "urgent", type: "noul", question: "Does this describe an urgent production incident?" }],
		});
		const verdicts = details.verdicts as { type: string; answer: unknown }[];
		expect(verdicts).toHaveLength(1);
		expect(verdicts[0]?.type).toBe("noul");
		expect(typeof verdicts[0]?.answer).toBe("number");
		expect(details.backend).toBe("mock");
	});
});
