import { describe, expect, it } from "vitest";
import { createAutoRAGTool } from "../../src/tool/tool.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";

describe("Tool integration", () => {
	it("createAutoRAGTool returns a working AgentTool", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		expect(tool.name).toBe("autorag_search");
		const result = await tool.execute("test", { query: "function" });
		expect(result.content[0].type).toBe("text");
		expect(result.details.resultCount).toBeGreaterThan(0);
		expect(result.details.methodsUsed).toContain("posix");
	});

	it("tool returns correct AgentToolResult shape", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { query: "export" });
		expect(Array.isArray(result.content)).toBe(true);
		expect(typeof result.details).toBe("object");
		expect(typeof result.details.resultCount).toBe("number");
		expect(Array.isArray(result.details.methodsUsed)).toBe(true);
		expect(typeof result.details.elapsedMs).toBe("number");
	});

	it("tool handles empty results gracefully", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { query: "absolutely_nonexistent_xyz_12345" });
		expect(result.details.resultCount).toBe(0);
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("No results found");
	});
});
