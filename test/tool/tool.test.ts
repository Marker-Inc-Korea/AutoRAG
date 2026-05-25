import { describe, expect, it } from "vitest";
import { createAutoRAGTool } from "../../src/tool/tool.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";

describe("createAutoRAGTool", () => {
	it("returns a valid AgentTool shape", () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		expect(tool.name).toBe("autorag_search");
		expect(tool.label).toBe("AutoRAG Search");
		expect(typeof tool.description).toBe("string");
		expect(tool.parameters).toBeDefined();
		expect(typeof tool.execute).toBe("function");
	});

	it("tool has correct parameter schema", () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const schema = tool.parameters;
		expect(schema.type).toBe("object");
		expect(schema.properties).toHaveProperty("query");
		expect(schema.properties).toHaveProperty("topK");
		expect(schema.properties).toHaveProperty("scope");
	});

	it("execute returns AgentToolResult with content and details", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test-id", { query: "function" }, undefined, undefined);
		expect(result.content).toBeDefined();
		expect(result.content.length).toBeGreaterThan(0);
		expect(result.content[0].type).toBe("text");
		expect(result.details).toBeDefined();
		expect(typeof result.details.resultCount).toBe("number");
		expect(Array.isArray(result.details.methodsUsed)).toBe(true);
		expect(typeof result.details.elapsedMs).toBe("number");
	});

	it("execute returns numbered results for known content", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test-id", { query: "function" }, undefined, undefined);
		expect(result.details.resultCount).toBeGreaterThan(0);
		expect(result.details.methodsUsed).toContain("posix");
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toMatch(/^\[1\]/);
		expect(result.details.numberedResults).toBeDefined();
		expect(result.details.numberedResults.length).toBeGreaterThan(0);
		expect(result.details.numberedResults[0].index).toBe(1);
		expect(result.details.numberedResults[0].method).toBe("posix");
	});

	it("execute handles empty results gracefully", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test-id", { query: "absolutely_nonexistent_xyz_12345" }, undefined, undefined);
		expect(result.details.resultCount).toBe(0);
		expect(result.content[0].type).toBe("text");
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("No results found");
		expect(result.details.numberedResults).toEqual([]);
	});

	it("numbered results have sequential 1-based indices with method attribution", async () => {
		const tool = createAutoRAGTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test-id", { query: "function" }, undefined, undefined);
		const nr = result.details.numberedResults;
		for (let i = 0; i < nr.length; i++) {
			expect(nr[i].index).toBe(i + 1);
			expect(nr[i].source).toBeTruthy();
			expect(nr[i].content).toBeTruthy();
			expect(nr[i].method).toBeTruthy();
		}
	});
});
