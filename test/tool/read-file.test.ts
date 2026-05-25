import { resolve } from "node:path";
import { describe, expect, it } from "vitest";
import { createReadFileTool } from "../../src/tool/read-file.ts";

const FIXTURE_DIR = resolve("test/fixtures/sample-project");

describe("createReadFileTool", () => {
	it("returns valid AgentTool shape", () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		expect(tool.name).toBe("read_file");
		expect(tool.label).toBe("Read File");
		expect(typeof tool.description).toBe("string");
		expect(tool.parameters).toBeDefined();
		expect(typeof tool.execute).toBe("function");
	});

	it("reads known fixture file with line numbers", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/src/main.ts` });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("1:");
		expect(text).toContain("function");
	});

	it("reads with startLine and endLine range", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/src/utils.ts`, startLine: 1, endLine: 1 });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		const lines = text.trim().split("\n");
		expect(lines.length).toBe(1);
		expect(lines[0]).toMatch(/^1:/);
	});

	it("reads from startLine to EOF when endLine omitted", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/src/utils.ts`, startLine: 2 });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toMatch(/^2:/);
		expect(text).not.toMatch(/^1:/m);
	});

	it("returns error text for nonexistent file", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/nonexistent.ts` });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Error");
	});

	it("returns error text for directory path", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/src` });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Error");
	});

	it("rejects paths outside searchPaths", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: "/etc/passwd" });
		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("outside");
	});

	it("details contains path and lineCount", async () => {
		const tool = createReadFileTool({ searchPaths: [FIXTURE_DIR] });
		const result = await tool.execute("test", { path: `${FIXTURE_DIR}/src/main.ts` });
		expect(result.details).toBeDefined();
		expect(result.details!.path).toContain("main.ts");
		expect(typeof result.details!.lineCount).toBe("number");
	});
});
