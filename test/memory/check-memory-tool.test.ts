import { mkdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { createCheckMemoryTool } from "../../src/memory/check-memory-tool.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-checkmem-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("createCheckMemoryTool", () => {
	it("returns summary with recommendation for matching query", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "posix", true);

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "find typescript files" });

		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("Retrieval Memory");
		expect(text).toContain("posix");
		expect(text).toContain("Recommended Methods");
	});

	it("returns 'No retrieval history' for cold start", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "anything" });

		const text = (result.content[0] as { type: "text"; text: string }).text;
		expect(text).toContain("No retrieval history available.");
	});

	it("does not modify memory state", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("test", "posix", true);
		const countBefore = memory.getEntries().length;

		const tool = createCheckMemoryTool(memory);
		await tool.execute("test-call", { query: "test" });

		expect(memory.getEntries().length).toBe(countBefore);
	});

	it("details contains entryCount and topMethod", async () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("query", "posix", true);

		const tool = createCheckMemoryTool(memory);
		const result = await tool.execute("test-call", { query: "query" });

		expect(result.details).toBeDefined();
		expect(result.details!.entryCount).toBe(1);
		expect(result.details!.topMethod).toBe("posix");
	});
});
