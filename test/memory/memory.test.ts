import { existsSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { RetrievalMemory } from "../../src/memory/memory.ts";

let tmpDir: string;
let memoryPath: string;

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-memory-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("RetrievalMemory", () => {
	it("starts with empty state when file does not exist", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getMethodPriority("test query")).toEqual([]);
	});

	it("records feedback and updates in-memory state", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("search code files", "posix", true);
		const priority = memory.getMethodPriority("search code files");
		expect(priority.length).toBeGreaterThan(0);
		expect(priority[0].method).toBe("posix");
	});

	it("persists data to disk with save()", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript functions", "posix", true);
		memory.save();
		expect(existsSync(memoryPath)).toBe(true);
	});

	it("loads persisted data after restart", () => {
		const memory1 = new RetrievalMemory({ storagePath: memoryPath });
		memory1.load();
		memory1.recordFeedback("code search query", "posix", true);
		memory1.recordFeedback("code search query", "posix", true);
		memory1.recordFeedback("code search query", "posix", true);
		memory1.save();

		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		const priority = memory2.getMethodPriority("code search query");
		expect(priority.length).toBeGreaterThan(0);
		expect(priority[0].method).toBe("posix");
		expect(priority[0].score).toBeGreaterThan(0);
	});

	it("handles corrupted memory file gracefully", () => {
		writeFileSync(memoryPath, "not valid json {{{", "utf-8");
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		expect(() => memory.load()).not.toThrow();
		expect(memory.getMethodPriority("test")).toEqual([]);
	});

	it("handles missing memory file gracefully", () => {
		const memory = new RetrievalMemory({ storagePath: join(tmpDir, "nonexistent.json") });
		expect(() => memory.load()).not.toThrow();
		expect(memory.getMethodPriority("test")).toEqual([]);
	});

	it("returns empty priority for cold start (no feedback)", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const priority = memory.getMethodPriority("find all documents");
		expect(priority).toEqual([]);
	});

	it("matches similar queries by keyword overlap", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "vector", false);

		const priority = memory.getMethodPriority("search typescript code");
		const posixEntry = priority.find((p) => p.method === "posix");
		const vectorEntry = priority.find((p) => p.method === "vector");
		expect(posixEntry).toBeDefined();
		if (posixEntry && vectorEntry) {
			expect(posixEntry.score).toBeGreaterThan(vectorEntry.score);
		}
	});
});
