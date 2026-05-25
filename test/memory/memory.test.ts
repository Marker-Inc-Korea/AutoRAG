import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
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
		expect(memory.getEntries()).toEqual([]);
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

	it("ranks methods by success rate for matching queries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "vector", false);

		const priority = memory.getMethodPriority("find typescript files");
		const posixEntry = priority.find((p) => p.method === "posix");
		const vectorEntry = priority.find((p) => p.method === "vector");
		expect(posixEntry).toBeDefined();
		expect(vectorEntry).toBeDefined();
		if (posixEntry && vectorEntry) {
			expect(posixEntry.score).toBeGreaterThan(vectorEntry.score);
		}
	});

	it("matches queries by substring containment", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files in project", "posix", true);

		const priority = memory.getMethodPriority("typescript files");
		expect(priority.length).toBeGreaterThan(0);
		expect(priority[0].method).toBe("posix");
	});

	it("append() creates entry with id and timestamp", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "test", method: "posix", outcome: "success" });
		expect(entry.id).toBeDefined();
		expect(typeof entry.id).toBe("string");
		expect(entry.timestamp).toBeGreaterThan(0);
		expect(entry.query).toBe("test");
		expect(entry.method).toBe("posix");
		expect(entry.outcome).toBe("success");
	});

	it("append() adds entry to in-memory entries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getEntries().length).toBe(0);
		memory.append({ query: "q1", method: "posix", outcome: "success" });
		memory.append({ query: "q2", method: "vector", outcome: "failure" });
		expect(memory.getEntries().length).toBe(2);
	});

	it("getEntries() returns all appended entries in order", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.append({ query: "first", method: "posix", outcome: "success" });
		memory.append({ query: "second", method: "vector", outcome: "failure" });
		const entries = memory.getEntries();
		expect(entries[0].query).toBe("first");
		expect(entries[1].query).toBe("second");
	});

	it("save() caps entries at 500", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 510; i++) {
			memory.append({ query: `query-${i}`, method: "posix", outcome: "success" });
		}
		expect(memory.getEntries().length).toBe(510);
		memory.save();

		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		expect(memory2.getEntries().length).toBe(500);
		expect(memory2.getEntries()[0].query).toBe("query-10");
	});

	it("load() migrates v1 format automatically", () => {
		const v1Data = {
			patterns: {
				"files find typescript": {
					posix: { success: 3, fail: 1 },
					vector: { success: 0, fail: 2 },
				},
			},
		};
		writeFileSync(memoryPath, JSON.stringify(v1Data), "utf-8");

		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const entries = memory.getEntries();
		expect(entries.length).toBe(6);

		const posixSuccess = entries.filter((e) => e.method === "posix" && e.outcome === "success");
		const posixFailure = entries.filter((e) => e.method === "posix" && e.outcome === "failure");
		const vectorFailure = entries.filter((e) => e.method === "vector" && e.outcome === "failure");
		expect(posixSuccess.length).toBe(3);
		expect(posixFailure.length).toBe(1);
		expect(vectorFailure.length).toBe(2);
	});

	it("load() saves migrated v1 data as v2", () => {
		const v1Data = {
			patterns: { "code search": { posix: { success: 1, fail: 0 } } },
		};
		writeFileSync(memoryPath, JSON.stringify(v1Data), "utf-8");

		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();

		const raw = JSON.parse(readFileSync(memoryPath, "utf-8"));
		expect(raw.version).toBe(2);
		expect(Array.isArray(raw.entries)).toBe(true);
	});

	it("recordFeedback is backward-compatible with append", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("test query", "bm25", false);
		const entries = memory.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0].query).toBe("test query");
		expect(entries[0].method).toBe("bm25");
		expect(entries[0].outcome).toBe("failure");
	});
});
