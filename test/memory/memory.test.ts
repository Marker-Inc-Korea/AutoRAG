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
		const entry = memory.append({ query: "test", method: "posix", outcome: "useful" });
		expect(entry.id).toBeDefined();
		expect(typeof entry.id).toBe("string");
		expect(entry.timestamp).toBeGreaterThan(0);
		expect(entry.outcome).toBe("useful");
	});

	it("append() adds entry to in-memory entries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getEntries().length).toBe(0);
		memory.append({ query: "q1", method: "posix", outcome: "useful" });
		memory.append({ query: "q2", method: "vector", outcome: "not_useful" });
		expect(memory.getEntries().length).toBe(2);
	});

	it("getEntries() returns all appended entries in order", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.append({ query: "first", method: "posix", outcome: "useful" });
		memory.append({ query: "second", method: "vector", outcome: "not_useful" });
		const entries = memory.getEntries();
		expect(entries[0].query).toBe("first");
		expect(entries[1].query).toBe("second");
	});

	it("save() caps entries at 500", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 510; i++) {
			memory.append({ query: `query-${i}`, method: "posix", outcome: "useful" });
		}
		memory.save();
		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		expect(memory2.getEntries().length).toBe(500);
	});

	it("load() migrates v1 format with chained V1→V2→V3", () => {
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
		expect(entries.filter((e) => e.method === "posix" && e.outcome === "useful").length).toBe(3);
		expect(entries.filter((e) => e.method === "posix" && e.outcome === "not_useful").length).toBe(1);
		expect(entries.filter((e) => e.method === "vector" && e.outcome === "not_useful").length).toBe(2);
	});

	it("load() saves migrated data as v3", () => {
		const v1Data = { patterns: { "code search": { posix: { success: 1, fail: 0 } } } };
		writeFileSync(memoryPath, JSON.stringify(v1Data), "utf-8");
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const raw = JSON.parse(readFileSync(memoryPath, "utf-8"));
		expect(raw.version).toBe(3);
	});

	it("recordFeedback is backward-compatible with append", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("test query", "bm25", false);
		const entries = memory.getEntries();
		expect(entries.length).toBe(1);
		expect(entries[0].method).toBe("bm25");
		expect(entries[0].outcome).toBe("not_useful");
	});

	it("registerAttempt() stores attempt and resolves via feedback", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "test", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "test",
			method: "posix",
			sources: ["src/a.ts", "src/b.ts"],
			timestamp: entry.timestamp,
		});
		memory.recordResultFeedback([{ source: "src/a.ts", useful: true }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("recordResultFeedback() resolves pending → useful", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["file.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([{ source: "file.ts", useful: true }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("recordResultFeedback() resolves pending → not_useful", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["file.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([{ source: "file.ts", useful: false }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("not_useful");
	});

	it("recordResultFeedback() useful wins over not_useful for same attempt", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["a.ts", "b.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([
			{ source: "a.ts", useful: false },
			{ source: "b.ts", useful: true },
		]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("recordResultFeedback() ignores unknown sources", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["file.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([{ source: "unknown.ts", useful: true }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("pending");
	});

	it("recordResultFeedback() does not change already-resolved entries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "useful" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["file.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([{ source: "file.ts", useful: false }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("resolvePendingEntries() bulk-resolves by query across all methods", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.append({ query: "test", method: "posix", outcome: "pending" });
		memory.append({ query: "test", method: "vector", outcome: "pending" });
		memory.append({ query: "other", method: "posix", outcome: "pending" });
		memory.resolvePendingEntries("test", null, "useful");
		const entries = memory.getEntries();
		expect(entries[0].outcome).toBe("useful");
		expect(entries[1].outcome).toBe("useful");
		expect(entries[2].outcome).toBe("pending");
	});

	it("getMethodPriority() ignores pending entries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.append({ query: "test", method: "posix", outcome: "pending" });
		memory.append({ query: "test", method: "posix", outcome: "useful" });
		const priority = memory.getMethodPriority("test");
		expect(priority[0].score).toBe(1.0);
	});

	it("V2→V3 migration maps success→useful and failure→not_useful", () => {
		const v2Data = {
			version: 2,
			entries: [
				{ id: "a", query: "q", method: "posix", outcome: "success", timestamp: 1000 },
				{ id: "b", query: "q", method: "posix", outcome: "failure", timestamp: 1001 },
			],
		};
		writeFileSync(memoryPath, JSON.stringify(v2Data), "utf-8");
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entries = memory.getEntries();
		expect(entries[0].outcome).toBe("useful");
		expect(entries[1].outcome).toBe("not_useful");
	});
});
