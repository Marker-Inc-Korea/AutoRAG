import { describe, expect, it } from "vitest";
import type { MemoryEntry } from "../../src/memory/memory.ts";
import { renderMemoryContext } from "../../src/memory/renderer.ts";

function makeEntry(overrides: Partial<MemoryEntry> & Pick<MemoryEntry, "query" | "method" | "outcome">): MemoryEntry {
	return {
		id: `test-${Math.random().toString(36).slice(2)}`,
		timestamp: Date.now(),
		...overrides,
	};
}

describe("renderMemoryContext", () => {
	it("returns 'No retrieval history' for empty entries", () => {
		expect(renderMemoryContext([])).toBe("No retrieval history available.");
	});

	it("renders single entry as markdown table", () => {
		const entries = [makeEntry({ query: "find files", method: "posix", outcome: "useful" })];
		const result = renderMemoryContext(entries);
		expect(result).toContain("## Retrieval Memory");
		expect(result).toContain("| Useful | Not Useful |");
		expect(result).toContain("| find files | posix | 1 | 0 |");
	});

	it("groups multiple entries for same query", () => {
		const entries = [
			makeEntry({ query: "search code", method: "posix", outcome: "useful" }),
			makeEntry({ query: "search code", method: "posix", outcome: "useful" }),
			makeEntry({ query: "search code", method: "posix", outcome: "not_useful" }),
		];
		const result = renderMemoryContext(entries);
		expect(result).toContain("| search code | posix | 2 | 1 |");
	});

	it("renders multiple methods for same query as separate rows", () => {
		const entries = [
			makeEntry({ query: "find docs", method: "posix", outcome: "useful" }),
			makeEntry({ query: "find docs", method: "vector", outcome: "not_useful" }),
		];
		const result = renderMemoryContext(entries);
		expect(result).toContain("| find docs | posix | 1 | 0 |");
		expect(result).toContain("| find docs | vector | 0 | 1 |");
	});

	it("sorts by most recent activity", () => {
		const old = makeEntry({ query: "old query", method: "posix", outcome: "useful", timestamp: 1000 });
		const recent = makeEntry({ query: "new query", method: "posix", outcome: "useful", timestamp: 9000 });
		const result = renderMemoryContext([old, recent]);
		const oldIdx = result.indexOf("old query");
		const newIdx = result.indexOf("new query");
		expect(newIdx).toBeLessThan(oldIdx);
	});

	it("caps at maxGroups", () => {
		const entries: MemoryEntry[] = [];
		for (let i = 0; i < 60; i++) {
			entries.push(makeEntry({ query: `query-${i}`, method: "posix", outcome: "useful", timestamp: i }));
		}
		const result = renderMemoryContext(entries, { maxGroups: 50 });
		const rowCount = result.split("\n").filter((line) => line.startsWith("| query-")).length;
		expect(rowCount).toBe(50);
	});

	it("formats date correctly", () => {
		const ts = new Date("2026-05-25T12:00:00Z").getTime();
		const entries = [makeEntry({ query: "test", method: "posix", outcome: "useful", timestamp: ts })];
		const result = renderMemoryContext(entries);
		expect(result).toContain("2026-05-25");
	});

	it("renders pending count as suffix", () => {
		const entries = [
			makeEntry({ query: "test", method: "posix", outcome: "useful" }),
			makeEntry({ query: "test", method: "posix", outcome: "pending" }),
		];
		const result = renderMemoryContext(entries);
		expect(result).toContain("(1 pending)");
	});
});
