import { describe, expect, it } from "vitest";
import type { MethodHint } from "../../src/memory/memory.ts";
import { renderMemoryContext } from "../../src/memory/renderer.ts";

function hint(overrides: Partial<MethodHint> & Pick<MethodHint, "method" | "score">): MethodHint {
	return {
		confidence: 0.5,
		reason: "matched feedback",
		...overrides,
	};
}

describe("renderMemoryContext", () => {
	it("returns no-hints text for empty hints", () => {
		expect(renderMemoryContext([])).toBe("No retrieval memory hints available.");
	});

	it("renders method hints as an advisory markdown table", () => {
		const result = renderMemoryContext([hint({ method: "bm25", score: 1 })]);
		expect(result).toContain("## Retrieval Memory Hints");
		expect(result).toContain("advisory");
		expect(result).toContain("| bm25 | 1.000 | 50% | matched feedback |");
	});

	it("caps rendered hints", () => {
		const hints = Array.from({ length: 12 }, (_, i) => hint({ method: `m${i}`, score: i }));
		const result = renderMemoryContext(hints, { maxHints: 5 });
		const rowCount = result.split("\n").filter((line) => line.startsWith("| m")).length;
		expect(rowCount).toBe(5);
	});
});
