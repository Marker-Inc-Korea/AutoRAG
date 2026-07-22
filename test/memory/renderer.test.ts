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

	it("renders durable insights as advisory context", () => {
		const result = renderMemoryContext([], {
			insights: [
				{
					id: "insight-1",
					clusterKey: "photo archive lookup",
					domain: "photo archive lookup",
					recommendedSources: ["Pictures/"],
					recommendedMethods: ["posix"],
					rationale: "100 useful signals consistently preferred posix",
					supportingSignalCount: 100,
					confidence: 1,
					createdAt: 1,
					updatedAt: 1,
				},
			],
		});
		expect(result).toContain("## Long-Term Retrieval Insights");
		expect(result).toContain("advisory");
		expect(result).toContain("photo archive lookup");
		expect(result).toContain("Pictures/");
		expect(result).toContain("posix");
	});
});
