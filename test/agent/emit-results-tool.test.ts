import { describe, expect, it } from "vitest";
import {
	type AutoRAGResultsDetails,
	createEmitResultsTool,
	EMIT_AUTORAG_RESULTS_TOOL_NAME,
} from "../../src/agent/emit-results-tool.ts";

describe("createEmitResultsTool", () => {
	it("returns typed details with terminate and forwards them to the capture sink", async () => {
		let captured: AutoRAGResultsDetails | undefined;
		const tool = createEmitResultsTool((details) => {
			captured = details;
		});

		expect(tool.name).toBe(EMIT_AUTORAG_RESULTS_TOOL_NAME);

		const result = await tool.execute("call-1", {
			answer: "the answer",
			results: [
				{
					number: 1,
					title: "Result one",
					summary: "summary one",
					evidence: [{ excerpt: "snippet", lineNumber: 12 }],
					confidence: 0.9,
				},
			],
			mapping: [{ number: 1, source: "/data/one.txt", method: "grep", content: "snippet" }],
			warnings: [],
		});

		expect(result.terminate).toBe(true);
		expect(result.details.answer).toBe("the answer");
		expect(result.details.results[0].evidence[0]).toEqual({ excerpt: "snippet", lineNumber: 12 });
		expect(result.details.mapping[0]).toEqual({
			number: 1,
			source: "/data/one.txt",
			method: "grep",
			content: "snippet",
			evidenceRefs: [{ method: "grep", source: "/data/one.txt", content: "snippet" }],
		});
		expect(captured).toBe(result.details);
	});

	it("omits lineNumber from evidence when not provided", async () => {
		const tool = createEmitResultsTool(() => {});
		const result = await tool.execute("call-2", {
			answer: "a",
			results: [{ number: 1, title: "t", summary: "s", evidence: [{ excerpt: "e" }], confidence: 1 }],
			mapping: [{ number: 1, source: "/x", method: "grep", content: "e" }],
		});
		expect(result.details.results[0].evidence[0]).toEqual({ excerpt: "e" });
		expect(result.details.warnings).toEqual([]);
	});
});
