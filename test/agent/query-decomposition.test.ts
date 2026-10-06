import { describe, expect, it } from "vitest";
import {
	buildDecompositionPrompt,
	decomposeQuery,
	MAX_DECOMPOSED_QUERIES,
	parseDecomposedQueries,
} from "../../src/agent/query-decomposition.ts";

describe("query decomposition", () => {
	it("caps decomposition at five search queries", () => {
		expect(MAX_DECOMPOSED_QUERIES).toBe(5);
		const raw = JSON.stringify({ queries: ["a1", "b2", "c3", "d4", "e5", "f6", "g7"] });
		expect(parseDecomposedQueries(raw)).toEqual(["a1", "b2", "c3", "d4", "e5"]);
	});

	it("parses a fenced JSON object or a bare JSON array", () => {
		expect(parseDecomposedQueries('```json\n{"queries": ["Q3 budget memo", "Q4 budget memo"]}\n```')).toEqual([
			"Q3 budget memo",
			"Q4 budget memo",
		]);
		expect(parseDecomposedQueries('Here you go: ["lease signer", "lease start date"]')).toEqual([
			"lease signer",
			"lease start date",
		]);
	});

	it("falls back to one query per line when the model ignores the JSON format", () => {
		expect(parseDecomposedQueries("1. Q3 budget approver\n- Q4 budget approver\n\n* budget memo dates")).toEqual([
			"Q3 budget approver",
			"Q4 budget approver",
			"budget memo dates",
		]);
	});

	it("drops blank and duplicate queries case-insensitively", () => {
		expect(parseDecomposedQueries(JSON.stringify(["Lease", " lease ", "", "deposit", 3]))).toEqual([
			"Lease",
			"deposit",
		]);
	});

	it("sends the question and the five-query cap to the decomposition model", async () => {
		const prompts: string[] = [];
		const queries = await decomposeQuery(async (prompt) => {
			prompts.push(prompt);
			return '{"queries": ["Q3 budget approver", "Q4 budget approver"]}';
		}, "Who approved the Q3 and Q4 budgets?");
		expect(queries).toEqual(["Q3 budget approver", "Q4 budget approver"]);
		expect(prompts).toEqual([buildDecompositionPrompt("Who approved the Q3 and Q4 budgets?")]);
		expect(prompts[0]).toContain("Who approved the Q3 and Q4 budgets?");
		expect(prompts[0]).toContain("at most 5");
	});

	it("searches the original question when the model returns nothing usable", async () => {
		await expect(decomposeQuery(async () => "   ", "original question")).resolves.toEqual(["original question"]);
	});
});
