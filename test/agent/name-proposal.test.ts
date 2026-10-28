import { describe, expect, it } from "vitest";
import { MAX_NAME_CANDIDATES } from "../../src/agent/name-candidates.ts";
import { buildNameProposalPrompt, proposeFileNames } from "../../src/agent/name-proposal.ts";

describe("file-name proposal", () => {
	it("sends the question and asks for plain name fragments in several spellings and languages", async () => {
		const prompts: string[] = [];
		const names = await proposeFileNames(async (prompt) => {
			prompts.push(prompt);
			return '{"names": ["invoice", "인보이스", "bill"]}';
		}, "3월 인보이스 pdf 찾아줘");
		expect(names).toEqual(["invoice", "인보이스", "bill"]);
		expect(prompts).toEqual([buildNameProposalPrompt("3월 인보이스 pdf 찾아줘")]);
		expect(prompts[0]).toContain("3월 인보이스 pdf 찾아줘");
		expect(prompts[0]).toMatch(/at most 8/u);
		expect(prompts[0]).toMatch(/no (search )?operators|plain/iu);
	});

	it("parses fenced JSON, a bare array, and one name per line", async () => {
		await expect(proposeFileNames(async () => '```json\n{"names": ["lease", "임대차"]}\n```', "q")).resolves.toEqual([
			"lease",
			"임대차",
		]);
		await expect(proposeFileNames(async () => '["lease", "임대차"]', "q")).resolves.toEqual(["lease", "임대차"]);
		await expect(proposeFileNames(async () => "1. lease\n- 임대차", "q")).resolves.toEqual(["lease", "임대차"]);
	});

	it("drops blanks and repeats and caps at the candidate limit", async () => {
		const many = Array.from({ length: MAX_NAME_CANDIDATES + 4 }, (_, index) => `name-${index}`);
		const names = await proposeFileNames(
			async () => JSON.stringify({ names: ["Lease", " lease ", "", ...many] }),
			"q",
		);
		expect(names).toHaveLength(MAX_NAME_CANDIDATES);
		expect(names.slice(0, 2)).toEqual(["Lease", "name-0"]);
	});

	it("returns nothing when the model gives nothing usable, so the caller can fall back to the question", async () => {
		await expect(proposeFileNames(async () => "   ", "q")).resolves.toEqual([]);
	});
});
