import { describe, expect, it } from "vitest";
import type { SearchDocumentsResponse } from "../../src/agent/search-documents.ts";
import { renderSearch } from "../../src/cli/output.ts";

const response: SearchDocumentsResponse = {
	sessionId: "s1",
	query: "q",
	answer: "제5조는 3개 항입니다 [1].",
	searched: 1,
	warnings: [],
	results: [
		{
			number: 1,
			title: "terms.md",
			summary: "제5조는 3개 항입니다",
			source: "/docs/terms.md",
			evidence: [
				{ excerpt: "① …", lineNumber: 7, lineEnd: 9, section: "제5조(회원의 의무)" },
				{ excerpt: "no location measured" },
			],
		},
	],
};

describe("renderSearch evidence location (#1692)", () => {
	it("puts the measured location on every result in JSON, without --debug", () => {
		const parsed = JSON.parse(renderSearch(response, { json: true }));
		expect(parsed.results[0].locations).toEqual([{ lineStart: 7, lineEnd: 9, section: "제5조(회원의 의무)" }]);
		expect(parsed.results[0]).not.toHaveProperty("evidence");
	});

	it("keeps lineEnd and section on debug evidence entries", () => {
		const parsed = JSON.parse(renderSearch(response, { json: true, debug: true }));
		expect(parsed.results[0].evidence[0]).toEqual({
			excerpt: "① …",
			lineNumber: 7,
			lineEnd: 9,
			section: "제5조(회원의 의무)",
		});
	});

	it("prints the location under the result in human output, once per distinct location", () => {
		const text = renderSearch(response, { json: false });
		expect(text).toContain('   location: lines 7-9, section "제5조(회원의 의무)"');
		expect(text.match(/location:/gu)).toHaveLength(1);
	});

	it("prints no location line for a result nothing measured", () => {
		const bare: SearchDocumentsResponse = {
			...response,
			results: [{ ...response.results[0]!, evidence: [{ excerpt: "x" }] }],
		};
		expect(renderSearch(bare, { json: false })).not.toContain("location:");
		expect(JSON.parse(renderSearch(bare, { json: true })).results[0]).not.toHaveProperty("locations");
	});
});
