import { describe, expect, it } from "vitest";
import { findUnverifiedLineClaims } from "../../src/agent/location-claims.ts";

const evidence = [{ lineStart: 7, lineEnd: 9 }];

describe("findUnverifiedLineClaims", () => {
	it("accepts line references inside a cited evidence range, in Korean and English prose", () => {
		expect(findUnverifiedLineClaims("제5조는 7~9행에 있습니다 [1].", evidence)).toEqual([]);
		expect(findUnverifiedLineClaims("It is on line 8 and lines 7-9.", evidence)).toEqual([]);
		expect(findUnverifiedLineClaims("8번째 줄에서 확인됩니다.", evidence)).toEqual([]);
	});

	it("reports a line number that no cited evidence covers", () => {
		expect(findUnverifiedLineClaims("제5조는 12행에 있습니다.", evidence)).toEqual([12]);
		expect(findUnverifiedLineClaims("see lines 7-20", evidence)).toEqual([20]);
	});

	it("reports each wrong number once, in order of appearance", () => {
		expect(findUnverifiedLineClaims("3행, 40줄, 3행, line 40", evidence)).toEqual([3, 40]);
	});

	it("cannot verify anything when no evidence carries a line range", () => {
		expect(findUnverifiedLineClaims("제5조는 12행에 있습니다.", [])).toEqual([]);
		expect(findUnverifiedLineClaims("제5조는 12행에 있습니다.", [{}])).toEqual([]);
	});

	it("does not mistake other numbers for line references", () => {
		expect(findUnverifiedLineClaims("제5조 3개 항, 총 12건, 2026년 10월 [1]", evidence)).toEqual([]);
		expect(findUnverifiedLineClaims("the baseline 12 was fine; outline 40", evidence)).toEqual([]);
	});

	it("checks every cited range, not only the first", () => {
		expect(findUnverifiedLineClaims("line 30 and line 8", [{ lineStart: 7, lineEnd: 9 }, { lineStart: 30 }])).toEqual(
			[],
		);
	});
});
