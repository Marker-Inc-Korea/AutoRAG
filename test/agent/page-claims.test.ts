import { describe, expect, it } from "vitest";
import { findUnverifiedPageClaims } from "../../src/agent/location-claims.ts";

const cited = [{ pageStart: 2, pageEnd: 3 }];

describe("findUnverifiedPageClaims", () => {
	it("accepts page numbers a cited chunk covers, in English and Korean forms", () => {
		expect(findUnverifiedPageClaims("It is on page 2 and pages 2-3.", cited)).toEqual([]);
		expect(findUnverifiedPageClaims("p. 3 shows it", cited)).toEqual([]);
		expect(findUnverifiedPageClaims("2페이지와 3쪽에 있습니다.", cited)).toEqual([]);
		expect(findUnverifiedPageClaims("2~3페이지", cited)).toEqual([]);
	});

	it("reports a page no cited chunk covers, once, in order", () => {
		expect(findUnverifiedPageClaims("page 9, 7쪽, page 9, pages 2-12", cited)).toEqual([9, 7, 12]);
	});

	it("does not mistake other numbers or words for page references", () => {
		expect(findUnverifiedPageClaims("제5장 3개 항, 총 12건, homepage 40, 2026년", cited)).toEqual([]);
	});

	it("checks nothing when no cited chunk carries a page", () => {
		expect(findUnverifiedPageClaims("page 9", [])).toEqual([]);
		expect(findUnverifiedPageClaims("page 9", [{}])).toEqual([]);
	});
});
