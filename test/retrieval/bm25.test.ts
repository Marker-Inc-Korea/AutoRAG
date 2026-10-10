import { describe, expect, it } from "vitest";
import { bm25Scores, tokenize } from "../../src/retrieval/bm25.ts";

describe("tokenize", () => {
	it("lowercases and splits on non letter/digit/Hangul", () => {
		expect(tokenize("Deploy, the Payments-Service!")).toEqual(["deploy", "the", "payments", "service"]);
		expect(tokenize("인증서를 발급해 주세요")).toEqual(["인증서를", "발급해", "주세요"]);
	});
});

describe("bm25Scores", () => {
	it("scores exact matches above prefix-only matches", () => {
		const [exact = 0, prefix = 0] = bm25Scores("deploy", ["deploy now", "deployment now"]);
		expect(exact).toBeGreaterThan(0);
		expect(prefix).toBeGreaterThan(0);
		expect(exact).toBeGreaterThan(prefix);
	});

	it("matches agglutinated Korean tokens by prefix", () => {
		const [korean = 0, other = 0] = bm25Scores("인증서", ["인증서를 발급", "점심 메뉴"]);
		expect(korean).toBeGreaterThan(0);
		expect(other).toBe(0);
	});

	it("scores unmatched documents zero", () => {
		expect(bm25Scores("zzz", ["hello world"])).toEqual([0]);
	});

	it("returns zeros for empty query terms and empty document lists", () => {
		expect(bm25Scores("", ["alpha beta"])).toEqual([0]);
		expect(bm25Scores("!!!", ["alpha beta"])).toEqual([0]);
		expect(bm25Scores("alpha", [])).toEqual([]);
	});

	it("weights a rarer term above a common one", () => {
		const docs = ["x y", "y", "y", "y"];
		const [rare = 0] = bm25Scores("x", docs);
		const [common = 0] = bm25Scores("y", docs);
		expect(rare).toBeGreaterThan(common);
	});

	it("normalizes by document length", () => {
		const [short = 0, long = 0] = bm25Scores("alpha", ["alpha", "alpha beta gamma delta"]);
		expect(short).toBeGreaterThan(long);
	});
});
