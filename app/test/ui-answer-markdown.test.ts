import { describe, expect, it } from "vitest";
import {
	parseAnswerBlocks,
	parseInline,
	splitAnswerParagraphs,
} from "../src/renderer/src/state/answer-markdown";

describe("parseInline", () => {
	it("splits bold spans, citation chips, and plain text in order", () => {
		expect(parseInline("최종 승인액은 **₩3.8억**이며 9월 15일 CFO가 승인했습니다. [2][4]")).toEqual([
			{ kind: "text", text: "최종 승인액은 " },
			{ kind: "bold", text: "₩3.8억" },
			{ kind: "text", text: "이며 9월 15일 CFO가 승인했습니다. " },
			{ kind: "citation", number: 2 },
			{ kind: "citation", number: 4 },
		]);
	});

	it("parses multi-digit citations and adjacent bold segments", () => {
		expect(parseInline("**합계** ₩380,000,000 [12]")).toEqual([
			{ kind: "bold", text: "합계" },
			{ kind: "text", text: " ₩380,000,000 " },
			{ kind: "citation", number: 12 },
		]);
	});

	it("keeps lone brackets that are not numeric citations as text", () => {
		expect(parseInline("0개면 [x] 그대로 [0] 둡니다")).toEqual([
			{ kind: "text", text: "0개면 [x] 그대로 [0] 둡니다" },
		]);
	});

	it("renders inline code and does not parse bold markers inside it", () => {
		expect(parseInline("`bun run **test**` 명령")).toEqual([
			{ kind: "code", text: "bun run **test**" },
			{ kind: "text", text: " 명령" },
		]);
	});

	it("leaves an unterminated bold marker as plain text", () => {
		expect(parseInline("중간에 **닫히지 않은 볼드")).toEqual([
			{ kind: "text", text: "중간에 **닫히지 않은 볼드" },
		]);
	});
});

describe("splitAnswerParagraphs", () => {
	it("splits on blank lines and trims each paragraph", () => {
		expect(splitAnswerParagraphs("첫 단락입니다.\n\n  둘째 단락입니다.  \n\n\n셋째")).toEqual([
			"첫 단락입니다.",
			"둘째 단락입니다.",
			"셋째",
		]);
	});

	it("returns an empty list for empty input", () => {
		expect(splitAnswerParagraphs("")).toEqual([]);
	});
});

describe("parseAnswerBlocks", () => {
	it("parses headings, bullets, ordered items, quotes, and code fences", () => {
		const blocks = parseAnswerBlocks(
			[
				"## 요약",
				"본문 한 줄.",
				"- 첫 항목 [1]",
				"- 둘째 항목",
				"1. 순서 항목",
				"> 인용 문장",
				"```",
				"line1",
				"line2 [3]",
				"```",
				"마무리 **강조**",
			].join("\n"),
		);
		expect(blocks).toEqual([
			{ kind: "heading", text: "요약" },
			{ kind: "paragraph", text: "본문 한 줄." },
			{ kind: "list", text: "첫 항목 [1]" },
			{ kind: "list", text: "둘째 항목" },
			{ kind: "list", text: "순서 항목" },
			{ kind: "quote", text: "인용 문장" },
			{ kind: "code", text: "line1\nline2 [3]" },
			{ kind: "paragraph", text: "마무리 **강조**" },
		]);
	});

	it("skips blank lines and treats an unterminated fence as code to the end", () => {
		const blocks = parseAnswerBlocks("앞\n\n```\not{json}\n");
		expect(blocks).toEqual([
			{ kind: "paragraph", text: "앞" },
			{ kind: "code", text: "ot{json}" },
		]);
	});

	it("keeps a fence language line intact (```json opens the block)", () => {
		const blocks = parseAnswerBlocks("```json\n{\"a\":1}\n```");
		expect(blocks).toEqual([{ kind: "code", text: "{\"a\":1}" }]);
	});
});
