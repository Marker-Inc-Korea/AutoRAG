import { describe, expect, it } from "vitest";
import { describeLocation, locateChunk, locationSuffix } from "../../src/retrieval/chunk-location.ts";

const STATUTE = [
	"# 이용약관",
	"",
	"제1조(목적)",
	"이 약관은 서비스 이용에 관한 사항을 정한다.",
	"",
	"제5조(회원의 의무)",
	"① 회원은 관계 법령을 준수하여야 한다.",
	"② 회원은 타인의 계정을 도용하여서는 아니 된다.",
	"③ 회원은 서비스의 운영을 방해하여서는 아니 된다.",
	"",
	"제6조(계약의 해지)",
	"① 회원은 언제든지 해지를 요청할 수 있다.",
	"",
].join("\n");

describe("locateChunk", () => {
	it("returns the 1-based line range and the enclosing article of a chunk", () => {
		const chunk = "① 회원은 관계 법령을 준수하여야 한다.\n② 회원은 타인의 계정을 도용하여서는 아니 된다.";
		expect(locateChunk(STATUTE, chunk)).toEqual({ lineStart: 7, lineEnd: 8, section: "제5조(회원의 의무)" });
	});

	it("names the article when the chunk itself starts at the article line", () => {
		const chunk = "제5조(회원의 의무)\n① 회원은 관계 법령을 준수하여야 한다.";
		expect(locateChunk(STATUTE, chunk)).toEqual({ lineStart: 6, lineEnd: 7, section: "제5조(회원의 의무)" });
	});

	it("falls back to the nearest markdown heading when there is no article line", () => {
		const doc = "# Deploy\n\nintro\n\n## Rollback\nrun the script\nthen verify\n";
		expect(locateChunk(doc, "run the script\nthen verify")).toEqual({
			lineStart: 6,
			lineEnd: 7,
			section: "Rollback",
		});
	});

	it("ignores blank lines and indentation differences between the chunk and the document", () => {
		const chunk =
			"  ② 회원은 타인의 계정을 도용하여서는 아니 된다.  \n\n③ 회원은 서비스의 운영을 방해하여서는 아니 된다.";
		expect(locateChunk(STATUTE, chunk)).toMatchObject({ lineStart: 8, lineEnd: 9 });
	});

	it("omits the section when nothing precedes the chunk", () => {
		expect(locateChunk("alpha\nbeta\n", "beta")).toEqual({ lineStart: 2, lineEnd: 2 });
	});

	it("returns nothing when the chunk is not in the document", () => {
		expect(locateChunk(STATUTE, "제9조(없는 조항)")).toBeUndefined();
	});

	it("returns nothing when the chunk text occurs twice, so a wrong line is never reported", () => {
		expect(locateChunk("same line\nother\nsame line\n", "same line")).toBeUndefined();
	});

	it("returns nothing for an empty chunk", () => {
		expect(locateChunk(STATUTE, " \n\n")).toBeUndefined();
	});
});

describe("describeLocation", () => {
	it("renders a range, a single line, and the section the same way every time", () => {
		expect(describeLocation({ lineStart: 7, lineEnd: 8, section: "제5조(회원의 의무)" })).toBe(
			'lines 7-8, section "제5조(회원의 의무)"',
		);
		expect(describeLocation({ lineStart: 7, lineEnd: 7 })).toBe("line 7");
		expect(describeLocation({ lineStart: 7 })).toBe("line 7");
		expect(describeLocation({ section: "Rollback" })).toBe('section "Rollback"');
	});

	it("renders nothing when no location is known", () => {
		expect(describeLocation({})).toBeUndefined();
	});
});

describe("locationSuffix", () => {
	it("annotates a result the backend measured and leaves others untouched", () => {
		expect(locationSuffix({ lineNumber: 7, lineEnd: 8, section: "제5조(회원의 의무)" })).toBe(
			' (lines 7-8, section "제5조(회원의 의무)")',
		);
		expect(locationSuffix({ method: "bm25" })).toBe("");
		expect(locationSuffix({ lineNumber: "7", section: "" })).toBe("");
	});
});
