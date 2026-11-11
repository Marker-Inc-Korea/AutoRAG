import { describe, expect, it } from "vitest";
import { contentSnippet, matchesQueryLexically } from "../../src/find/text.ts";

describe("matchesQueryLexically", () => {
	it("keeps an exact NFKC case-folded substring", () => {
		expect(matchesQueryLexically("Report", "the annual REPORT for 2026")).toBe(true);
	});

	it("keeps a Hangul noun after stripping a trailing postposition", () => {
		expect(matchesQueryLexically("비행기를", "비행기 표를 예약했다")).toBe(true);
	});

	it("does not match a token that is absent", () => {
		expect(matchesQueryLexically("zebra", "start the engine")).toBe(false);
	});

	it("matches an ASCII token beside punctuation", () => {
		expect(matchesQueryLexically("rag", "AutoRAG-notes.md")).toBe(true);
	});

	it("returns false for an empty query or candidate", () => {
		expect(matchesQueryLexically("   ", "anything")).toBe(false);
		expect(matchesQueryLexically("query", "   ")).toBe(false);
	});
});

describe("contentSnippet", () => {
	it("collapses whitespace and returns short text whole", () => {
		expect(contentSnippet("  a\n\nb   c ", "a")).toBe("a b c");
	});

	it("cuts a long passage around the first query term", () => {
		const text = `${"x".repeat(400)}needle${"y".repeat(400)}`;
		const snippet = contentSnippet(text, "needle");
		expect(snippet).toContain("needle");
		expect(snippet.startsWith("…")).toBe(true);
		expect(snippet.endsWith("…")).toBe(true);
		expect(snippet.length).toBeLessThanOrEqual(180);
	});
});
