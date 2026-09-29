import { describe, expect, it } from "vitest";
import { extractPartialAnswer } from "../../src/agent/answer-delta.ts";

function stream(raw: string, chunkSize: number): string[] {
	const chunks: string[] = [];
	for (let index = 0; index < raw.length; index += chunkSize) chunks.push(raw.slice(index, index + chunkSize));
	return chunks;
}

describe("extractPartialAnswer", () => {
	it("returns undefined until the answer value starts streaming", () => {
		expect(extractPartialAnswer("")).toBeUndefined();
		expect(extractPartialAnswer("{")).toBeUndefined();
		expect(extractPartialAnswer('{"answ')).toBeUndefined();
		expect(extractPartialAnswer('{"answer"')).toBeUndefined();
		expect(extractPartialAnswer('{"answer":')).toBeUndefined();
		expect(extractPartialAnswer('{"answer": ')).toBeUndefined();
		expect(extractPartialAnswer('{"other": "x", ')).toBeUndefined();
	});

	it("grows the decoded prefix as raw fragments arrive", () => {
		const raw = '{"answer": "Refund exceptions require director approval [1]."}';
		const prefixes: (string | undefined)[] = [];
		let consumed = "";
		for (const chunk of stream(raw, 7)) {
			consumed += chunk;
			prefixes.push(extractPartialAnswer(consumed));
		}
		expect(prefixes[0]).toBeUndefined();
		for (let index = 1; index < prefixes.length; index += 1) {
			const previous = prefixes[index - 1] ?? "";
			const current = prefixes[index];
			expect(current).toBeDefined();
			expect((current ?? "").startsWith(previous)).toBe(true);
		}
		expect(prefixes.at(-1)).toBe("Refund exceptions require director approval [1].");
	});

	it("decodes escape sequences only once they are complete", () => {
		expect(extractPartialAnswer('{"answer": "line1\\nline2')).toBe("line1\nline2");
		expect(extractPartialAnswer('{"answer": "line1\\')).toBe("line1");
		expect(extractPartialAnswer('{"answer": "quote: \\"')).toBe('quote: "');
		expect(extractPartialAnswer('{"answer": "tab\\t')).toBe("tab\t");
		expect(extractPartialAnswer('{"answer": "unicode \\uac00')).toBe("unicode 가");
		expect(extractPartialAnswer('{"answer": "unicode \\uac')).toBe("unicode ");
	});

	it("stops at the closing quote and ignores later keys", () => {
		expect(extractPartialAnswer('{"answer": "done", "results": [{"summary": "x"}]}')).toBe("done");
	});

	it("waits for an earlier sibling value to finish streaming", () => {
		const partial = '{"results": [{"summary": "still stream';
		expect(extractPartialAnswer(partial)).toBeUndefined();
		const completeSibling = '{"results": [{"summary": "ok"}], "answer": "final';
		expect(extractPartialAnswer(completeSibling)).toBe("final");
	});

	it("holds back a dangling high surrogate split across deltas", () => {
		const highOnly = `{"answer": "가${String.fromCharCode(0xd83d)}`;
		expect(extractPartialAnswer(highOnly)).toBe("가");
		const pair = `{"answer": "가${String.fromCharCode(0xd83d, 0xde00)}`;
		expect(extractPartialAnswer(pair)).toBe(`가${String.fromCharCode(0xd83d, 0xde00)}`);
	});

	it("ignores non-string or nested answer values", () => {
		expect(extractPartialAnswer('{"answer": 42}')).toBeUndefined();
		expect(extractPartialAnswer('{"results": [{"answer": "nested"}], "answer": "top"')).toBe("top");
	});

	it("returns undefined for a closed object without an answer key", () => {
		expect(extractPartialAnswer('{"results": []}')).toBeUndefined();
	});
});
