import { describe, expect, it } from "vitest";
import { parseInternalMapping } from "../../src/agent/agent.ts";

describe("parseInternalMapping", () => {
	it("parses valid block with multiple entries", () => {
		const text = `<results>some output</results>

<internal_mapping>
1:src/auth.ts:posix
2:src/config.ts:vector
</internal_mapping>`;
		const results = parseInternalMapping(text);
		expect(results.length).toBe(2);
		expect(results[0]).toEqual({ index: 1, content: "", source: "src/auth.ts", method: "posix" });
		expect(results[1]).toEqual({ index: 2, content: "", source: "src/config.ts", method: "vector" });
	});

	it("returns empty array when no internal_mapping found", () => {
		expect(parseInternalMapping("just some text")).toEqual([]);
		expect(parseInternalMapping("<results>output</results>")).toEqual([]);
	});

	it("skips malformed lines", () => {
		const text = `<internal_mapping>
1:src/auth.ts:posix
bad line no colons
:missing-index:posix
3:src/config.ts:vector
</internal_mapping>`;
		const results = parseInternalMapping(text);
		expect(results.length).toBe(2);
		expect(results[0].index).toBe(1);
		expect(results[1].index).toBe(3);
	});

	it("handles extra whitespace and blank lines", () => {
		const text = `<internal_mapping>
  1:src/auth.ts:posix  

  2:src/config.ts:vector
</internal_mapping>`;
		const results = parseInternalMapping(text);
		expect(results.length).toBe(2);
	});

	it("handles single entry", () => {
		const text = `<internal_mapping>
1:src/main.ts:posix
</internal_mapping>`;
		const results = parseInternalMapping(text);
		expect(results.length).toBe(1);
		expect(results[0].source).toBe("src/main.ts");
	});

	it("handles empty block", () => {
		const text = "<internal_mapping></internal_mapping>";
		const results = parseInternalMapping(text);
		expect(results.length).toBe(0);
	});
});
