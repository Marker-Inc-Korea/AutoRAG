import { describe, expect, it } from "vitest";
import { CitationNumbers, parseCitationNumberEntries } from "../../src/agent/citation-numbers.ts";

describe("CitationNumbers", () => {
	it("numbers sources by first request and hands the same number back for the same source", () => {
		const numbers = new CitationNumbers();

		expect(numbers.numberFor("/a.txt")).toBe(1);
		expect(numbers.numberFor("/b.txt")).toBe(2);
		expect(numbers.numberFor("/a.txt")).toBe(1);
		expect(numbers.numberFor("/c.txt")).toBe(3);
	});

	it("never reissues a number after clear, so an old [n] cannot name a different source", () => {
		const numbers = new CitationNumbers();
		numbers.numberFor("/a.txt");
		numbers.numberFor("/b.txt");
		numbers.clear();

		expect(numbers.numberFor("/z.txt")).toBe(3);
		expect(numbers.numberFor("/a.txt")).toBe(4);
	});

	it("starts over from 1 on reset, for a run that owns no conversation", () => {
		const numbers = new CitationNumbers();
		numbers.numberFor("/a.txt");
		numbers.numberFor("/b.txt");
		numbers.reset();

		expect(numbers.takeUnsaved()).toEqual([]);
		expect(numbers.numberFor("/b.txt")).toBe(1);
	});

	it("restores persisted numbers and keeps counting past them", () => {
		const before = new CitationNumbers();
		before.numberFor("/a.txt");
		before.numberFor("/b.txt");
		const saved = parseCitationNumberEntries(JSON.parse(JSON.stringify({ entries: before.takeUnsaved() })));

		const after = new CitationNumbers();
		after.restore(saved);

		expect(after.numberFor("/b.txt")).toBe(2);
		expect(after.numberFor("/c.txt")).toBe(3);
	});

	it("never reissues a number issued on another branch of the conversation", () => {
		const numbers = new CitationNumbers();
		numbers.restore([{ source: "/a.txt", number: 1 }], 6);

		expect(numbers.numberFor("/a.txt")).toBe(1);
		expect(numbers.numberFor("/b.txt")).toBe(7);
	});

	it("keeps counting past numbers it issued itself when a restore brings lower ones", () => {
		const numbers = new CitationNumbers();
		for (const source of ["/a", "/b", "/c", "/d"]) numbers.numberFor(source);
		numbers.restore([{ source: "/a", number: 1 }]);

		expect(numbers.numberFor("/e")).toBe(5);
	});

	it("hands out only the numbers issued since the last save", () => {
		const numbers = new CitationNumbers();
		numbers.numberFor("/a.txt");
		expect(numbers.takeUnsaved()).toEqual([{ source: "/a.txt", number: 1 }]);
		expect(numbers.takeUnsaved()).toEqual([]);

		numbers.numberFor("/a.txt");
		numbers.numberFor("/b.txt");
		expect(numbers.takeUnsaved()).toEqual([{ source: "/b.txt", number: 2 }]);
	});

	it("does not re-save restored numbers", () => {
		const numbers = new CitationNumbers();
		numbers.restore([{ source: "/a.txt", number: 1 }]);

		expect(numbers.takeUnsaved()).toEqual([]);
	});

	it("skips a malformed persisted entry instead of mis-numbering a source", () => {
		const parsed = parseCitationNumberEntries({
			entries: [
				{ source: "/a.txt", number: 1 },
				{ source: "/b.txt", number: 0 },
				{ source: "/c.txt", number: 1.5 },
				{ source: 7, number: 2 },
				{ number: 3 },
			],
		});

		expect(parsed).toEqual([{ source: "/a.txt", number: 1 }]);
		expect(parseCitationNumberEntries("nope")).toEqual([]);
	});

	it("persists through the registered sink on flush, and keeps numbers unsaved while no sink is set", () => {
		const numbers = new CitationNumbers();
		numbers.numberFor("/a.txt");
		numbers.flush();
		const saved: unknown[] = [];
		numbers.setSink((entries) => saved.push(entries));

		numbers.flush();
		expect(saved).toEqual([[{ source: "/a.txt", number: 1 }]]);

		numbers.numberFor("/b.txt");
		numbers.flush();
		numbers.flush();
		expect(saved).toEqual([[{ source: "/a.txt", number: 1 }], [{ source: "/b.txt", number: 2 }]]);
	});
});
