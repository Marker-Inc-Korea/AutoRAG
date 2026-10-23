import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { deriveResultsFromAnswer } from "../../src/agent/answer-citations.ts";
import { CitationNumbers } from "../../src/agent/citation-numbers.ts";
import { answerMarkers } from "../../src/agent/citations.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";

const allowLocal = () => ({ allowLocalFiles: true, numbers: new CitationNumbers() });
const denyLocal = () => ({ allowLocalFiles: false, numbers: new CitationNumbers() });

const cite = (id: string): string => `[${id}]`;

let root: string;
beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-citations-"));
});
afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

describe("answerMarkers", () => {
	it("scans a comma-separated evidence marker into its ids", () => {
		expect(answerMarkers("answer [e3, e7] done")).toEqual([
			{ kind: "evidence", start: 6, end: 15, ids: ["e3", "e7"] },
		]);
	});

	it("scans adjacent bracketed markers as separate evidence markers", () => {
		expect(answerMarkers("x [e3][e7]")).toEqual([
			{ kind: "evidence", start: 1, end: 6, ids: ["e3"] },
			{ kind: "evidence", start: 6, end: 10, ids: ["e7"] },
		]);
	});

	it("scans a model-typed numeric marker as a number", () => {
		expect(answerMarkers("see [2]")).toEqual([{ kind: "number", start: 3, end: 7, number: 2 }]);
	});

	it("accepts [file:<path>] only when isFile says the path is real", () => {
		const file = "/tmp/x/y.txt";
		expect(answerMarkers(`[file:${file}]`, (path) => path === file)).toEqual([
			{ kind: "file", start: 0, end: file.length + 7, path: file },
		]);
		expect(answerMarkers(`[file:${file}]`, () => false)).toEqual([]);
	});

	it("reads a file path that itself contains a closing bracket", () => {
		const file = "/tmp/x/[final] a.txt";
		const markers = answerMarkers(`see [file:${file}] here`, (path) => path === file);
		expect(markers).toHaveLength(1);
		expect(markers[0]).toMatchObject({ kind: "file", path: file });
	});

	it("does not treat a markdown link or an image embed as a citation", () => {
		expect(answerMarkers("[e1](http://x)")).toEqual([]);
		expect(answerMarkers("![x](</p/[e1].png>)")).toEqual([]);
	});

	it("keeps leading spaces inside the marker so answer spacing is preserved", () => {
		const [marker] = answerMarkers("  [e1]");
		expect(marker?.start).toBe(0);
	});

	it("leaves a bracketed year inside a law-report reference as prose", () => {
		expect(answerMarkers("XRM v XRN [2025] SGHCF 55 applies")).toEqual([]);
		expect(answerMarkers("see [1999] 2 SLR 392")).toEqual([]);
	});

	it("leaves legal paragraph headings and pinpoint references as prose (#1832)", () => {
		expect(answerMarkers("**[18]** In assessing division of matrimonial assets")).toEqual([]);
		expect(answerMarkers('> **[18]** "In assessing division of matrimonial assets"')).toEqual([]);
		expect(answerMarkers("**[18]**: The court applied the structured approach.")).toEqual([]);
		expect(answerMarkers("At [18], Choo Han Teck J held that")).toEqual([]);
		expect(answerMarkers("At [18] and [22], the court held that")).toEqual([]);
		expect(answerMarkers("see [2024] SGHCF 24 at [18]–[22].")).toEqual([]);
		expect(answerMarkers("see [2024] SGHCF 24 at [18] – [22].")).toEqual([]);
		expect(answerMarkers("see [2024] SGHCF 24 at [18], [22], and [25].")).toEqual([]);
		expect(answerMarkers("see WXY v ZAB at [18].")).toEqual([]);
		expect(answerMarkers("under para. [18] and paras [22], [24].")).toEqual([]);
	});

	it("still reads a numeric marker that ends a clause (including after ordinary 'at', colon, or bold) as a marker", () => {
		expect(answerMarkers("held so [3]. Next [4]\n[5], [6]")).toHaveLength(4);
		expect(answerMarkers("The cap is the rate they agreed at [99].")).toEqual([
			{ kind: "number", start: 34, end: 39, number: 99 },
		]);
		expect(answerMarkers("The policy states as follows [99]: all requests need approval.")).toEqual([
			{ kind: "number", start: 28, end: 33, number: 99 },
		]);
		expect(answerMarkers("**The cap is 28% [99]** Next sentence.")).toEqual([
			{ kind: "number", start: 16, end: 21, number: 99 },
		]);
	});
});

describe("deriveResultsFromAnswer", () => {
	it("rewrites evidence ids to [n] in first-appearance order across distinct sources", () => {
		const ledger = new EvidenceLedger();
		const a = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "alpha" });
		const b = ledger.register({ method: "web_search", source: "https://b.example/x", content: "beta" });

		const { details, diagnostics } = deriveResultsFromAnswer(
			`First ${cite(b)} second ${cite(a)}.`,
			ledger,
			denyLocal(),
		);

		expect(details.answer).toBe("First [1] second [2].");
		expect(details.mapping.map((entry) => entry.source)).toEqual(["https://b.example/x", "/docs/a.txt"]);
		expect(details.results.map((result) => result.title)).toEqual(["x", "a.txt"]);
		expect(diagnostics).toEqual([]);
	});

	it("collapses two evidence ids on the same source into one result and one number", () => {
		const ledger = new EvidenceLedger();
		const first = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "one" });
		const second = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "two" });

		const { details } = deriveResultsFromAnswer(`${cite(first)}${cite(second)}`, ledger, denyLocal());

		// The same source cited twice back to back reads as one citation, not "[1][1]".
		expect(details.answer).toBe("[1]");
		expect(details.results).toHaveLength(1);
		expect(details.results[0]?.evidence.map((evidence) => evidence.excerpt)).toEqual(["one", "two"]);
		expect(details.mapping).toHaveLength(1);
		expect(details.mapping[0]?.source).toBe("/docs/a.txt");
	});

	it("repeats a number when the same source backs a later, separate sentence", () => {
		const ledger = new EvidenceLedger();
		const first = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "one" });
		const second = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "two" });

		const { details } = deriveResultsFromAnswer(
			`The cap is 28% ${cite(first)}. It renews yearly ${cite(second)}.`,
			ledger,
			denyLocal(),
		);

		expect(details.answer).toBe("The cap is 28% [1]. It renews yearly [1].");
		expect(details.results).toHaveLength(1);
	});

	it("handles both the [e3, e7] and [e3][e7] citation forms", () => {
		const ledger = new EvidenceLedger();
		const a = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "alpha" });
		const b = ledger.register({ method: "bm25", source: "/docs/b.txt", content: "beta" });

		const comma = deriveResultsFromAnswer(`x [${a}, ${b}] y`, ledger, denyLocal());
		expect(comma.details.answer).toBe("x [1][2] y");
		expect(comma.details.mapping.map((entry) => entry.source)).toEqual(["/docs/a.txt", "/docs/b.txt"]);

		const adjacent = deriveResultsFromAnswer(`${cite(a)}${cite(b)}`, ledger, denyLocal());
		expect(adjacent.details.answer).toBe("[1][2]");
		expect(adjacent.details.results.map((result) => result.number)).toEqual([1, 2]);
	});

	it("drops an id that matches no evidence and reports a single diagnostic naming it", () => {
		const ledger = new EvidenceLedger();
		const { details, diagnostics } = deriveResultsFromAnswer(
			"Refund needs approval [e99] and finance too [e98].",
			ledger,
			denyLocal(),
		);

		expect(details.answer).toBe("Refund needs approval and finance too.");
		expect(details.results).toEqual([]);
		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]).toMatchObject({ code: "citation-without-result", severity: "warning" });
		expect(diagnostics[0]?.message).toContain("e99");
		expect(diagnostics[0]?.message).toContain("e98");
	});

	it("drops a numeric marker the model typed itself", () => {
		const ledger = new EvidenceLedger();
		const { details, diagnostics } = deriveResultsFromAnswer("see [2]", ledger, denyLocal());

		expect(details.answer).toBe("see");
		expect(details.results).toEqual([]);
		expect(diagnostics).toHaveLength(1);
		expect(diagnostics[0]?.message).toContain("[2]");
	});

	it("accepts [file:<abs>] for a real file only when local files are allowed", () => {
		const file = join(root, "note [final].txt");
		writeFileSync(file, "hello from disk");

		const allowed = deriveResultsFromAnswer(`Found [file:${file}].`, new EvidenceLedger(), allowLocal());
		expect(allowed.details.results).toHaveLength(1);
		expect(allowed.details.results[0]?.title).toBe("note [final].txt");
		expect(allowed.details.mapping[0]).toMatchObject({ source: file, method: "bash", content: "hello from disk" });

		const denied = deriveResultsFromAnswer(`Found [file:${file}].`, new EvidenceLedger(), denyLocal());
		expect(denied.details.results).toEqual([]);
		expect(denied.details.answer).toContain(`[file:${file}]`);
	});

	it("carries the ledger's source, method, content, and evidenceRefs, never model text", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", {
			id: "bm25:one",
			source: "/data/one.txt",
			content: "snippet text",
			score: 1,
			metadata: { method: "bm25", chunkIndex: 3 },
		});

		const { details } = deriveResultsFromAnswer(`The answer is ${cite(id)}.`, ledger, denyLocal());

		expect(details.mapping[0]).toMatchObject({
			number: 1,
			source: "/data/one.txt",
			method: "bm25",
			content: "snippet text",
		});
		expect(details.mapping[0]?.evidenceRefs[0]).toMatchObject({
			method: "bm25",
			source: "/data/one.txt",
			content: "snippet text",
			chunkIndex: 3,
		});
	});

	it("summarises with the sentence that precedes the marker, stripped of markers and bullets", () => {
		const ledger = new EvidenceLedger();
		const a = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "alpha" });
		const b = ledger.register({ method: "bm25", source: "/docs/b.txt", content: "beta" });

		const { details } = deriveResultsFromAnswer(`- Alpha ${cite(a)} and more ${cite(b)}`, ledger, denyLocal());

		expect(details.results[0]?.summary).toBe("Alpha");
		expect(details.results[1]?.summary).toBe("Alpha and more");
	});

	it("uses only the sentence immediately before the marker as the summary", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "x" });

		const { details } = deriveResultsFromAnswer(
			`Refund policy needs approval. Director sign-off ${cite(id)}`,
			ledger,
			denyLocal(),
		);

		expect(details.results[0]?.summary).toBe("Director sign-off");
	});

	it("never attaches a confidence to a derived result", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "x" });

		const { details } = deriveResultsFromAnswer(cite(id), ledger, denyLocal());

		expect(details.results[0]).not.toHaveProperty("confidence");
		expect(details.results[0]).not.toHaveProperty("source");
	});

	it("scans pathological input in linear time", () => {
		const input = `${"(<".repeat(20_000)}${"[e".repeat(20_000)}`;
		const started = performance.now();
		const markers = answerMarkers(input);
		const { details } = deriveResultsFromAnswer(input, new EvidenceLedger(), denyLocal());
		const elapsed = performance.now() - started;

		expect(markers).toEqual([]);
		expect(details.results).toEqual([]);
		expect(elapsed).toBeLessThan(1_000);
	});

	describe("conversation-level numbering", () => {
		function threeSources() {
			const ledger = new EvidenceLedger();
			const a = ledger.register({ method: "bm25", source: "/docs/a.txt", content: "alpha" });
			const b = ledger.register({ method: "bm25", source: "/docs/b.txt", content: "beta" });
			const c = ledger.register({ method: "bm25", source: "/docs/c.txt", content: "gamma" });
			return { ledger, a, b, c };
		}

		it("keeps a source's number from an earlier answer and numbers a new source with the next unused one", () => {
			const { ledger, a, b, c } = threeSources();
			const options = { allowLocalFiles: false, numbers: new CitationNumbers() };

			const first = deriveResultsFromAnswer(`Alpha ${cite(a)}. Beta ${cite(b)}.`, ledger, options);
			const second = deriveResultsFromAnswer(`Gamma ${cite(c)}. Alpha again ${cite(a)}.`, ledger, options);

			expect(first.details.answer).toBe("Alpha [1]. Beta [2].");
			expect(second.details.answer).toBe("Gamma [3]. Alpha again [1].");
			expect(second.details.results.map((result) => [result.number, result.title])).toEqual([
				[3, "c.txt"],
				[1, "a.txt"],
			]);
			expect(second.details.mapping.map((entry) => [entry.number, entry.source])).toEqual([
				[3, "/docs/c.txt"],
				[1, "/docs/a.txt"],
			]);
		});

		it("gives the quick and the verification answer of one turn one shared numbering", () => {
			const { ledger, a, b } = threeSources();
			const options = { allowLocalFiles: false, numbers: new CitationNumbers() };

			const quick = deriveResultsFromAnswer(`A ${cite(a)}. B ${cite(b)}.`, ledger, options);
			const verified = deriveResultsFromAnswer(`B ${cite(b)}. A ${cite(a)}.`, ledger, options);

			expect(quick.details.answer).toBe("A [1]. B [2].");
			expect(verified.details.answer).toBe("B [2]. A [1].");
		});

		it("does not spend a number on a citation that resolves to no evidence", () => {
			const { ledger, a } = threeSources();
			const options = { allowLocalFiles: false, numbers: new CitationNumbers() };

			deriveResultsFromAnswer("Ghost [e99].", ledger, options);
			const { details } = deriveResultsFromAnswer(`Real ${cite(a)}.`, ledger, options);

			expect(details.answer).toBe("Real [1].");
		});

		it("numbers a source cited by file path and by evidence id as one source", () => {
			const file = join(root, "note.txt");
			writeFileSync(file, "hello");
			const ledger = new EvidenceLedger();
			const id = ledger.register({ method: "bm25", source: file, content: "hello" });
			const options = { allowLocalFiles: true, numbers: new CitationNumbers() };

			const first = deriveResultsFromAnswer(`Opened [file:${file}].`, ledger, options);
			const second = deriveResultsFromAnswer(`Again ${cite(id)}.`, ledger, options);

			expect(first.details.answer).toBe("Opened [1].");
			expect(second.details.answer).toBe("Again [1].");
		});
	});
});
