import { describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { renderEvidenceRows } from "../../src/agent/result-render.ts";
import { createSearchAllDocumentsTool, type SearchAllDocumentsProvider } from "../../src/agent/search-all-tool.ts";
import {
	createSingleDatasourceSearchTools,
	type SingleDatasourceSearchProvider,
} from "../../src/agent/search-single-datasource-tool.ts";
import { planReadBudget, RecordCut } from "../../src/retrieval/record-cut.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

function result(id: string, source: string, content: string, score = 0.8): RetrievalResult {
	return { id, source, content, score, metadata: {} };
}

function text(out: { content: readonly { type: string; text?: string }[] }): string {
	return out.content.map((part) => part.text ?? "").join("");
}

describe("planReadBudget", () => {
	it("reads every result in full when the budget covers them", () => {
		expect(planReadBudget([100, 200, 300], 10_000, 50)).toEqual([100, 200, 300]);
	});

	it("never lets the first result spend the whole budget", () => {
		const plan = planReadBudget([1_000_000, 4_000, 4_000], 12_000, 500);
		expect(plan).toEqual([4_000, 4_000, 4_000]);
	});

	it("hands room a short result left over on to the later ones", () => {
		expect(planReadBudget([100, 9_000, 9_000], 5_100, 500)).toEqual([100, 2_500, 2_500]);
	});

	it("marks a result as not read when its share is under the floor, and keeps reading the ones after it", () => {
		// 10 large results, budget for ~2.5: the tail is skipped instead of every result being clipped to nothing.
		const sizes = Array.from({ length: 10 }, () => 10_000);
		const plan = planReadBudget(sizes, 25_000, 8_000);
		// 25_000 / 8_000 = 3 results can each get their floor; the other 7 are announced, not clipped to nothing.
		expect(plan.slice(0, 3)).toEqual([8_333, 8_333, 8_333]);
		expect(plan.slice(3)).toEqual(Array.from({ length: 7 }, () => undefined));
	});

	it("always reads the best result, even when the whole budget is under the floor", () => {
		expect(planReadBudget([20_000, 20_000], 1_000, 8_000)).toEqual([1_000, undefined]);
	});
});

describe("renderEvidenceRows", () => {
	const rows = [
		{ id: "e1", header: "/docs/a.md score=0.9000", content: "alpha   beta", repeat: false },
		{ id: "e2", header: "/docs/b.md score=0.8000", content: "gamma", repeat: true },
	];

	it("renders the legacy shape when no policy is given, repeats included", () => {
		expect(renderEvidenceRows(rows, undefined).text).toBe(
			"[e1] /docs/a.md score=0.9000\nalpha beta\n\n[e2] /docs/b.md score=0.8000\ngamma",
		);
	});

	it("says a repeat was already shown instead of sending its text again", () => {
		const out = renderEvidenceRows(rows, new RecordCut()).text;
		expect(out).toContain("[e2] /docs/b.md score=0.8000\n(already shown earlier in this run; text not repeated)");
		expect(out).not.toContain("gamma");
		expect(out).toContain("alpha beta");
	});

	it("cuts an oversized result to its share, announces the cut, and still shows the later results", () => {
		const huge = "x".repeat(50_000);
		const out = renderEvidenceRows(
			[
				{ id: "e1", header: "/docs/huge.md", content: huge, repeat: false },
				{ id: "e2", header: "/docs/small.md", content: "needle fact", repeat: false },
				{ id: "e3", header: "/docs/small2.md", content: "second fact", repeat: false },
			],
			new RecordCut({ resultChars: 6_000, minReadChars: 100 }),
		).text;
		expect(out).toContain("needle fact");
		expect(out).toContain("second fact");
		expect(out).toMatch(/\[e1\] \/docs\/huge\.md\nx+ …\[\+\d+ chars cut to fit the result budget\]/u);
		expect(out.length).toBeLessThan(7_000);
	});

	it("lists results it had no budget to read, with their ids so they can be cited or fetched", () => {
		const rendered = renderEvidenceRows(
			Array.from({ length: 6 }, (_, index) => ({
				id: `e${index + 1}`,
				header: `/docs/${index}.md`,
				content: "y".repeat(5_000),
				repeat: false,
			})),
			new RecordCut({ resultChars: 15_000, minReadChars: 4_000 }),
		);
		const out = rendered.text;
		// Only the results whose text reached the model count as shown; a later search may still deliver the rest.
		expect([...rendered.delivered].sort()).toEqual(["e1", "e2", "e3"]);
		expect(out).toContain("[e1] /docs/0.md\nyyyy");
		expect(out).toMatch(/\[e6\] \/docs\/5\.md\n\(not read: result budget used up; narrow the query to read it\)/u);
	});
});

describe("EvidenceLedger shown-in-run tracking", () => {
	it("reports a chunk as shown on its second mention and forgets at the start of a run", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("t", result("c1", "/docs/a.md", "same chunk"));
		expect(ledger.hasShown(id)).toBe(false);
		expect(ledger.markShown(id)).toBe(false);
		expect(ledger.hasShown(id)).toBe(true);
		expect(ledger.markShown(id)).toBe(true);
		ledger.beginRun();
		expect(ledger.markShown(id)).toBe(false);
	});

	it("gives the same chunk the same id, so a repeat is recognised across tools", () => {
		const ledger = new EvidenceLedger();
		const first = ledger.registerResult("search_all_documents", result("c1", "/docs/a.md", "same chunk"));
		const second = ledger.registerResult(
			"semantic_search_local_docs",
			result("other-id", "/docs/a.md", "same chunk"),
		);
		expect(second).toBe(first);
	});
});

describe("search tools with a record cut", () => {
	const provider = (batches: RetrievalResult[][]): SearchAllDocumentsProvider => {
		let call = 0;
		return {
			async searchAllDocuments() {
				return { results: batches[call++] ?? [], diagnostics: [] };
			},
		};
	};

	it("sends a passage once per run: a second search that returns it again names it without its text", async () => {
		const a = result("a", "/docs/a.md", "refund exceptions need director approval");
		const b = result("b", "/docs/b.md", "keyword only passage about SKU-4471");
		const ledger = new EvidenceLedger();
		const tool = createSearchAllDocumentsTool(provider([[a], [a, b]]), ledger, { recordCut: new RecordCut() });

		const first = text(await tool.execute("1", { query: "refund" }));
		const second = text(await tool.execute("2", { query: "refund SKU-4471" }));

		expect(first).toContain("refund exceptions need director approval");
		expect(second).not.toContain("refund exceptions need director approval");
		expect(second).toContain("(already shown earlier in this run; text not repeated)");
		expect(second).toContain("keyword only passage about SKU-4471");
	});

	it("sends repeats again after a new run begins, and always without a record cut", async () => {
		const a = result("a", "/docs/a.md", "refund exceptions need director approval");
		const ledger = new EvidenceLedger();
		const withCut = createSearchAllDocumentsTool(provider([[a], [a]]), ledger, { recordCut: new RecordCut() });
		await withCut.execute("1", { query: "q" });
		ledger.beginRun();
		expect(text(await withCut.execute("2", { query: "q" }))).toContain("refund exceptions need director approval");

		const plain = createSearchAllDocumentsTool(provider([[a], [a]]), new EvidenceLedger());
		await plain.execute("3", { query: "q" });
		expect(text(await plain.execute("4", { query: "q" }))).toContain("refund exceptions need director approval");
	});

	it("applies the same repeat rule and result budget to a datasource tool", async () => {
		const mail = result("m1", "/mailcrawl/inbox/1", "z".repeat(40_000));
		const note = result("m2", "/mailcrawl/inbox/2", "the invoice total is 4,200 USD");
		const datasource: SingleDatasourceSearchProvider = {
			async searchSingleDatasourceDocuments() {
				return { results: [mail, note], diagnostics: [] };
			},
		};
		const [tool] = createSingleDatasourceSearchTools(
			datasource,
			[{ datasourceId: "mailcrawl", description: "mail", instanceScopes: [] }],
			new EvidenceLedger(),
			{ recordCut: new RecordCut({ resultChars: 5_000, minReadChars: 100 }) },
		);

		const out = text(await tool!.execute("1", { query: "invoice" }));
		expect(out).toContain("the invoice total is 4,200 USD");
		expect(out).toContain("chars cut to fit the result budget");
	});
});
