import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { deriveResultsFromAnswer } from "../../src/agent/answer-citations.ts";
import { CitationNumbers } from "../../src/agent/citation-numbers.ts";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { recordStructuredResultsSession } from "../../src/agent/search-documents.ts";
import { evidenceFor } from "../../src/cli/commands/evidence.ts";
import { renderSearch } from "../../src/cli/output.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let dir: string;
beforeEach(() => {
	dir = mkdtempSync(join(tmpdir(), "autorag-page-evidence-"));
});
afterEach(() => rmSync(dir, { recursive: true, force: true }));

const content = "② 회원은 타인의 계정을 도용하여서는 아니 된다.";
const hit: RetrievalResult = {
	id: "minsync:/docs/terms.pdf:c1",
	source: "/docs/terms.pdf",
	content,
	score: 1,
	metadata: { method: "minsync", lineNumber: 7, lineEnd: 8, pageStart: 2, pageEnd: 3, section: "제5조(회원의 의무)" },
};
const cite = (id: string) => `[${id}]`;
const denyLocal = () => ({ allowLocalFiles: false, numbers: new CitationNumbers() });

describe("page evidence (#2111)", () => {
	it("records the page on the ledger entry and copies it into results[].evidence and the mapping ref", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", hit);
		expect(ledger.lookup(id, { allowLocalFiles: true })[0]).toMatchObject({ pageStart: 2, pageEnd: 3 });

		const { details } = deriveResultsFromAnswer(`제5조 ${cite(id)}.`, ledger, denyLocal());
		expect(details.results[0]?.evidence[0]).toMatchObject({ lineNumber: 7, lineEnd: 8, pageStart: 2, pageEnd: 3 });
		expect(details.mapping[0]?.evidenceRefs[0]).toMatchObject({ pageStart: 2, pageEnd: 3 });
	});

	it("omits the page keys, rather than null or 0, when none was measured", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("t", { ...hit, metadata: { lineNumber: 7, lineEnd: 8 } });
		const { details } = deriveResultsFromAnswer(`x ${cite(id)}.`, ledger, denyLocal());
		expect(details.results[0]?.evidence[0]).not.toHaveProperty("pageStart");
		expect(details.results[0]?.evidence[0]).not.toHaveProperty("pageEnd");
	});

	it("survives the session record, the persisted memory, and `autorag evidence`", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", hit);
		const { details } = deriveResultsFromAnswer(`제5조 ${cite(id)}.`, ledger, denyLocal());
		const memory = new RetrievalMemory({ storagePath: join(dir, "memory.json") });
		memory.load();

		const response = recordStructuredResultsSession("s1", "q", details, memory);
		expect(response.results[0]?.evidence[0]).toMatchObject({ pageStart: 2, pageEnd: 3 });

		const reloaded = new RetrievalMemory({ storagePath: join(dir, "memory.json") });
		reloaded.load();
		const view = evidenceFor(reloaded.getSchema(), "s1");
		expect(view.results[0]?.chunks[0]).toMatchObject({ lineNumber: 7, lineEnd: 8, pageStart: 2, pageEnd: 3 });
	});

	it("rejects an invalid page rather than persisting it", () => {
		for (const bad of [0, -1, 1.5, Number.NaN]) {
			const ref = normalizeSessionEvidenceRef({
				method: "m",
				source: "opaque:s",
				content: "c",
				pageStart: bad,
				pageEnd: bad,
			});
			expect(ref).not.toHaveProperty("pageStart");
			expect(ref).not.toHaveProperty("pageEnd");
		}
	});

	it("drops a page range that runs backwards", () => {
		const ref = normalizeSessionEvidenceRef({
			method: "m",
			source: "opaque:s",
			content: "c",
			pageStart: 5,
			pageEnd: 2,
		});
		expect(ref).not.toHaveProperty("pageStart");
		expect(ref).not.toHaveProperty("pageEnd");
	});

	it("shows the page in search output: locations in JSON, a location line in text, and the debug evidence", () => {
		const response = {
			sessionId: "s1",
			query: "q",
			answer: "a",
			searched: 1,
			warnings: [],
			results: [
				{
					number: 1,
					title: "terms.pdf",
					summary: "s",
					source: "/docs/terms.pdf",
					evidence: [{ excerpt: "x", lineNumber: 7, lineEnd: 8, pageStart: 2, pageEnd: 3 }],
				},
			],
		};
		const json = JSON.parse(renderSearch(response, { json: true, debug: true }));
		expect(json.results[0].locations).toEqual([{ lineStart: 7, lineEnd: 8, pageStart: 2, pageEnd: 3 }]);
		expect(json.results[0].evidence[0]).toMatchObject({ pageStart: 2, pageEnd: 3 });
		const text = renderSearch(response, { json: false, debug: true });
		expect(text).toContain("location: lines 7-8, pages 2-3");
		expect(text).toContain("evidence (lines 7-8, pages 2-3): x");
	});

	it("shows a page even when no line was measured", () => {
		const response = {
			sessionId: "s1",
			query: "q",
			answer: "a",
			searched: 1,
			warnings: [],
			results: [
				{
					number: 1,
					title: "t",
					summary: "s",
					evidence: [{ excerpt: "x", pageStart: 4, pageEnd: 4 }],
				},
			],
		};
		expect(renderSearch(response, { json: false })).toContain("location: page 4");
	});
});

describe("page claims in the answer (#2111)", () => {
	it("flags a page in the prose that no cited evidence covers, and not one it covers", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("t", hit);
		const wrong = deriveResultsFromAnswer(`It is on page 9 ${cite(id)}.`, ledger, denyLocal());
		const right = deriveResultsFromAnswer(`It is on page 3 ${cite(id)}.`, ledger, denyLocal());
		expect(wrong.diagnostics.map((d) => d.code)).toContain("page-reference-unverified");
		expect(right.diagnostics.map((d) => d.code)).not.toContain("page-reference-unverified");
	});
});
