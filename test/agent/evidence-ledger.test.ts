import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { EvidenceLedger } from "../../src/agent/evidence-ledger.ts";
import { normalizeSessionEvidenceRef } from "../../src/memory/memory.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-ledger-"));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function result(source: string, content: string, metadata: Record<string, unknown> = {}): RetrievalResult {
	return { id: `r:${source}`, source, content, score: 1, metadata };
}

const allowLocal = { allowLocalFiles: true };
const denyLocal = { allowLocalFiles: false };

describe("EvidenceLedger", () => {
	it("hands out short sequential ids and reuses the id for the same chunk", () => {
		const ledger = new EvidenceLedger();
		const first = ledger.registerResult("search_all_documents", result("/a.txt", "alpha"));
		const second = ledger.registerResult("search_all_documents", result("/b.txt", "beta"));
		const again = ledger.registerResult("semantic_search_local_docs", result("/a.txt", "alpha"));

		expect(first).toBe("e1");
		expect(second).toBe("e2");
		expect(again).toBe("e1");
	});

	it("resolves ids (bare or bracketed) to harness-held evidence, never model text", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult(
			"search_all_documents",
			result("/docs/a.txt", "the real chunk", { method: "bm25" }),
		);

		const [bare] = ledger.lookup(id, allowLocal);
		const [bracketed] = ledger.lookup(`[${id}]`, allowLocal);

		expect(bare).toMatchObject({ method: "bm25", source: "/docs/a.txt", content: "the real chunk" });
		expect(bracketed).toEqual(bare);
	});

	it("accumulates every method that surfaced a chunk into retrieverMix", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/a.txt", "same", { method: "bm25" }));
		ledger.registerResult("tool", result("/a.txt", "same", { method: "minsync" }));

		const [ref] = ledger.lookup(id, allowLocal);

		expect(ref?.method).toBe("bm25");
		expect(ref?.retrieverMix).toEqual(["bm25", "minsync"]);
	});

	it("falls back to the tool name when a result carries no method", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", result("/a.txt", "x"));
		expect(ledger.lookup(id, allowLocal)[0]?.method).toBe("search_all_documents");
	});

	it("resolves the exact source string of observed evidence (web urls, paths)", () => {
		const ledger = new EvidenceLedger();
		ledger.register({ method: "web_search", source: "https://example.com/x", content: "Title — snippet" });

		const [ref] = ledger.lookup("https://example.com/x", allowLocal);

		expect(ref).toMatchObject({ method: "web_search", source: "https://example.com/x", content: "Title — snippet" });
	});

	it("returns every recorded chunk of a source when the source string is cited", () => {
		const ledger = new EvidenceLedger();
		ledger.register({ method: "tool", source: "/a.txt", content: "one" });
		ledger.register({ method: "tool", source: "/a.txt", content: "two" });

		const refs = ledger.lookup("/a.txt", allowLocal);

		expect(refs.map((ref) => ref.content)).toEqual(["one", "two"]);
	});

	it("returns no match for an id the run never issued, so the caller drops it", () => {
		const ledger = new EvidenceLedger();
		ledger.registerResult("tool", result("/a.txt", "x"));

		expect(ledger.lookup("e99", allowLocal)).toEqual([]);
	});

	it("returns no match for a hallucinated path that is not a real file", () => {
		const ledger = new EvidenceLedger();
		expect(ledger.lookup("/docs/invented.txt", allowLocal)).toEqual([]);
	});

	it("returns no match for a datasource virtual id that was never retrieved", () => {
		const ledger = new EvidenceLedger();
		expect(ledger.lookup("/kakao/acct-1/room", allowLocal)).toEqual([]);
	});

	it("accepts a real local file the model opened itself, taking the content from the file", () => {
		const ledger = new EvidenceLedger();
		const file = join(root, "read-with-bash.txt");
		writeFileSync(file, "file contents");

		const [ref] = ledger.lookup(file, allowLocal);

		expect(ref).toMatchObject({ method: "bash", source: file, content: "file contents" });
	});

	it("falls back to the path for a local file that is not readable text", () => {
		const ledger = new EvidenceLedger();
		const file = join(root, "binary.bin");
		writeFileSync(file, Buffer.from([0x00, 0x01, 0x02, 0xff]));

		const [ref] = ledger.lookup(file, allowLocal);

		expect(ref).toMatchObject({ method: "bash", source: file, content: file });
	});

	it("refuses unobserved local files when the session forbids them", () => {
		const ledger = new EvidenceLedger();
		const file = join(root, "secret.txt");
		writeFileSync(file, "contents");

		expect(ledger.lookup(file, denyLocal)).toEqual([]);
	});

	it("never resolves a directory as evidence", () => {
		const ledger = new EvidenceLedger();
		expect(ledger.lookup(root, allowLocal)).toEqual([]);
	});

	it("forgets everything on clear and never reissues an id, so a stale id cannot alias new evidence", () => {
		const ledger = new EvidenceLedger();
		ledger.registerResult("tool", result("/a.txt", "x"));
		ledger.clear();

		expect(ledger.lookup("e1", allowLocal)).toEqual([]);
		const next = ledger.registerResult("tool", result("/z.txt", "z"));
		expect(next).toBe("e2");
		expect(ledger.lookup("e1", allowLocal)).toEqual([]);
	});

	it("keeps the stable evidence id independent of what the model writes", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/a.txt", "the exact chunk text"));
		const [ref] = ledger.lookup(id, allowLocal);
		const again = new EvidenceLedger();
		const sameId = again.registerResult("tool", result("/a.txt", "the exact chunk text"));
		const [sameRef] = again.lookup(sameId, allowLocal);

		expect(ref).toBeDefined();
		expect(normalizeSessionEvidenceRef(ref as never).stableEvidenceId).toBe(
			normalizeSessionEvidenceRef(sameRef as never).stableEvidenceId,
		);
		expect(ref?.content).toBe("the exact chunk text");
	});

	it("caps stored content so one huge chunk cannot bloat the run", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/big.txt", "x".repeat(50_000)));
		expect((ledger.lookup(id, allowLocal)[0]?.content ?? "").length).toBeLessThanOrEqual(2_000);
	});

	it("keeps same-source chunks with identical stored prefixes distinct", () => {
		const ledger = new EvidenceLedger();
		const shared = "a".repeat(2_000);
		const first = ledger.registerResult("tool", result("/a.txt", `${shared} tail one`, { chunkIndex: 0 }));
		const second = ledger.registerResult("tool", result("/a.txt", `${shared} tail two`, { chunkIndex: 1 }));

		expect(first).not.toBe(second);
		expect(ledger.lookup(first, allowLocal)[0]?.chunkIndex).toBe(0);
		expect(ledger.lookup(second, allowLocal)[0]?.chunkIndex).toBe(1);
	});

	it("carries chunk-level metadata the backend provided", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult(
			"tool",
			result("/a.pdf", "x", {
				chunkIndex: 7,
				lineNumber: 12,
				parserType: "kordoc",
				documentType: "pdf",
				documentArea: "body",
				evidenceType: "quote",
				evidenceLocation: "page 3",
			}),
		);
		expect(ledger.lookup(id, allowLocal)[0]).toMatchObject({
			chunkIndex: 7,
			lineNumber: 12,
			parserType: "kordoc",
			documentType: "pdf",
			documentArea: "body",
			evidenceType: "quote",
			evidenceLocation: "page 3",
		});
	});
});
