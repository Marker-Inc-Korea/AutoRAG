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

const resolveOptions = { label: "emit", number: 1, fallbackContent: "model excerpt", allowLocalFiles: true };

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

		const [bare] = ledger.resolve([id], resolveOptions);
		const [bracketed] = ledger.resolve([`[${id}]`], resolveOptions);

		expect(bare).toMatchObject({ method: "bm25", source: "/docs/a.txt", content: "the real chunk" });
		expect(bracketed).toEqual(bare);
	});

	it("accumulates every method that surfaced a chunk into retrieverMix", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/a.txt", "same", { method: "bm25" }));
		ledger.registerResult("tool", result("/a.txt", "same", { method: "minsync" }));

		const [ref] = ledger.resolve([id], resolveOptions);

		expect(ref?.method).toBe("bm25");
		expect(ref?.retrieverMix).toEqual(["bm25", "minsync"]);
	});

	it("falls back to the tool name when a result carries no method", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("search_all_documents", result("/a.txt", "x"));
		expect(ledger.resolve([id], resolveOptions)[0]?.method).toBe("search_all_documents");
	});

	it("resolves the exact source string of observed evidence (web urls, paths)", () => {
		const ledger = new EvidenceLedger();
		ledger.register({ method: "web_search", source: "https://example.com/x", content: "Title — snippet" });

		const [ref] = ledger.resolve(["https://example.com/x"], resolveOptions);

		expect(ref).toMatchObject({ method: "web_search", source: "https://example.com/x", content: "Title — snippet" });
	});

	it("rejects an id the run never issued and tells the model what is valid", () => {
		const ledger = new EvidenceLedger();
		ledger.registerResult("tool", result("/a.txt", "x"));

		expect(() => ledger.resolve(["e99"], resolveOptions)).toThrow(/e99/u);
		expect(() => ledger.resolve(["e99"], resolveOptions)).toThrow(/result 1/u);
	});

	it("rejects a hallucinated path that no tool surfaced and that is not a real file", () => {
		const ledger = new EvidenceLedger();
		expect(() => ledger.resolve(["/docs/invented.txt"], resolveOptions)).toThrow(/invented\.txt/u);
	});

	it("rejects a datasource virtual id that was never retrieved", () => {
		const ledger = new EvidenceLedger();
		expect(() => ledger.resolve(["/kakao/acct-1/room"], resolveOptions)).toThrow(/kakao/u);
	});

	it("accepts a real local file the model opened itself, using the model excerpt as content", () => {
		const ledger = new EvidenceLedger();
		const file = join(root, "read-with-bash.txt");
		writeFileSync(file, "contents");

		const [ref] = ledger.resolve([file], resolveOptions);

		expect(ref).toMatchObject({ method: "bash", source: file, content: "model excerpt" });
	});

	it("refuses unobserved local files when the session forbids them", () => {
		const ledger = new EvidenceLedger();
		const file = join(root, "secret.txt");
		writeFileSync(file, "contents");

		expect(() => ledger.resolve([file], { ...resolveOptions, allowLocalFiles: false })).toThrow(/secret\.txt/u);
	});

	it("never resolves a directory as evidence", () => {
		const ledger = new EvidenceLedger();
		expect(() => ledger.resolve([root], resolveOptions)).toThrow();
	});

	it("de-duplicates refs that resolve to the same evidence", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/a.txt", "x"));
		expect(ledger.resolve([id, `[${id}]`, "/a.txt"], resolveOptions)).toHaveLength(1);
	});

	it("forgets everything on clear so ids restart for the next run", () => {
		const ledger = new EvidenceLedger();
		ledger.registerResult("tool", result("/a.txt", "x"));
		ledger.clear();

		expect(() => ledger.resolve(["e1"], resolveOptions)).toThrow();
		expect(ledger.registerResult("tool", result("/z.txt", "z"))).toBe("e1");
	});

	it("keeps the stable evidence id independent of what the model writes", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/a.txt", "the exact chunk text"));
		const [ref] = ledger.resolve([id], { ...resolveOptions, fallbackContent: "a paraphrase" });
		const again = new EvidenceLedger();
		const sameId = again.registerResult("tool", result("/a.txt", "the exact chunk text"));
		const [sameRef] = again.resolve([sameId], { ...resolveOptions, fallbackContent: "different paraphrase" });

		expect(ref).toBeDefined();
		expect(normalizeSessionEvidenceRef(ref as never).stableEvidenceId).toBe(
			normalizeSessionEvidenceRef(sameRef as never).stableEvidenceId,
		);
	});

	it("caps stored content so one huge chunk cannot bloat the run", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult("tool", result("/big.txt", "x".repeat(50_000)));
		expect((ledger.resolve([id], resolveOptions)[0]?.content ?? "").length).toBeLessThanOrEqual(2_000);
	});

	it("carries chunk-level metadata the backend provided", () => {
		const ledger = new EvidenceLedger();
		const id = ledger.registerResult(
			"tool",
			result("/a.pdf", "x", { chunkIndex: 7, lineNumber: 12, parserType: "kordoc", documentType: "pdf" }),
		);
		expect(ledger.resolve([id], resolveOptions)[0]).toMatchObject({
			chunkIndex: 7,
			lineNumber: 12,
			parserType: "kordoc",
			documentType: "pdf",
		});
	});
});
