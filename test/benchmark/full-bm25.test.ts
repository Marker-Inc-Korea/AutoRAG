import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { FullCorpusBm25Method } from "../../benchmark/miracl/full-bm25.ts";

describe("MIRACL full-corpus BM25", () => {
	const roots: string[] = [];

	afterEach(() => {
		for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
	});

	it("streams an attested corpus directly into Tantivy without parsed mirror files", async () => {
		const parent = mkdtempSync(join(tmpdir(), "autorag-miracl-full-bm25-"));
		roots.push(parent);
		const root = join(parent, "workspace");
		mkdirSync(root, { mode: 0o700 });
		const corpusPath = join(parent, "corpus.jsonl");
		const records = [
			{ documentId: "doc-a", title: "A", text: `needle ${"long ".repeat(700)}` },
			{ documentId: "doc-b", title: "B", text: "needle second result" },
			{ documentId: "doc-c", title: "C", text: "unrelated" },
		];
		const contents = `${records.map((record) => JSON.stringify(record)).join("\n")}\n`;
		writeFileSync(corpusPath, contents, { mode: 0o600 });
		const method = new FullCorpusBm25Method({
			root,
			corpusPath,
			attestation: {
				sha256: createHash("sha256").update(contents).digest("hex"),
				bytes: Buffer.byteLength(contents),
				records: records.length,
			},
		});

		const sync = await method.sync();
		const hits = await method.retrieve("needle", { topK: 10 });

		expect(sync).toMatchObject({ engine: "tantivy", indexedDocuments: 3 });
		expect(sync.indexedChunks).toBeGreaterThan(3);
		expect(hits.map((hit) => hit.source)).toEqual(expect.arrayContaining(["/miracl/doc-a.md", "/miracl/doc-b.md"]));
		expect(readdirSync(root)).toEqual([".autorag"]);
		expect(() => readdirSync(join(root, ".autorag", "parsed"))).toThrow();
	});
});
