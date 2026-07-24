import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { readQrels, readTopicsTsv } from "../../benchmark/miracl/jsonl.ts";

describe("MIRACL input parsing", () => {
	const roots: string[] = [];
	const makeRoot = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-"));
		roots.push(root);
		return root;
	};

	afterEach(() => roots.splice(0).forEach((root) => rmSync(root, { recursive: true, force: true })));

	it("parses strict topics and qrels", async () => {
		const root = makeRoot();
		mkdirSync(root, { recursive: true });
		writeFileSync(join(root, "topics.tsv"), "q1\t한국어 질문\n");
		writeFileSync(join(root, "qrels.txt"), "q1 Q0 d1 2\nq1 Q0 d2 0\n");
		expect(await readTopicsTsv(join(root, "topics.tsv"))).toEqual([{ queryId: "q1", text: "한국어 질문" }]);
		expect(await readQrels(join(root, "qrels.txt"))).toEqual([
			{ queryId: "q1", documentId: "d1", relevance: 2 },
			{ queryId: "q1", documentId: "d2", relevance: 0 },
		]);
	});

	it("rejects duplicate topic ids and malformed relevance", async () => {
		const root = makeRoot();
		writeFileSync(join(root, "topics.tsv"), "q1\ta\nq1\tb\n");
		writeFileSync(join(root, "qrels.txt"), "q1 Q0 d1 NaN\n");
		await expect(readTopicsTsv(join(root, "topics.tsv"))).rejects.toThrow("duplicate query id");
		await expect(readQrels(join(root, "qrels.txt"))).rejects.toThrow("finite integer");
	});
});
