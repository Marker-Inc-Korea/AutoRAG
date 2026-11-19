import { chmodSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import { ResultMerger } from "../../src/retrieval/merger.ts";
import { OVERFETCH_ABSOLUTE_CAP, RecordCut } from "../../src/retrieval/record-cut.ts";
import type { RetrievalMethod, RetrievalResult } from "../../src/retrieval/types.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";
import { seedMinSyncStage } from "../helpers/seed-minsync-stage.ts";

function chunk(source: string, index: number, score: number): RetrievalResult {
	return { id: `${source}#${index}`, source, content: `${source} passage ${index}`, score, metadata: {} };
}

/** One long document whose 30 passages all outscore ten other documents' single passages. */
function longDocumentPool(): Map<string, RetrievalResult[]> {
	const long = Array.from({ length: 30 }, (_, index) => chunk("/docs/long.md", index, 0.99 - index / 1000));
	const others = Array.from({ length: 10 }, (_, index) => chunk(`/docs/other-${index}.md`, 0, 0.5 - index / 100));
	return new Map([["hybrid", [...long, ...others]]]);
}

describe("ResultMerger with a record cut", () => {
	it("lets a long document take every slot of a plain top-N (control)", () => {
		const merged = new ResultMerger().merge(longDocumentPool(), { topK: 12, dedup: true });
		expect(new Set(merged.map((result) => result.source))).toEqual(new Set(["/docs/long.md"]));
	});

	it("caps the long document and gives the rest of the slots to documents not shown yet", () => {
		const merged = new ResultMerger().merge(longDocumentPool(), {
			topK: 12,
			dedup: true,
			recordCut: new RecordCut(),
		});

		expect(merged).toHaveLength(12);
		// per-record = max(4, 12 / 6) = 4; head 6 = long x4 + two others; reserve 6 = six more others.
		expect(merged.filter((result) => result.source === "/docs/long.md")).toHaveLength(4);
		expect(new Set(merged.map((result) => result.source)).size).toBe(9);
	});

	it("does nothing when the pool already fits topK", () => {
		const pool = new Map([["hybrid", [chunk("/docs/a.md", 0, 0.9), chunk("/docs/a.md", 1, 0.8)]]]);
		const merged = new ResultMerger().merge(pool, { topK: 5, dedup: true, recordCut: new RecordCut() });
		expect(merged).toHaveLength(2);
	});
});

describe("RetrievalEngine with a record cut", () => {
	function stubMethod(results: RetrievalResult[]): RetrievalMethod {
		return {
			describe: () => ({ name: "hybrid", type: "hybrid", description: "", status: "active", capabilities: [] }),
			retrieve: async () => results,
		};
	}

	it("applies the cut to retrieve() and retrieveSelected() when constructed with one", async () => {
		const pool = longDocumentPool().get("hybrid")!;
		const engine = new RetrievalEngine({ recordCut: new RecordCut() });
		engine.register(stubMethod(pool));

		const { results } = await engine.retrieve("q", { topK: 12 });
		expect(results.filter((result) => result.source === "/docs/long.md")).toHaveLength(4);
		const selected = await engine.retrieveSelected("q", {}, { topK: 12 });
		expect(selected.results.filter((result) => result.source === "/docs/long.md")).toHaveLength(4);
	});

	it("keeps the plain top-N without one", async () => {
		const engine = new RetrievalEngine();
		engine.register(stubMethod(longDocumentPool().get("hybrid")!));
		const { results } = await engine.retrieve("q", { topK: 12 });
		expect(new Set(results.map((result) => result.source))).toEqual(new Set(["/docs/long.md"]));
	});
});

describe("MinSyncVectorMethod with a record cut", () => {
	let root: string;
	let binary: string;
	let queryLog: string;
	let hitsPath: string;
	let longMirror: string;
	let keywordMirror: string;

	beforeEach(async () => {
		root = mkdtempSync(join(tmpdir(), "autorag-record-cut-"));
		const source = join(root, "docs");
		const parsedRoot = join(root, ".autorag", "parsed", "files", "docs");
		mkdirSync(source, { recursive: true });
		mkdirSync(parsedRoot, { recursive: true });
		mkdirSync(join(root, ".autorag", "minsync"), { recursive: true });
		longMirror = join(parsedRoot, "long.txt.md");
		keywordMirror = join(parsedRoot, "keyword.txt.md");
		writeFileSync(join(source, "long.txt"), "long\n");
		writeFileSync(join(source, "keyword.txt"), "keyword\n");
		writeFileSync(longMirror, "long mirror\n");
		writeFileSync(keywordMirror, "keyword mirror\n");
		const entry = (name: string, outputPath: string) => ({
			virtualPath: `/docs/${name}`,
			sourcePath: join(source, name),
			outputPath,
			parserName: "plain-text",
			sourceMtimeNs: 1,
			sourceSizeBytes: 5,
			updatedAt: "2026-01-01T00:00:00.000Z",
		});
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/long.txt": entry("long.txt", longMirror),
				"/docs/keyword.txt": entry("keyword.txt", keywordMirror),
			},
		});
		queryLog = join(root, "query-k.log");
		hitsPath = join(root, "hits.json");
		binary = join(root, "fake-minsync.mjs");
		// Honors `-k`, like the real index: a hit ranked past k is never returned.
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { appendFileSync, readFileSync } from "node:fs";
import { createRequire } from "node:module";
const args = process.argv.slice(2);
if (args[0] === "stage") {
  const flagAt = args.indexOf("--state-dir");
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: args[flagAt + 1] });
} else if (args[0] === "query") {
  const k = Number(args[args.indexOf("-k") + 1]);
  appendFileSync(${JSON.stringify(queryLog)}, k + "\\n");
  console.log(JSON.stringify({ results: JSON.parse(readFileSync(${JSON.stringify(hitsPath)}, "utf8")).slice(0, k) }));
} else console.log(JSON.stringify({ ok: true }));
`,
		);
		chmodSync(binary, 0o755);
		// Eight passages of the long document, then the keyword-only document at ranks 9 and 10.
		writeFileSync(
			hitsPath,
			JSON.stringify([
				...Array.from({ length: 8 }, (_, index) => ({
					path: longMirror,
					score: 0.9 - index / 100,
					text: `long passage ${index}`,
					doc_id: `l${index}`,
				})),
				{ path: keywordMirror, score: 0.3, text: "keyword-only passage", doc_id: "k0" },
				{ path: keywordMirror, score: 0.29, text: "keyword-only second", doc_id: "k1" },
			]),
		);
	});

	afterEach(() => {
		rmSync(root, { recursive: true, force: true });
	});

	function method(recordCut?: RecordCut): MinSyncVectorMethod {
		return new MinSyncVectorMethod({
			binaryPath: binary,
			root,
			workspacePath: join(root, ".autorag", "minsync"),
			...(recordCut === undefined ? {} : { recordCut }),
		});
	}

	const askedK = (): number[] => readFileSync(queryLog, "utf8").trim().split("\n").map(Number);

	it("without a cut asks for topK and a long document fills the result", async () => {
		const results = await method().retrieve("q", { topK: 4 });
		expect(askedK()).toEqual([4]);
		expect(results.map((result) => result.content)).toEqual([
			"long passage 0",
			"long passage 1",
			"long passage 2",
			"long passage 3",
		]);
	});

	it("with a cut asks for depth x topK and returns the keyword-only document inside topK", async () => {
		const results = await method(new RecordCut()).retrieve("q", { topK: 4 });

		expect(askedK()).toEqual([12]);
		// per-record = 4, reserve = 2, head = 2: long 0 and 1; reserve: k0; fill: long 2.
		expect(results.map((result) => result.content)).toEqual([
			"long passage 0",
			"long passage 1",
			"long passage 2",
			"keyword-only passage",
		]);
		expect(results).toHaveLength(4);
	});

	it("never fetches past the over-fetch cap, and never less than the caller's own topK", async () => {
		await method(new RecordCut({ depth: 10 })).retrieve("q", { topK: 100 });
		expect(askedK()).toEqual([OVERFETCH_ABSOLUTE_CAP]);
	});

	it("cuts what survives a scope filter, not what MinSync returned", async () => {
		const results = await method(new RecordCut()).retrieve("q", { topK: 4, scope: "/docs/keyword.txt" });
		expect(results.map((result) => result.content)).toEqual(["keyword-only passage", "keyword-only second"]);
	});
});
