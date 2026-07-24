import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readJsonLines } from "../../benchmark/miracl/jsonl.ts";
import { prepareMiracl, selectSmokeDataset, validatePreparedManifest } from "../../benchmark/miracl/prepare.ts";
import { MIRACL_SOURCES } from "../../benchmark/miracl/profiles.ts";
import type { BenchmarkQuery, CorpusDocument, Qrel } from "../../benchmark/miracl/types.ts";

const sha256 = (bytes: Uint8Array) => createHash("sha256").update(bytes).digest("hex");

function fixtureFetch(overrides: ReadonlyMap<string, Uint8Array> = new Map()) {
	const topics = Buffer.from("q1\t첫 질문\nq2\t둘째 질문\n");
	const qrels = Buffer.from("q1 Q0 gold 2\nq1 Q0 judged-negative 0\nq2 Q0 other-gold 0\n");
	const corpus = [
		gzipSync(
			`${[
				JSON.stringify({ docid: "d2", title: "방해 2", text: "둘" }),
				JSON.stringify({ docid: "gold", title: "정답", text: "정답 본문" }),
			].join("\n")}\n`,
		),
		gzipSync(
			`${[
				JSON.stringify({ docid: "judged-negative", title: "판정 오답", text: "오답" }),
				JSON.stringify({ docid: "d1", title: "방해 1", text: "하나" }),
			].join("\n")}\n`,
		),
		gzipSync(`${JSON.stringify({ docid: "other-gold", title: "다른 정답", text: "본문" })}\n`),
	];
	const bodies = new Map<string, Uint8Array>([
		[MIRACL_SOURCES.topics.topicsUrl, topics],
		[MIRACL_SOURCES.topics.qrelsUrl, qrels],
		...MIRACL_SOURCES.corpus.urls.map((url, index) => [url, corpus[index]] as const),
		...overrides,
	]);
	const redirectedBodies = new Map<string, Uint8Array>();
	for (const [url, body] of bodies) {
		redirectedBodies.set(`https://cdn.hf.co/${sha256(Buffer.from(url))}`, body);
	}

	return {
		bodies,
		fetchImpl: vi.fn(async (input: string | URL) => {
			const url = input.toString();
			const body = bodies.get(url);
			if (body) {
				return new Response(null, {
					status: 302,
					headers: { location: `https://cdn.hf.co/${sha256(Buffer.from(url))}` },
				});
			}
			const redirectedBody = redirectedBodies.get(url);
			if (!redirectedBody) {
				throw new Error(`unexpected network request: ${url}`);
			}
			return new Response(new Uint8Array(redirectedBody), {
				status: 200,
				headers: { "content-length": String(redirectedBody.byteLength) },
			});
		}),
	};
}

function cancellableResponse(headers: Record<string, string> = {}) {
	let cancelled = false;
	const response = new Response(
		new ReadableStream<Uint8Array>({
			cancel() {
				cancelled = true;
			},
		}),
		{ status: 200, headers },
	);
	return { response, wasCancelled: () => cancelled };
}

describe("MIRACL preparation", () => {
	const roots: string[] = [];
	const makeOutput = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-prepare-"));
		roots.push(root);
		return join(root, "prepared");
	};

	afterEach(() => {
		for (const root of roots.splice(0)) {
			rmSync(root, { recursive: true, force: true });
		}
	});

	it("downloads pinned files through bounded redirects and writes normalized smoke data", async () => {
		const outputDir = makeOutput();
		const fixture = fixtureFetch();

		const manifest = await prepareMiracl({
			outputDir,
			seed: 20260723,
			queryCount: 1,
			distractorCount: 1,
			fetchImpl: fixture.fetchImpl,
			maxRedirects: 1,
			maxDownloadBytes: 1024 * 1024,
		});

		expect(fixture.fetchImpl).toHaveBeenCalledTimes(10);
		expect(manifest).toEqual(validatePreparedManifest(manifest));
		expect(manifest.revisions).toEqual({
			topics: MIRACL_SOURCES.topics.revision,
			corpus: MIRACL_SOURCES.corpus.revision,
		});
		expect(manifest.sources.corpus).toHaveLength(3);
		expect(manifest.sources.topics.sha256).toBe(
			sha256(fixture.bodies.get(MIRACL_SOURCES.topics.topicsUrl) as Uint8Array),
		);
		expect(manifest.seed).toBe(20260723);
		expect(manifest.selectedIds.queryIds).toEqual(["q1"]);
		expect(manifest.selectedIds.documentIds).toEqual(["gold", "other-gold", "judged-negative"]);
		expect(manifest.counts).toMatchObject({ queries: 1, qrels: 2, corpus: 3, distractors: 1 });

		const queries = await readJsonLines<BenchmarkQuery>(join(outputDir, "queries.jsonl"));
		const qrels = await readJsonLines<Qrel>(join(outputDir, "qrels.jsonl"));
		const corpus = await readJsonLines<CorpusDocument>(join(outputDir, "corpus.jsonl"));
		expect(queries).toEqual([{ queryId: "q1", text: "첫 질문" }]);
		expect(qrels).toHaveLength(2);
		expect(corpus.map((document) => document.documentId)).toEqual(manifest.selectedIds.documentIds);
		expect(JSON.parse(readFileSync(join(outputDir, "prepared-manifest.json"), "utf8"))).toEqual(manifest);
		expect(existsSync(join(outputDir, ".duplicate-check"))).toBe(false);
		expect(() =>
			validatePreparedManifest({
				...manifest,
				revisions: { ...manifest.revisions, corpus: "unpinned" },
			}),
		).toThrow("corpus revision");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				counts: { ...manifest.counts, corpus: manifest.counts.corpus + 1 },
			}),
		).toThrow("corpus count");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				selectedIds: {
					...manifest.selectedIds,
					documentIds: [...manifest.selectedIds.documentIds].reverse(),
				},
			}),
		).toThrow("deterministic order");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				counts: { ...manifest.counts, positiveQrels: 0 },
			}),
		).toThrow("positive qrel count must cover every selected query");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				counts: {
					...manifest.counts,
					judgedDocuments: manifest.counts.qrels + 1,
					distractors: 0,
				},
			}),
		).toThrow("judged document count exceeds qrels");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				counts: { ...manifest.counts, qrels: 3 },
			}),
		).toThrow("possible query/document pairs");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				counts: { ...manifest.counts, distractors: manifest.counts.distractors + 1 },
			}),
		).toThrow("distractor count");
	});

	it("rejects redirects outside the pinned download hosts before following them", async () => {
		const outputDir = makeOutput();
		const fetchImpl = vi.fn(async () => {
			if (fetchImpl.mock.calls.length === 1) {
				return new Response(null, {
					status: 302,
					headers: { location: "https://127.0.0.1/private-dataset" },
				});
			}
			throw new Error("attempted an untrusted redirect");
		});

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl,
			}),
		).rejects.toThrow("allowed download host");
		expect(fetchImpl).toHaveBeenCalledTimes(1);
		expect(existsSync(outputDir)).toBe(false);
	});

	it("enforces redirect and streamed byte limits and removes partial output", async () => {
		const redirectOutput = makeOutput();
		const fixture = fixtureFetch();
		await expect(
			prepareMiracl({
				outputDir: redirectOutput,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
				maxRedirects: 0,
			}),
		).rejects.toThrow("redirects");
		expect(existsSync(redirectOutput)).toBe(false);

		const byteOutput = makeOutput();
		const oversizedFetch = vi.fn(async () => {
			return new Response(
				new ReadableStream({
					start(controller) {
						controller.enqueue(new Uint8Array(17));
						controller.close();
					},
				}),
				{ status: 200 },
			);
		});
		await expect(
			prepareMiracl({
				outputDir: byteOutput,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: oversizedFetch,
				maxDownloadBytes: 16,
			}),
		).rejects.toThrow("byte limit");
		expect(existsSync(byteOutput)).toBe(false);
	});

	it.each([
		["invalid", "not-a-number", 1024, "invalid content-length"],
		["oversized", "1025", 1024, "byte limit"],
	] as const)("cancels the body for %s Content-Length", async (_case, contentLength, maxBytes, message) => {
		const outputDir = makeOutput();
		const body = cancellableResponse({ "content-length": contentLength });

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: async () => body.response,
				maxDownloadBytes: maxBytes,
			}),
		).rejects.toThrow(message);

		expect(body.wasCancelled()).toBe(true);
		expect(existsSync(outputDir)).toBe(false);
	});

	it("cancels the body when exclusive destination creation fails", async () => {
		const outputDir = makeOutput();
		const body = cancellableResponse();
		const fetchImpl = vi.fn(async () => {
			writeFileSync(join(outputDir, "downloads", "topics.tsv"), "competing file");
			return body.response;
		});

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl,
			}),
		).rejects.toThrow();

		expect(body.wasCancelled()).toBe(true);
		expect(existsSync(outputDir)).toBe(false);
	});

	it("rejects a declared Content-Length mismatch and removes the partial output", async () => {
		const outputDir = makeOutput();
		const fetchImpl = vi.fn(async () => {
			return new Response(new Uint8Array([1, 2]), {
				status: 200,
				headers: { "content-length": "3" },
			});
		});

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl,
			}),
		).rejects.toThrow("declared 3 bytes but returned 2");
		expect(existsSync(outputDir)).toBe(false);
	});

	it("rejects an explicit corpus line larger than 16 MiB", async () => {
		const outputDir = makeOutput();
		const oversizedShard = gzipSync(
			`${JSON.stringify({
				docid: "gold",
				title: "oversized",
				text: "x".repeat(16 * 1024 * 1024 + 1),
			})}\n`,
		);
		const fixture = fixtureFetch(new Map([[MIRACL_SOURCES.corpus.urls[0], oversizedShard]]));

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow("exceeds 16 MiB");
		expect(existsSync(outputDir)).toBe(false);
	});

	it("rejects duplicate document IDs across corpus shards", async () => {
		const outputDir = makeOutput();
		const duplicateId = "cross-shard-duplicate";
		const firstShard = gzipSync(
			`${[
				JSON.stringify({ docid: "gold", title: "정답", text: "본문" }),
				JSON.stringify({ docid: duplicateId, title: "첫 사본", text: "본문" }),
			].join("\n")}\n`,
		);
		const secondShard = gzipSync(
			`${[
				JSON.stringify({ docid: "judged-negative", title: "오답", text: "본문" }),
				JSON.stringify({ docid: duplicateId, title: "둘째 사본", text: "본문" }),
			].join("\n")}\n`,
		);
		const fixture = fixtureFetch(
			new Map([
				[MIRACL_SOURCES.corpus.urls[0], firstShard],
				[MIRACL_SOURCES.corpus.urls[1], secondShard],
			]),
		);

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow(`duplicate corpus document id: ${duplicateId}`);
		expect(existsSync(outputDir)).toBe(false);
	});

	it("still rejects a selected qrel document missing from every corpus shard", async () => {
		const outputDir = makeOutput();
		const shardWithoutGold = gzipSync(`${JSON.stringify({ docid: "d2", title: "방해 문서", text: "본문" })}\n`);
		const fixture = fixtureFetch(new Map([[MIRACL_SOURCES.corpus.urls[0], shardWithoutGold]]));

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow("qrels reference missing corpus document: gold");
		expect(existsSync(outputDir)).toBe(false);
	});

	it("bounds decompressed corpus bytes and preserves a pre-existing output directory", async () => {
		const decompressionOutput = makeOutput();
		const fixture = fixtureFetch();
		await expect(
			prepareMiracl({
				outputDir: decompressionOutput,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
				maxDecompressedBytesPerShard: 8,
			}),
		).rejects.toThrow("decompressed data");
		expect(existsSync(decompressionOutput)).toBe(false);

		const existingOutput = makeOutput();
		mkdirSync(existingOutput);
		writeFileSync(join(existingOutput, "keep.txt"), "keep");
		await expect(
			prepareMiracl({
				outputDir: existingOutput,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow();
		expect(readFileSync(join(existingOutput, "keep.txt"), "utf8")).toBe("keep");
	});
});

describe("MIRACL smoke selection", () => {
	it("keeps every judged document and selects stable distractors", () => {
		const input = {
			queries: [{ queryId: "q1", text: "질문" }],
			qrels: [
				{ queryId: "q1", documentId: "gold", relevance: 2 },
				{ queryId: "q1", documentId: "judged-negative", relevance: 0 },
			],
			corpus: [
				{ documentId: "d2", title: "2", text: "방해 2" },
				{ documentId: "judged-negative", title: "오답", text: "판정된 오답" },
				{ documentId: "gold", title: "정답", text: "정답 문서" },
				{ documentId: "d1", title: "1", text: "방해 1" },
			],
		};
		const options = { seed: 20260723, queryCount: 1, distractorCount: 1 };

		const result = selectSmokeDataset(input, options);

		expect(result.corpus.some((document) => document.documentId === "gold")).toBe(true);
		expect(result.corpus.some((document) => document.documentId === "judged-negative")).toBe(true);
		expect(result.corpus).toHaveLength(3);
		expect(result.corpus.some((document) => document.documentId === "d1")).toBe(true);
		expect(selectSmokeDataset(input, options)).toEqual(result);
		expect(
			selectSmokeDataset(
				{
					queries: [...input.queries].reverse(),
					qrels: [...input.qrels].reverse(),
					corpus: [...input.corpus].reverse(),
				},
				options,
			),
		).toEqual(result);
	});

	it("rejects selection when too few queries have a positive qrel", () => {
		expect(() =>
			selectSmokeDataset(
				{
					queries: [{ queryId: "q1", text: "질문" }],
					qrels: [{ queryId: "q1", documentId: "d1", relevance: 0 }],
					corpus: [{ documentId: "d1", title: "문서", text: "본문" }],
				},
				{ seed: 20260723, queryCount: 1, distractorCount: 0 },
			),
		).toThrow("positive relevance");
	});

	it("fails when a qrel references a missing corpus document", () => {
		expect(() =>
			selectSmokeDataset(
				{
					queries: [{ queryId: "q1", text: "질문" }],
					qrels: [{ queryId: "q1", documentId: "missing", relevance: 1 }],
					corpus: [],
				},
				{ seed: 1, queryCount: 1, distractorCount: 0 },
			),
		).toThrow("missing");
	});

	it("rejects duplicate identifiers and invalid selection counts", () => {
		const input = {
			queries: [
				{ queryId: "q1", text: "첫 질문" },
				{ queryId: "q1", text: "중복 질문" },
			],
			qrels: [{ queryId: "q1", documentId: "d1", relevance: 1 }],
			corpus: [{ documentId: "d1", title: "문서", text: "본문" }],
		};
		expect(() => selectSmokeDataset(input, { seed: 1, queryCount: 1, distractorCount: 0 })).toThrow(
			"duplicate query id",
		);
		expect(() =>
			selectSmokeDataset(
				{
					...input,
					queries: input.queries.slice(0, 1),
				},
				{ seed: 1, queryCount: 0, distractorCount: 0 },
			),
		).toThrow("queryCount");
	});
});
