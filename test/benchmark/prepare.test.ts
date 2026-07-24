import { createHash } from "node:crypto";
import {
	existsSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	renameSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { gzipSync } from "node:zlib";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readJsonLines } from "../../benchmark/miracl/jsonl.ts";
import { prepareMiracl, selectSmokeDataset, validatePreparedManifest } from "../../benchmark/miracl/prepare.ts";
import { MIRACL_FULL_CORPUS_PASSAGES, MIRACL_SOURCES } from "../../benchmark/miracl/profiles.ts";
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
		vi.doUnmock("node:fs/promises");
		vi.resetModules();
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
		expect(manifest.normalized).toEqual({
			queries: {
				sha256: sha256(readFileSync(join(outputDir, "queries.jsonl"))),
				bytes: readFileSync(join(outputDir, "queries.jsonl")).byteLength,
				records: 1,
			},
			qrels: {
				sha256: sha256(readFileSync(join(outputDir, "qrels.jsonl"))),
				bytes: readFileSync(join(outputDir, "qrels.jsonl")).byteLength,
				records: 2,
			},
			corpus: {
				sha256: sha256(readFileSync(join(outputDir, "corpus.jsonl"))),
				bytes: readFileSync(join(outputDir, "corpus.jsonl")).byteLength,
				records: 3,
			},
		});
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
		expect(() => validatePreparedManifest({ ...manifest, unknownRoot: true })).toThrow("unknown field unknownRoot");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				revisions: { ...manifest.revisions, unknownNested: true },
			}),
		).toThrow("revisions");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				sources: {
					...manifest.sources,
					topics: { ...manifest.sources.topics, unknownNested: true },
				},
			}),
		).toThrow("sources.topics");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				selectedIds: { ...manifest.selectedIds, unknownNested: true },
			}),
		).toThrow("selectedIds");
		expect(() =>
			validatePreparedManifest({
				...manifest,
				normalized: {
					...manifest.normalized,
					queries: { ...manifest.normalized.queries, records: 2 },
				},
			}),
		).toThrow("normalized.queries.records");
		const parsedSmoke = JSON.parse(JSON.stringify(manifest)) as unknown;
		const normalizedSmoke = validatePreparedManifest(parsedSmoke);
		expect(normalizedSmoke).toEqual(manifest);
		expect(normalizedSmoke).not.toBe(parsedSmoke);
		expect(normalizedSmoke.revisions).not.toBe((parsedSmoke as typeof manifest).revisions);
	});

	it("streams every record for the full profile and attests normalized outputs without selected ids", async () => {
		const outputDir = makeOutput();
		const fixture = fixtureFetch();

		const manifest = await prepareMiracl(
			{
				profile: "full",
				outputDir,
				fetchImpl: fixture.fetchImpl,
				maxRedirects: 1,
				maxDownloadBytes: 1024 * 1024,
			},
			{ expectedFullCorpusPassages: 5 },
		);

		expect(MIRACL_FULL_CORPUS_PASSAGES).toBe(1_486_752);
		expect(manifest.profile).toBe("full");
		expect(manifest.counts).toEqual({
			queries: 2,
			qrels: 3,
			positiveQrels: 1,
			corpus: 5,
			judgedDocuments: 3,
		});
		expect("seed" in manifest).toBe(false);
		expect("selectedIds" in manifest).toBe(false);
		expect(
			(await readJsonLines<BenchmarkQuery>(join(outputDir, "queries.jsonl"))).map((query) => query.queryId),
		).toEqual(["q1", "q2"]);
		expect((await readJsonLines<Qrel>(join(outputDir, "qrels.jsonl"))).map((qrel) => qrel.queryId)).toEqual([
			"q1",
			"q1",
			"q2",
		]);
		expect(
			(await readJsonLines<CorpusDocument>(join(outputDir, "corpus.jsonl"))).map((document) => document.documentId),
		).toEqual(["d2", "gold", "judged-negative", "d1", "other-gold"]);
		for (const normalized of Object.values(manifest.normalized)) {
			expect(normalized.sha256).toMatch(/^[0-9a-f]{64}$/);
			expect(normalized.bytes).toBeGreaterThan(0);
			expect(normalized.records).toBeGreaterThan(0);
		}
		expect(validatePreparedManifest(manifest, { expectedFullCorpusPassages: 5 })).toEqual(manifest);
		expect(() => validatePreparedManifest(manifest)).toThrow(`${MIRACL_FULL_CORPUS_PASSAGES}`);
		expect(() =>
			validatePreparedManifest(
				{
					...manifest,
					selectedIds: { queryIds: [], documentIds: [] },
				},
				{ expectedFullCorpusPassages: 5 },
			),
		).toThrow("selectedIds");
		expect(() =>
			validatePreparedManifest(
				{
					...manifest,
					counts: { ...manifest.counts, distractors: 0 },
				},
				{ expectedFullCorpusPassages: 5 },
			),
		).toThrow("counts");
		expect(() =>
			validatePreparedManifest(
				{
					...manifest,
					normalized: {
						...manifest.normalized,
						queries: { ...manifest.normalized.queries, unknownNested: true },
					},
				},
				{ expectedFullCorpusPassages: 5 },
			),
		).toThrow("normalized.queries");
		expect(() =>
			validatePreparedManifest(
				{
					...manifest,
					unknownRoot: true,
				},
				{ expectedFullCorpusPassages: 5 },
			),
		).toThrow("unknown field unknownRoot");
		const parsedFull = JSON.parse(JSON.stringify(manifest)) as unknown;
		const normalizedFull = validatePreparedManifest(parsedFull, { expectedFullCorpusPassages: 5 });
		if (normalizedFull.profile !== "full") throw new Error("expected full manifest");
		expect(normalizedFull).toEqual(manifest);
		expect(normalizedFull).not.toBe(parsedFull);
		expect(normalizedFull.normalized).not.toBe((parsedFull as typeof manifest).normalized);
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

	it("aborts a stalled fetch at the configured preparation timeout", async () => {
		const outputDir = makeOutput();
		let observedSignal: AbortSignal | undefined;
		const fetchImpl = vi.fn(
			async (_input: string | URL, init?: RequestInit): Promise<Response> =>
				new Promise((_resolve, reject) => {
					observedSignal = init?.signal ?? undefined;
					if (observedSignal === undefined) {
						setTimeout(() => reject(new Error("missing abort signal")), 25);
						return;
					}
					observedSignal.addEventListener("abort", () => reject(observedSignal?.reason), { once: true });
				}),
		);

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl,
				fetchTimeoutMs: 10,
			}),
		).rejects.toThrow("timed out");
		expect(observedSignal?.aborted).toBe(true);
		expect(existsSync(outputDir)).toBe(false);
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

	it("never follows a preexisting duplicate-partition symlink", async () => {
		const outputDir = makeOutput();
		const external = `${outputDir}-external.txt`;
		const fixture = fixtureFetch();
		writeFileSync(external, "external data\n", { mode: 0o600 });
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			open: async (path: string, flags: string | number, mode?: number) => {
				const pathText = String(path);
				if (!injected && pathText.includes(".duplicate-check/partition-")) {
					injected = true;
					symlinkSync(external, pathText);
				}
				return actual.open(path, flags, mode);
			},
		}));
		vi.resetModules();
		const { prepareMiracl: prepareWithSymlink } = await import("../../benchmark/miracl/prepare.ts");

		await expect(
			prepareWithSymlink({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow();
		expect(readFileSync(external, "utf8")).toBe("external data\n");
	});

	it("bounds duplicate partition handles and refuses a substituted evicted partition", async () => {
		const outputDir = makeOutput();
		const external = `${outputDir}-reopen-target.txt`;
		writeFileSync(external, "external data\n", { mode: 0o600 });
		const padding = "x".repeat(66 * 1024);
		const idsByPartition = new Map<number, string>();
		let candidate = 0;
		while (idsByPartition.size < 256) {
			const id = `partition-candidate-${candidate}-${padding}`;
			const partition = createHash("sha256").update(id, "utf8").digest()[0];
			if (!idsByPartition.has(partition)) idsByPartition.set(partition, id);
			candidate += 1;
		}
		let secondPartitionZero = "";
		while (secondPartitionZero.length === 0) {
			const id = `partition-reopen-${candidate}-${padding}`;
			if (createHash("sha256").update(id, "utf8").digest()[0] === 0 && id !== idsByPartition.get(0)) {
				secondPartitionZero = id;
			}
			candidate += 1;
		}
		const documents = [
			...[...idsByPartition.entries()]
				.sort(([left], [right]) => left - right)
				.map(([, docid]) => ({ docid, title: "partition", text: "body" })),
			{ docid: "gold", title: "gold", text: "body" },
			{ docid: "judged-negative", title: "judged", text: "body" },
			{ docid: "other-gold", title: "other", text: "body" },
			{ docid: secondPartitionZero, title: "reopen", text: "body" },
		];
		const shardSize = Math.ceil(documents.length / 3);
		const overrides = new Map(
			MIRACL_SOURCES.corpus.urls.map((url, index) => [
				url,
				gzipSync(
					`${documents
						.slice(index * shardSize, (index + 1) * shardSize)
						.map((document) => JSON.stringify(document))
						.join("\n")}\n`,
				),
			]),
		);
		const fixture = fixtureFetch(overrides);
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let activePartitionHandles = 0;
		let maxPartitionHandles = 0;
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			open: async (path: string, flags: string | number, mode?: number) => {
				const pathText = String(path);
				const isPartition = pathText.includes(".duplicate-check/partition-");
				if (isPartition && typeof flags === "number" && pathText.endsWith("partition-00.jsonl") && !injected) {
					injected = true;
					renameSync(pathText, `${pathText}.owned`);
					symlinkSync(external, pathText);
				}
				const handle = await actual.open(path, flags, mode);
				if (!isPartition) return handle;
				activePartitionHandles += 1;
				maxPartitionHandles = Math.max(maxPartitionHandles, activePartitionHandles);
				let closed = false;
				return new Proxy(handle, {
					get(target, property) {
						if (property === "close") {
							return async () => {
								if (!closed) {
									closed = true;
									activePartitionHandles -= 1;
								}
								return target.close();
							};
						}
						const value = Reflect.get(target, property, target);
						return typeof value === "function" ? value.bind(target) : value;
					},
				});
			},
		}));
		vi.resetModules();
		const { prepareMiracl: prepareWithReopenRace } = await import("../../benchmark/miracl/prepare.ts");

		await expect(
			prepareWithReopenRace({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: fixture.fetchImpl,
			}),
		).rejects.toThrow("duplicate-check partition 0 changed");
		expect(injected).toBe(true);
		expect(maxPartitionHandles).toBeLessThanOrEqual(32);
		expect(activePartitionHandles).toBe(0);
		expect(readFileSync(external, "utf8")).toBe("external data\n");
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

	it("does not clean a replacement that appears at an owned output pathname", async () => {
		const outputDir = makeOutput();
		const displaced = `${outputDir}-displaced`;
		const replacement = join(outputDir, "replacement.txt");
		const fixture = fixtureFetch();
		const fetchImpl = vi.fn(async (input: string | URL, _init?: RequestInit) => {
			if (fetchImpl.mock.calls.length === 1) {
				renameSync(outputDir, displaced);
				mkdirSync(outputDir);
				writeFileSync(replacement, "replacement data\n");
				throw new Error("injected preparation failure");
			}
			return fixture.fetchImpl(input);
		});

		await expect(
			prepareMiracl({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl,
			}),
		).rejects.toThrow("injected preparation failure");
		expect(readFileSync(replacement, "utf8")).toBe("replacement data\n");
		expect(existsSync(displaced)).toBe(true);
	});

	it("uses nonrecursive identity-checked cleanup for a last-window output replacement", async () => {
		const outputDir = makeOutput();
		const displaced = `${outputDir}-displaced`;
		const replacement = join(outputDir, "replacement.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let replaced = false;
		let usedRecursiveRemoval = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rm: async (path: string, options: Parameters<typeof actual.rm>[1]) => {
				if (options?.recursive === true && String(path).includes("autorag-miracl-prepare-")) {
					usedRecursiveRemoval = true;
				}
				return actual.rm(path, options);
			},
			rmdir: async (path: string) => {
				if (!replaced && String(path) === outputDir) {
					replaced = true;
					renameSync(outputDir, displaced);
					mkdirSync(outputDir);
					writeFileSync(replacement, "replacement data\n");
				}
				return actual.rmdir(path);
			},
		}));
		vi.resetModules();
		const { prepareMiracl: prepareWithCleanup } = await import("../../benchmark/miracl/prepare.ts");

		await expect(
			prepareWithCleanup({
				outputDir,
				queryCount: 1,
				distractorCount: 0,
				fetchImpl: async () => {
					throw new Error("injected preparation failure");
				},
			}),
		).rejects.toThrow("injected preparation failure");
		expect(readFileSync(replacement, "utf8")).toBe("replacement data\n");
		expect(existsSync(displaced)).toBe(true);
		expect(usedRecursiveRemoval).toBe(false);
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
