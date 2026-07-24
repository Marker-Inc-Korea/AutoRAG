import {
	existsSync,
	lstatSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	renameSync,
	rmSync,
	symlinkSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MethodMetrics } from "../../benchmark/miracl/metrics.ts";
import {
	MIRACL_FULL_CORPUS_PASSAGES,
	MIRACL_NORMALIZATION_VERSION,
	MIRACL_SOURCES,
} from "../../benchmark/miracl/profiles.ts";
import {
	normalizeRunManifest,
	normalizeRunMetrics,
	validateQueryRunRecord,
	writeRunReport,
} from "../../benchmark/miracl/report.ts";
import type { QueryRunRecord } from "../../benchmark/miracl/types.ts";

const attestation = (character: string, records?: number) => ({
	sha256: character.repeat(64),
	bytes: 1,
	...(records === undefined ? {} : { records }),
});

const metrics: MethodMetrics[] = [
	{
		method: "minsync",
		queryCount: 1,
		failureCount: 0,
		recallAt: { "5": 0, "10": 0, "100": 0 },
		mrrAt10: 0,
		successAt: { "1": 0, "5": 0 },
		ndcgAt10: 0,
		latencyMs: { mean: 4, p50: 4, p95: 4 },
	},
	{
		method: "bm25",
		queryCount: 1,
		failureCount: 0,
		recallAt: { "5": 1, "10": 1, "100": 1 },
		mrrAt10: 1,
		successAt: { "1": 1, "5": 1 },
		ndcgAt10: 1,
		latencyMs: { mean: 2, p50: 2, p95: 2 },
	},
];

const records: QueryRunRecord[] = [
	{
		schemaVersion: 1,
		method: "minsync",
		queryId: "q1",
		latencyMs: 4,
		hits: [{ documentId: "b", score: 1, rank: 1 }],
	},
	{
		schemaVersion: 1,
		method: "bm25",
		queryId: "q1",
		latencyMs: 2,
		hits: [{ documentId: "a", score: 1, rank: 1 }],
	},
];

function manifest() {
	return {
		schemaVersion: 1 as const,
		profile: "smoke" as const,
		dataset: {
			normalizationVersion: MIRACL_NORMALIZATION_VERSION,
			revisions: {
				topics: MIRACL_SOURCES.topics.revision,
				corpus: MIRACL_SOURCES.corpus.revision,
			},
			seed: 20260723,
			counts: { queries: 1, qrels: 1, positiveQrels: 1, corpus: 1, judgedDocuments: 1, distractors: 0 },
			input: {
				topics: attestation("a"),
				qrels: attestation("b"),
				corpus: [attestation("c"), attestation("d"), attestation("e")],
			},
			normalized: {
				queries: attestation("f", 1),
				qrels: attestation("0", 1),
				corpus: attestation("1", 1),
			},
			evaluation: {
				schemaVersion: 1 as const,
				qrels: [{ queryId: "q1", documentId: "a", relevance: 1 }],
			},
		},
		methods: ["minsync", "bm25"] as const,
		methodConfig: {
			embedderId: "safe-model",
			endpointKind: "remote" as const,
			apiKeyEnv: "TOKEN",
			dimension: 1024,
		},
		environment: {
			autoRagCommit: "abc123",
			platform: "darwin",
			architecture: "arm64",
			node: "v24",
			bun: "1.3",
			measuredAt: "2026-07-24T00:00:00.000Z",
		},
	};
}

describe("MIRACL run reports", () => {
	const roots: string[] = [];
	const makeRoot = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-report-"));
		roots.push(root);
		return root;
	};

	afterEach(() => {
		for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
		vi.doUnmock("node:fs/promises");
		vi.resetModules();
	});

	it("writes private versioned reports in stable order without secrets", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");

		await writeRunReport({
			directory: output,
			manifest: manifest() as never,
			records,
			metrics,
			indexingLatencyMs: { minsync: 20, bm25: 10 },
			peakRssBytes: 123_456,
		});

		const serialized = ["manifest.json", "results.jsonl", "metrics.json", "summary.md"]
			.map((file) => readFileSync(join(output, file), "utf8"))
			.join("\n");
		const persistedManifest = JSON.parse(readFileSync(join(output, "manifest.json"), "utf8"));
		const persistedMetrics = JSON.parse(readFileSync(join(output, "metrics.json"), "utf8"));
		expect(persistedManifest.schemaVersion).toBe(1);
		expect(persistedManifest.methods).toEqual(["bm25", "minsync"]);
		expect(persistedManifest.methodConfig).toEqual({
			embedderId: "safe-model",
			endpointKind: "remote",
			apiKeyEnv: "TOKEN",
			dimension: 1024,
		});
		expect(readFileSync(join(output, "results.jsonl"), "utf8").split("\n")[0]).toContain('"method":"bm25"');
		expect(persistedMetrics).toMatchObject({
			schemaVersion: 1,
			indexingLatencyMs: { bm25: 10, minsync: 20 },
			peakRssBytes: 123_456,
		});
		expect(persistedMetrics.methods.map((entry: MethodMetrics) => entry.method)).toEqual(["bm25", "minsync"]);
		expect(readFileSync(join(output, "summary.md"), "utf8")).toContain("| Method | nDCG@10 |");
		expect(readFileSync(join(output, "summary.md"), "utf8")).toContain("## Limitations");
		expect(serialized).not.toContain(parent);
		expect(lstatSync(output).mode & 0o777).toBe(0o700);
		for (const file of ["manifest.json", "results.jsonl", "metrics.json", "summary.md"]) {
			expect(lstatSync(join(output, file)).mode & 0o777).toBe(0o600);
		}
		await expect(
			writeRunReport({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("already exists");
	});

	it("recursively normalizes exact discriminated run schemas", () => {
		const smoke = manifest();
		expect(() => normalizeRunManifest({ ...smoke, unknownRoot: true } as never)).toThrow("unknown field unknownRoot");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: {
					...smoke.dataset,
					normalized: {
						...smoke.dataset.normalized,
						qrels: { ...smoke.dataset.normalized.qrels, unknownNested: true },
					},
				},
			} as never),
		).toThrow("dataset normalized qrels");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				methodConfig: { ...smoke.methodConfig, baseUrl: "https://private.example" },
			} as never),
		).toThrow("methodConfig");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: { ...smoke.dataset, normalizationVersion: MIRACL_NORMALIZATION_VERSION + 1 },
			} as never),
		).toThrow("normalizationVersion");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: {
					...smoke.dataset,
					revisions: { ...smoke.dataset.revisions, topics: "unpinned" },
				},
			} as never),
		).toThrow("topics revision");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: {
					...smoke.dataset,
					counts: { ...smoke.dataset.counts, positiveQrels: 0 },
					evaluation: {
						schemaVersion: 1,
						qrels: [{ queryId: "q1", documentId: "a", relevance: 0 }],
					},
				},
			} as never),
		).toThrow("positive");
		const parsedSmoke = JSON.parse(JSON.stringify(smoke)) as unknown;
		const normalizedSmoke = normalizeRunManifest(parsedSmoke as never);
		expect(normalizedSmoke).not.toBe(parsedSmoke);
		expect(normalizedSmoke.dataset).not.toBe((parsedSmoke as typeof smoke).dataset);

		const { seed: _seed, ...withoutSeed } = smoke.dataset;
		const full = {
			...smoke,
			profile: "full" as const,
			dataset: {
				...withoutSeed,
				counts: {
					queries: 1,
					qrels: 1,
					positiveQrels: 1,
					corpus: MIRACL_FULL_CORPUS_PASSAGES,
					judgedDocuments: 1,
				},
				normalized: {
					...withoutSeed.normalized,
					corpus: attestation("1", MIRACL_FULL_CORPUS_PASSAGES),
				},
			},
		};
		expect(() =>
			normalizeRunManifest({
				...full,
				dataset: { ...full.dataset, seed: 1 },
			} as never),
		).toThrow("dataset");
		expect(() =>
			normalizeRunManifest({
				...full,
				dataset: {
					...full.dataset,
					counts: { ...full.dataset.counts, distractors: 0 },
				},
			} as never),
		).toThrow("counts");
		expect(() =>
			normalizeRunManifest({
				...full,
				dataset: {
					...full.dataset,
					counts: { ...full.dataset.counts, corpus: MIRACL_FULL_CORPUS_PASSAGES - 1 },
					normalized: {
						...full.dataset.normalized,
						corpus: attestation("1", MIRACL_FULL_CORPUS_PASSAGES - 1),
					},
				},
			} as never),
		).toThrow("full corpus");

		const normalizedMetrics = normalizeRunMetrics({
			schemaVersion: 1,
			methods: metrics,
			indexingLatencyMs: { bm25: 1 },
		});
		expect(normalizedMetrics).not.toBe(metrics);
		expect(() =>
			normalizeRunMetrics({
				schemaVersion: 1,
				methods: metrics,
				indexingLatencyMs: {},
				unknownRoot: true,
			} as never),
		).toThrow("unknown field unknownRoot");
	});

	it("bounds embedded qrels, MIRACL ids, and canonical manifest bytes", () => {
		const smoke = manifest();
		const tooManyQrels = Array.from({ length: 10_001 }, (_, index) => ({
			queryId: "q1",
			documentId: `${index}`,
			relevance: 1,
		}));
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: {
					...smoke.dataset,
					counts: {
						queries: 1,
						qrels: tooManyQrels.length,
						positiveQrels: tooManyQrels.length,
						corpus: tooManyQrels.length,
						judgedDocuments: tooManyQrels.length,
						distractors: 0,
					},
					normalized: {
						...smoke.dataset.normalized,
						qrels: attestation("0", tooManyQrels.length),
						corpus: attestation("1", tooManyQrels.length),
					},
					evaluation: { schemaVersion: 1, qrels: tooManyQrels },
				},
			} as never),
		).toThrow("at most 10000");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				dataset: {
					...smoke.dataset,
					evaluation: {
						schemaVersion: 1,
						qrels: [{ queryId: "q".repeat(129), documentId: "a", relevance: 1 }],
					},
				},
			} as never),
		).toThrow("128 bytes");
		expect(() =>
			normalizeRunManifest({
				...smoke,
				environment: { ...smoke.environment, autoRagCommit: "a".repeat(4 * 1024 * 1024) },
			} as never),
		).toThrow("manifest");
	});

	it("requires persisted hit ranks to be contiguous and bounded at 100", () => {
		const base = {
			schemaVersion: 1,
			method: "bm25",
			queryId: "q1",
			latencyMs: 1,
		};
		expect(() =>
			validateQueryRunRecord({
				...base,
				hits: [
					{ documentId: "a", score: 2, rank: 1 },
					{ documentId: "b", score: 1, rank: 3 },
				],
			}),
		).toThrow("contiguous");
		expect(() =>
			validateQueryRunRecord({
				...base,
				hits: [{ documentId: "a", score: 1, rank: 101 }],
			}),
		).toThrow("at most 100");
		expect(() =>
			validateQueryRunRecord({
				...base,
				hits: Array.from({ length: 101 }, (_, index) => ({
					documentId: `d${index}`,
					score: 101 - index,
					rank: index + 1,
				})),
			}),
		).toThrow("at most 100");
	});

	it("publishes only a complete query-method grid with recomputed matching metrics", async () => {
		const parent = makeRoot();
		const twoQueryManifest = {
			...manifest(),
			dataset: {
				...manifest().dataset,
				counts: {
					queries: 2,
					qrels: 2,
					positiveQrels: 2,
					corpus: 2,
					judgedDocuments: 2,
					distractors: 0,
				},
				normalized: {
					...manifest().dataset.normalized,
					queries: attestation("f", 2),
					qrels: attestation("0", 2),
					corpus: attestation("1", 2),
				},
				evaluation: {
					schemaVersion: 1 as const,
					qrels: [
						{ queryId: "q1", documentId: "a", relevance: 1 },
						{ queryId: "q2", documentId: "b", relevance: 1 },
					],
				},
			},
		};
		await expect(
			writeRunReport({
				directory: join(parent, "incomplete"),
				manifest: twoQueryManifest as never,
				records,
				metrics,
			}),
		).rejects.toThrow("incomplete");
		await expect(
			writeRunReport({
				directory: join(parent, "wrong-metrics"),
				manifest: manifest() as never,
				records,
				metrics: metrics.map((entry) => (entry.method === "bm25" ? { ...entry, ndcgAt10: 0.25 } : entry)),
			}),
		).rejects.toThrow("metrics do not match");
	});

	it("publishes without a lock file", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let enteredResolve!: () => void;
		let continueResolve!: () => void;
		const entered = new Promise<void>((resolve) => {
			enteredResolve = resolve;
		});
		const continuation = new Promise<void>((resolve) => {
			continueResolve = resolve;
		});
		vi.doMock("node:fs/promises", () => ({
			...actual,
			open: async (path: string, flags: string, mode?: number) => {
				if (String(path).endsWith("manifest.json") && flags === "wx") {
					enteredResolve();
					await continuation;
				}
				return actual.open(path, flags, mode);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writePaused } = await import("../../benchmark/miracl/report.ts");
		const publication = writePaused({ directory: output, manifest: manifest() as never, records, metrics });
		await entered;
		const lock = `${join(realpathSync(parent), "run")}.publish.lock`;
		const lockExists = existsSync(lock);
		continueResolve();
		await publication;

		expect(lockExists).toBe(false);
		expect(existsSync(lock)).toBe(false);
	});

	it("rejects and preserves a replaced regular staging child", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const displaced = join(parent, "owned-manifest");
		const stagingPathFile = join(parent, "staging-path.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			lstat: async (path: string) => {
				const pathText = String(path);
				if (!injected && pathText.includes(".staging-") && pathText.endsWith("manifest.json")) {
					injected = true;
					renameSync(pathText, displaced);
					writeFileSync(pathText, "replacement data\n", { mode: 0o600 });
					writeFileSync(stagingPathFile, pathText);
				}
				return actual.lstat(path);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithReplacement } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithReplacement({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("report file changed");
		const stagingChild = readFileSync(stagingPathFile, "utf8");
		expect(readFileSync(stagingChild, "utf8")).toBe("replacement data\n");
		expect(readFileSync(displaced, "utf8")).toContain('"schemaVersion":1');
	});

	it("rejects and preserves a symlink replacement staging child", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const external = join(parent, "external.txt");
		const stagingPathFile = join(parent, "staging-path.txt");
		writeFileSync(external, "external data\n");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			lstat: async (path: string) => {
				const pathText = String(path);
				if (!injected && pathText.includes(".staging-") && pathText.endsWith("manifest.json")) {
					injected = true;
					renameSync(pathText, `${pathText}.owned`);
					symlinkSync(external, pathText);
					writeFileSync(stagingPathFile, pathText);
				}
				return actual.lstat(path);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithSymlink } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithSymlink({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("report file changed");
		expect(readFileSync(external, "utf8")).toBe("external data\n");
		expect(lstatSync(readFileSync(stagingPathFile, "utf8")).isSymbolicLink()).toBe(true);
	});

	it("leaves private staging fail-closed without pathname cleanup", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let failedSync = false;
		let stagingPath: string | undefined;
		let pathnameCleanupAttempted = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			open: async (path: string, flags: string, mode?: number) => {
				if (!failedSync && String(path).includes(".staging-") && flags === "r") {
					failedSync = true;
					stagingPath = String(path);
					throw new Error("injected staging sync failure");
				}
				return actual.open(path, flags, mode);
			},
			unlink: async (path: string) => {
				if (String(path).includes(".staging-")) pathnameCleanupAttempted = true;
				return actual.unlink(path);
			},
			rmdir: async (path: string) => {
				if (String(path).includes(".staging-")) pathnameCleanupAttempted = true;
				return actual.rmdir(path);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithLastWindow } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithLastWindow({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("injected staging sync failure");
		expect(stagingPath).toBeDefined();
		expect(existsSync(stagingPath as string)).toBe(true);
		expect(lstatSync(stagingPath as string).mode & 0o077).toBe(0);
		expect(pathnameCleanupAttempted).toBe(false);
	});

	it("preserves a final-window destination replacement and reports corruption", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const canonicalOutput = join(realpathSync(parent), "run");
		const displaced = join(parent, "published-report");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			lstat: async (path: string) => {
				const pathText = String(path);
				const stats = await actual.lstat(path);
				if (!injected && pathText === canonicalOutput && stats.isDirectory()) {
					injected = true;
					renameSync(pathText, displaced);
					mkdirSync(pathText);
					writeFileSync(join(pathText, "replacement.txt"), "replacement data\n");
				}
				return stats;
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithFinalReplacement } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithFinalReplacement({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow(/changed|corrupt/u);
		expect(readFileSync(join(output, "replacement.txt"), "utf8")).toBe("replacement data\n");
		expect(readFileSync(join(displaced, "manifest.json"), "utf8")).toContain('"schemaVersion":1');
	});

	it("fails closed when no true no-replace publication runtime is available", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const actual = await vi.importActual<typeof import("node:child_process")>("node:child_process");
		vi.doMock("node:child_process", () => ({
			...actual,
			execFile: (
				_file: string,
				_args: readonly string[],
				_options: unknown,
				callback: (error: NodeJS.ErrnoException) => void,
			) => {
				const error = new Error("runtime unavailable") as NodeJS.ErrnoException;
				error.code = "ENOENT";
				callback(error);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithoutPrimitive } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithoutPrimitive({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("no-replace");
		expect(existsSync(output)).toBe(false);
	});

	it("allows only one concurrent publisher and preserves competing output", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const options = { directory: output, manifest: manifest() as never, records, metrics };

		const settled = await Promise.allSettled([writeRunReport(options), writeRunReport(options)]);

		expect(settled.filter((result) => result.status === "fulfilled")).toHaveLength(1);
		expect(settled.filter((result) => result.status === "rejected")).toHaveLength(1);
		expect(JSON.parse(readFileSync(join(output, "manifest.json"), "utf8")).schemaVersion).toBe(1);
	});

	it("preserves a competing nonempty destination created at publication time", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const sentinel = join(output, "keep.txt");
		mkdirSync(output);
		writeFileSync(sentinel, "competing data\n");

		await expect(
			writeRunReport({
				directory: output,
				manifest: manifest() as never,
				records,
				metrics,
			}),
		).rejects.toThrow();
		expect(readFileSync(sentinel, "utf8")).toBe("competing data\n");
	});

	it("rejects symlink destinations without writing through them", async () => {
		const parent = makeRoot();
		const target = join(parent, "target");
		const output = join(parent, "run");
		writeFileSync(target, "sentinel\n");
		symlinkSync(target, output);

		await expect(
			writeRunReport({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("already exists");
		expect(readFileSync(target, "utf8")).toBe("sentinel\n");
	});
});
