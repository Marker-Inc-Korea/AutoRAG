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
		recallAt: { "5": 1, "10": 1, "100": 1 },
		mrrAt10: 1,
		successAt: { "1": 1, "5": 1 },
		ndcgAt10: 1,
		latencyMs: { mean: 4, p50: 4, p95: 4 },
	},
	{
		method: "bm25",
		queryCount: 1,
		failureCount: 0,
		recallAt: { "5": 0.5, "10": 0.5, "100": 0.5 },
		mrrAt10: 0.5,
		successAt: { "1": 0, "5": 1 },
		ndcgAt10: 0.5,
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
			normalizationVersion: 1,
			revisions: { topics: "topics-pin", corpus: "corpus-pin" },
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
					corpus: 1_486_752,
					judgedDocuments: 1,
				},
				normalized: {
					...withoutSeed.normalized,
					corpus: attestation("1", 1_486_752),
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

	it("writes private path-free lock metadata while publishing", async () => {
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
		const rawLock = readFileSync(lock, "utf8");
		const lockMode = lstatSync(lock).mode & 0o777;
		continueResolve();
		await publication;

		expect(lockMode).toBe(0o600);
		expect(JSON.parse(rawLock)).toEqual({
			schemaVersion: 1,
			pid: process.pid,
			startedAt: expect.any(String),
			nonce: expect.stringMatching(/^[0-9a-f-]+$/),
		});
		expect(rawLock).not.toContain(parent);
	});

	it("recovers a valid dead-process lock while live and invalid locks fail closed", async () => {
		const parent = makeRoot();
		const staleOutput = join(parent, "stale-run");
		writeFileSync(
			`${staleOutput}.publish.lock`,
			`${JSON.stringify({
				schemaVersion: 1,
				pid: 2_147_483_647,
				startedAt: "2026-07-24T00:00:00.000Z",
				nonce: "00000000-0000-4000-8000-000000000000",
			})}\n`,
			{ mode: 0o600 },
		);
		await expect(
			writeRunReport({ directory: staleOutput, manifest: manifest() as never, records, metrics }),
		).resolves.toBeUndefined();

		const liveOutput = join(parent, "live-run");
		writeFileSync(
			`${liveOutput}.publish.lock`,
			`${JSON.stringify({
				schemaVersion: 1,
				pid: process.pid,
				startedAt: "2026-07-24T00:00:00.000Z",
				nonce: "00000000-0000-4000-8000-000000000000",
			})}\n`,
			{ mode: 0o600 },
		);
		await expect(
			writeRunReport({ directory: liveOutput, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("being published");

		const invalidOutput = join(parent, "invalid-run");
		writeFileSync(`${invalidOutput}.publish.lock`, "not-json\n", { mode: 0o600 });
		await expect(
			writeRunReport({ directory: invalidOutput, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("being published");
	});

	it("fails closed and preserves a stale-lock pathname replacement", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const lock = `${join(realpathSync(parent), "run")}.publish.lock`;
		const displaced = join(parent, "stale-lock");
		writeFileSync(
			lock,
			`${JSON.stringify({
				schemaVersion: 1,
				pid: 2_147_483_647,
				startedAt: "2026-07-24T00:00:00.000Z",
				nonce: "00000000-0000-4000-8000-000000000000",
			})}\n`,
			{ mode: 0o600 },
		);
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let replaced = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			lstat: async (path: string) => {
				if (!replaced && String(path) === lock) {
					replaced = true;
					renameSync(lock, displaced);
					writeFileSync(lock, "replacement lock\n", { mode: 0o600 });
				}
				return actual.lstat(path);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithLockRace } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithLockRace({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("being published");
		expect(readFileSync(lock, "utf8")).toBe("replacement lock\n");
		expect(existsSync(displaced)).toBe(true);
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

	it("does not recursively remove a last-window staging-directory replacement", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const displaced = join(parent, "owned-staging");
		const replacementPathFile = join(parent, "replacement-path.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let failedSync = false;
		let replaced = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			open: async (path: string, flags: string, mode?: number) => {
				if (!failedSync && String(path).includes(".staging-") && flags === "r") {
					failedSync = true;
					throw new Error("injected staging sync failure");
				}
				return actual.open(path, flags, mode);
			},
			rmdir: async (path: string) => {
				const pathText = String(path);
				if (!replaced && pathText.includes(".staging-")) {
					replaced = true;
					renameSync(pathText, displaced);
					mkdirSync(pathText);
					writeFileSync(join(pathText, "replacement.txt"), "replacement data\n");
					writeFileSync(replacementPathFile, pathText);
				}
				return actual.rmdir(path);
			},
		}));
		vi.resetModules();
		const { writeRunReport: writeWithLastWindow } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithLastWindow({ directory: output, manifest: manifest() as never, records, metrics }),
		).rejects.toThrow("injected staging sync failure");
		const replacement = readFileSync(replacementPathFile, "utf8");
		expect(readFileSync(join(replacement, "replacement.txt"), "utf8")).toBe("replacement data\n");
		expect(existsSync(displaced)).toBe(true);
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
