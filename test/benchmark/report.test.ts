import { lstatSync, mkdtempSync, readFileSync, realpathSync, rmSync, symlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { MethodMetrics } from "../../benchmark/miracl/metrics.ts";
import { writeRunReport } from "../../benchmark/miracl/report.ts";
import type { QueryRunRecord } from "../../benchmark/miracl/types.ts";

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
		queryId: "q2",
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

function manifest(preparedDirectory: string) {
	return {
		profile: "smoke" as const,
		preparedDirectory,
		dataset: {
			normalizationVersion: 1,
			revisions: { topics: "topics-pin", corpus: "corpus-pin" },
			seed: 20260723,
			counts: { queries: 1, qrels: 1, positiveQrels: 1, corpus: 1, judgedDocuments: 1, distractors: 0 },
		},
		methods: ["minsync", "bm25"] as const,
		methodConfig: {
			embedderId: "safe-model",
			endpointKind: "remote" as const,
			apiKeyEnv: "TOKEN",
			dimension: 1024,
			baseUrl: "https://private.example/v1",
			apiKey: "literal-secret",
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
			manifest: manifest(join(parent, "prepared")),
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
		expect(serialized).not.toContain("private.example");
		expect(serialized).not.toContain("literal-secret");
		expect(lstatSync(output).mode & 0o777).toBe(0o700);
		for (const file of ["manifest.json", "results.jsonl", "metrics.json", "summary.md"]) {
			expect(lstatSync(join(output, file)).mode & 0o777).toBe(0o600);
		}
		await expect(
			writeRunReport({ directory: output, manifest: manifest(join(parent, "prepared")), records, metrics }),
		).rejects.toThrow("already exists");
	});

	it("allows only one concurrent publisher and preserves competing output", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const options = { directory: output, manifest: manifest(join(parent, "prepared")), records, metrics };

		const settled = await Promise.allSettled([writeRunReport(options), writeRunReport(options)]);

		expect(settled.filter((result) => result.status === "fulfilled")).toHaveLength(1);
		expect(settled.filter((result) => result.status === "rejected")).toHaveLength(1);
		expect(JSON.parse(readFileSync(join(output, "manifest.json"), "utf8")).schemaVersion).toBe(1);
	});

	it("preserves a competing nonempty destination created at publication time", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const canonicalOutput = join(realpathSync(parent), "run");
		const sentinel = join(output, "keep.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async (source: string, destination: string) => {
				if (!injected && destination === canonicalOutput) {
					injected = true;
					await actual.mkdir(output);
					await actual.writeFile(sentinel, "competing data\n");
				}
				return actual.rename(source, destination);
			},
		}));
		const { writeRunReport: writeWithRace } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithRace({
				directory: output,
				manifest: manifest(join(parent, "prepared")),
				records,
				metrics,
			}),
		).rejects.toThrow();
		expect(readFileSync(sentinel, "utf8")).toBe("competing data\n");
	});

	it("does not recursively clean a replacement staging directory", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const canonicalOutput = join(realpathSync(parent), "run");
		const replacement = join(parent, "replacement.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let injected = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async (source: string, destination: string) => {
				if (!injected && destination === canonicalOutput) {
					injected = true;
					await actual.rename(source, `${source}-owned`);
					await actual.mkdir(source);
					await actual.writeFile(join(source, "replacement.txt"), "replacement data\n");
					writeFileSync(replacement, source);
					throw new Error("injected publication failure");
				}
				return actual.rename(source, destination);
			},
		}));
		const { writeRunReport: writeWithRace } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithRace({
				directory: output,
				manifest: manifest(join(parent, "prepared")),
				records,
				metrics,
			}),
		).rejects.toThrow("injected publication failure");
		const staging = readFileSync(replacement, "utf8");
		expect(readFileSync(join(staging, "replacement.txt"), "utf8")).toBe("replacement data\n");
	});

	it("rechecks staging ownership after moving it for cleanup", async () => {
		const parent = makeRoot();
		const output = join(parent, "run");
		const canonicalOutput = join(realpathSync(parent), "run");
		const displaced = join(parent, "displaced-staging");
		const replacementPathFile = join(parent, "replacement-path.txt");
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let replaced = false;
		const replace = async (path: string) => {
			replaced = true;
			await actual.rename(path, displaced);
			await actual.mkdir(path);
			await actual.writeFile(join(path, "replacement.txt"), "replacement data\n");
			writeFileSync(replacementPathFile, path);
		};
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async (source: string, destination: string) => {
				if (destination === canonicalOutput) {
					throw new Error("injected publication failure");
				}
				if (!replaced && String(destination).includes(".cleanup-")) {
					await replace(source);
				}
				return actual.rename(source, destination);
			},
			rm: async (path: string, options: Parameters<typeof actual.rm>[1]) => {
				if (!replaced && String(path).includes(".staging-")) {
					await replace(path);
				}
				return actual.rm(path, options);
			},
		}));
		const { writeRunReport: writeWithRace } = await import("../../benchmark/miracl/report.ts");

		await expect(
			writeWithRace({
				directory: output,
				manifest: manifest(join(parent, "prepared")),
				records,
				metrics,
			}),
		).rejects.toThrow("injected publication failure");
		const replacement = readFileSync(replacementPathFile, "utf8");
		expect(readFileSync(join(replacement, "replacement.txt"), "utf8")).toBe("replacement data\n");
		expect(lstatSync(displaced).isDirectory()).toBe(true);
	});

	it("rejects symlink destinations without writing through them", async () => {
		const parent = makeRoot();
		const target = join(parent, "target");
		const output = join(parent, "run");
		writeFileSync(target, "sentinel\n");
		symlinkSync(target, output);

		await expect(
			writeRunReport({ directory: output, manifest: manifest(join(parent, "prepared")), records, metrics }),
		).rejects.toThrow("already exists");
		expect(readFileSync(target, "utf8")).toBe("sentinel\n");
	});
});
