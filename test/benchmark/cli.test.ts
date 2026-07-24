import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";
import { runCli } from "../../benchmark/miracl/cli.ts";
import type { CreatedBenchmarkMethods } from "../../benchmark/miracl/methods.ts";
import type { PreparedManifest } from "../../benchmark/miracl/prepare.ts";
import { MIRACL_SOURCES } from "../../benchmark/miracl/profiles.ts";
import type { RetrievalMethod } from "../../src/retrieval/types.ts";

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex");

function preparedManifest(): PreparedManifest {
	const source = (url: string, path: string) => ({ url, path, sha256: sha256(url), bytes: 1 });
	return {
		schemaVersion: 1,
		normalizationVersion: 1,
		profile: "smoke",
		revisions: {
			topics: MIRACL_SOURCES.topics.revision,
			corpus: MIRACL_SOURCES.corpus.revision,
		},
		sources: {
			topics: source(MIRACL_SOURCES.topics.topicsUrl, "downloads/topics.tsv"),
			qrels: source(MIRACL_SOURCES.topics.qrelsUrl, "downloads/qrels.tsv"),
			corpus: MIRACL_SOURCES.corpus.urls.map((url, index) => source(url, `downloads/docs-${index}.jsonl.gz`)),
		},
		seed: 1,
		selectedIds: { queryIds: ["q1"], documentIds: ["doc"] },
		counts: {
			queries: 1,
			qrels: 1,
			positiveQrels: 1,
			corpus: 1,
			judgedDocuments: 1,
			distractors: 0,
		},
		files: { queries: "queries.jsonl", qrels: "qrels.jsonl", corpus: "corpus.jsonl" },
	};
}

function writePrepared(directory: string): void {
	mkdirSync(directory, { mode: 0o700 });
	writeFileSync(join(directory, "prepared-manifest.json"), `${JSON.stringify(preparedManifest())}\n`, { mode: 0o600 });
	writeFileSync(join(directory, "queries.jsonl"), `${JSON.stringify({ queryId: "q1", text: "needle" })}\n`, {
		mode: 0o600,
	});
	writeFileSync(
		join(directory, "qrels.jsonl"),
		`${JSON.stringify({ queryId: "q1", documentId: "doc", relevance: 1 })}\n`,
		{ mode: 0o600 },
	);
	writeFileSync(
		join(directory, "corpus.jsonl"),
		`${JSON.stringify({ documentId: "doc", title: "title", text: "needle" })}\n`,
		{ mode: 0o600 },
	);
}

function methodFactory(fails = false) {
	const retrieval: RetrievalMethod = {
		describe: () => ({
			name: "bm25",
			type: "bm25",
			description: "synthetic benchmark method",
			status: "active",
			capabilities: [],
		}),
		retrieve: async () => {
			if (fails) throw new Error("private query failure");
			return [{ id: "hit", source: "/miracl/doc.md", content: "needle", score: 1, metadata: {} }];
		},
	};
	return vi.fn(
		async (): Promise<CreatedBenchmarkMethods> => ({
			methods: new Map([["bm25", retrieval]]),
			indexingLatencyMs: { bm25: 3 },
		}),
	);
}

describe("MIRACL benchmark CLI", () => {
	const roots: string[] = [];
	const makeRoot = () => {
		const root = mkdtempSync(join(tmpdir(), "autorag-miracl-cli-"));
		roots.push(root);
		return root;
	};

	afterEach(() => {
		for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
		vi.restoreAllMocks();
		vi.doUnmock("node:fs/promises");
		vi.resetModules();
	});

	it("rejects unknown, duplicate, missing, and conflicting flags", async () => {
		await expect(runCli(["run", "--unknown"])).rejects.toThrow("Unknown option");
		await expect(runCli(["run"])).rejects.toThrow("--profile is required");
		await expect(runCli(["run", "--profile", "smoke", "--profile", "full"])).rejects.toThrow("Duplicate option");
		await expect(runCli(["run", "--profile=smoke"])).rejects.toThrow("Unknown option");
		await expect(
			runCli(["run", "--profile", "smoke", "--prepared", "prepared", "--output", "out", "--methods", "bm25,bm25"]),
		).rejects.toThrow("duplicates");
		await expect(
			runCli([
				"run",
				"--profile",
				"smoke",
				"--prepared",
				"prepared",
				"--output",
				"out",
				"--methods",
				"bm25",
				"--config",
				"config.json",
			]),
		).rejects.toThrow("conflicts");
		await expect(runCli(["prepare", "--profile", "smoke", "--output", "out", "--confirm-full"])).rejects.toThrow(
			"--confirm-full",
		);
		await expect(runCli(["evaluate", "--run", "run", "extra"])).rejects.toThrow("Unexpected argument");
	});

	it("requires full confirmation and reports the exact scope before preparation starts", async () => {
		const prepare = vi.fn(async () => preparedManifest());
		const output: string[] = [];
		const directory = join(makeRoot(), "full");

		await expect(
			runCli(["prepare", "--profile", "full", "--output", directory], {
				prepareMiracl: prepare,
				writeStdout: (line) => output.push(line),
			}),
		).rejects.toThrow("--confirm-full");
		expect(prepare).not.toHaveBeenCalled();
		expect(output.join("\n")).toContain("1,486,752");

		await runCli(["prepare", "--profile", "full", "--output", directory, "--confirm-full"], {
			prepareMiracl: prepare,
			writeStdout: (line) => output.push(line),
		});
		expect(prepare).toHaveBeenCalledWith(expect.objectContaining({ profile: "full", outputDir: directory }));
	});

	it("runs and evaluates a prepared profile through isolated method lifecycles", async () => {
		const parent = makeRoot();
		const prepared = join(parent, "prepared");
		const output = join(parent, "run");
		writePrepared(prepared);

		const exitCode = await runCli(
			["run", "--profile", "smoke", "--prepared", prepared, "--output", output, "--methods", "bm25"],
			{ createBenchmarkMethods: methodFactory() },
		);

		expect(exitCode).toBe(0);
		expect(JSON.parse(readFileSync(join(output, "manifest.json"), "utf8")).methods).toEqual(["bm25"]);
		expect(JSON.parse(readFileSync(join(output, "metrics.json"), "utf8")).methods[0]).toMatchObject({
			method: "bm25",
			failureCount: 0,
		});
		expect(await runCli(["evaluate", "--run", output], { writeStdout: vi.fn() })).toBe(0);
	});

	it("publishes query failures durably and returns nonzero", async () => {
		const parent = makeRoot();
		const prepared = join(parent, "prepared");
		const output = join(parent, "run");
		writePrepared(prepared);

		const exitCode = await runCli(
			["run", "--profile", "smoke", "--prepared", prepared, "--output", output, "--methods", "bm25"],
			{ createBenchmarkMethods: methodFactory(true) },
		);

		expect(exitCode).toBe(1);
		expect(readFileSync(join(output, "results.jsonl"), "utf8")).toContain('"errorCode":"retrieval-failed"');
		expect(readFileSync(join(output, "results.jsonl"), "utf8")).not.toContain("private query failure");
		expect(await runCli(["evaluate", "--run", output], { writeStdout: vi.fn() })).toBe(1);
	});

	it("strictly rejects malformed persisted records before evaluation", async () => {
		const parent = makeRoot();
		const prepared = join(parent, "prepared");
		const output = join(parent, "run");
		writePrepared(prepared);
		await runCli(["run", "--profile", "smoke", "--prepared", prepared, "--output", output, "--methods", "bm25"], {
			createBenchmarkMethods: methodFactory(),
		});
		writeFileSync(
			join(output, "results.jsonl"),
			`${JSON.stringify({
				schemaVersion: 2,
				method: "bm25",
				queryId: "q1",
				latencyMs: 1,
				hits: [],
			})}\n`,
		);

		await expect(runCli(["evaluate", "--run", output], { writeStdout: vi.fn() })).rejects.toThrow(
			"record schemaVersion",
		);
	});

	it("strictly rejects unsanitized persisted method configuration", async () => {
		const parent = makeRoot();
		const prepared = join(parent, "prepared");
		const output = join(parent, "run");
		writePrepared(prepared);
		await runCli(["run", "--profile", "smoke", "--prepared", prepared, "--output", output, "--methods", "bm25"], {
			createBenchmarkMethods: methodFactory(),
		});
		const manifestPath = join(output, "manifest.json");
		const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
		manifest.methodConfig = {
			endpointKind: "remote",
			dimension: 1024,
			baseUrl: "https://private.example/v1",
		};
		writeFileSync(manifestPath, `${JSON.stringify(manifest)}\n`);

		await expect(runCli(["evaluate", "--run", output], { writeStdout: vi.fn() })).rejects.toThrow("methodConfig");
	});

	it("does not recursively clean a replacement benchmark workspace", async () => {
		const parent = makeRoot();
		const prepared = join(parent, "prepared");
		const output = join(parent, "run");
		const displaced = join(parent, "displaced-workspace");
		const replacementPathFile = join(parent, "replacement-path.txt");
		writePrepared(prepared);
		const actual = await vi.importActual<typeof import("node:fs/promises")>("node:fs/promises");
		let replaced = false;
		vi.doMock("node:fs/promises", () => ({
			...actual,
			rename: async (source: string, destination: string) => {
				const pathText = String(source);
				if (!replaced && pathText.includes(".run.workspace-") && String(destination).includes(".cleanup-")) {
					replaced = true;
					renameSync(pathText, displaced);
					mkdirSync(pathText);
					writeFileSync(join(pathText, "replacement.txt"), "replacement data\n");
					writeFileSync(replacementPathFile, pathText);
				}
				return actual.rename(source, destination);
			},
		}));
		const { runCli: runWithRace } = await import("../../benchmark/miracl/cli.ts");

		await runWithRace(
			["run", "--profile", "smoke", "--prepared", prepared, "--output", output, "--methods", "bm25"],
			{ createBenchmarkMethods: methodFactory() },
		);

		const replacement = readFileSync(replacementPathFile, "utf8");
		expect(readFileSync(join(replacement, "replacement.txt"), "utf8")).toBe("replacement data\n");
		expect(existsSync(displaced)).toBe(true);
	});
});
