import {
	chmodSync,
	copyFileSync,
	existsSync,
	lstatSync,
	mkdirSync,
	mkdtempSync,
	readFileSync,
	realpathSync,
	rmSync,
	statSync,
	symlinkSync,
	utimesSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { delimiter, join } from "node:path";
import { parse } from "smol-toml";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	ensureLocalEmbedder,
	ensureMinSyncBinary,
	MINSYNC_VERSION,
	MinSyncClient,
	MinSyncHybridMethod,
	MinSyncQueryError,
	MinSyncReleaseError,
	MinSyncRequiredError,
	MinSyncVectorMethod,
	minSyncConfigPath,
	rewriteEmbedderConfig,
} from "../../src/minsync/index.ts";
import type { MinSyncEmbedderConfig } from "../../src/minsync/types.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";
import type { RetrievalFallbackEvent } from "../../src/retrieval/types.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";
import { seedMinSyncStage } from "../helpers/seed-minsync-stage.ts";

let root: string;
let source: string;
let parsedOutput: string;
let minsyncBinary: string;
let minsyncWorkspace: string;
let logPath: string;

beforeEach(async () => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-test-"));
	source = join(root, "docs");
	const parsedRoot = join(root, ".autorag", "parsed", "files", "docs");
	parsedOutput = join(parsedRoot, "policy.txt.md");
	minsyncWorkspace = join(root, ".autorag", "minsync");
	logPath = join(root, "minsync-calls.jsonl");
	minsyncBinary = join(root, "fake-minsync.mjs");
	mkdirSync(source, { recursive: true });
	mkdirSync(parsedRoot, { recursive: true });
	mkdirSync(minsyncWorkspace, { recursive: true });
	writeFileSync(join(source, "policy.txt"), "raw policy source\n");
	writeFileSync(parsedOutput, "Parsed renewal policy with cancellation terms.\n");
	await seedMinSyncStage(root, {
		version: 1,
		entries: {
			"/docs/policy.txt": {
				virtualPath: "/docs/policy.txt",
				sourcePath: join(source, "policy.txt"),
				outputPath: parsedOutput,
				parserName: "plain-text",
				sourceMtimeNs: 1,
				sourceSizeBytes: 18,
				updatedAt: "2026-01-01T00:00:00.000Z",
			},
		},
	});
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

/** What each fake-minsync invocation saw of OPENAI_API_KEY, kept apart from the call log other tests compare exactly. */
function loggedEnv(): { command: string; openaiKey: string | null }[] {
	const path = `${logPath}.env`;
	return existsSync(path)
		? readFileSync(path, "utf8")
				.trim()
				.split("\n")
				.filter((line) => line.length > 0)
				.map((line) => JSON.parse(line) as { command: string; openaiKey: string | null })
		: [];
}

function writeFakeMinSync(queryJson: string, strictQuery = false, syncDelayMs = 0): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, cwd: process.cwd() }) + "\\n");
appendFileSync(${JSON.stringify(`${logPath}.env`)}, JSON.stringify({ command: args[0], openaiKey: process.env.OPENAI_API_KEY ?? null }) + "\\n");

if (args[0] === "stage") { createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({stateDir: args[args.indexOf("--state-dir") + 1]}); } else {
if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  const languageIndex = args.indexOf("--language");
  const lexical = languageIndex === -1 ? "" : "\\n[lexical]\\nlanguage = \\"" + args[languageIndex + 1] + "\\"\\n";
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n" + lexical);
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}

if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}

if (args[0] === "sync") {
  ${syncDelayMs > 0 ? `await new Promise((resolve) => setTimeout(resolve, ${syncDelayMs}));` : ""}
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ files_processed: 1, files_processed_paths: ["files/docs/policy.txt.md"] }));
  process.exit(0);
}

if (args[0] === "query") {
  ${strictQuery ? 'const supported = args.length === 8 && args[1] === "--format" && args[2] === "json" && args[3] === "--mode" && args[4] === "bm25" && args[5] === "-k" && !args[6].startsWith("--");\n  if (!supported) { console.error("unsupported query arguments: " + args.join(" ")); process.exit(2); }' : ""}
  console.log(${JSON.stringify(queryJson)});
  process.exit(0);
}

console.error("unexpected fake minsync command: " + args.join(" "));
process.exit(2);
}
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function loggedCalls(): readonly string[] {
	return readFileSync(logPath, "utf8")
		.trim()
		.split("\n")
		.filter((line) => line.length > 0);
}

function minSyncCwd(): string {
	return realpathSync(minsyncWorkspace);
}

function requireValue<T>(value: T | undefined, label: string): T {
	if (value === undefined) throw new Error(`missing ${label}`);
	return value;
}

describe("MinSyncClient", () => {
	it("uses the official query command with its selected mode", async () => {
		// Given
		writeFakeMinSync(JSON.stringify({ results: [{ path: parsedOutput, score: 0.9, text: "semantic hit" }] }), true);
		const client = new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace });

		// When
		const results = await client.query("renewal cancellation", 2, "bm25");

		// Then
		expect(results).toEqual([{ path: parsedOutput, score: 0.9, text: "semantic hit" }]);
		expect(JSON.parse(loggedCalls()[0] ?? "{}").args).toEqual([
			"query",
			"--format",
			"json",
			"--mode",
			"bm25",
			"-k",
			"2",
			"renewal cancellation",
		]);
	});

	it("does not rewrite the managed config during a query", async () => {
		const config = join(minsyncWorkspace, ".minsync", "config.toml");
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		const original = '[embedder]\nid = "indexed"\n';
		writeFileSync(config, original);
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const client = new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: { id: "query-only", dimension: 1024 },
		});

		await client.query("renewal cancellation", 2);

		expect(readFileSync(config, "utf8")).toBe(original);
	});

	it("surfaces the native query failure instead of returning empty hits", async () => {
		// Given
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
if (process.argv[2] === "query") {
  console.error("embedding failed: local endpoint unavailable");
  process.exit(5);
}
process.exit(2);
`,
		);
		chmodSync(minsyncBinary, 0o755);
		const client = new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace });

		// When
		const query = client.query("renewal cancellation", 2);

		// Then
		await expect(query).rejects.toBeInstanceOf(MinSyncQueryError);
		await expect(query).rejects.toMatchObject({
			code: 5,
			stderr: "embedding failed: local endpoint unavailable\n",
		});
	});

	it("stops a running query when its retrieval signal aborts", async () => {
		// A steer aborts the baseline prefetch's MinSync query: the process must
		// stop and free the workspace instead of running to completion. The fake
		// query announces itself, then never finishes on its own.
		const startedMarker = join(root, "query-started");
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { writeFileSync } from "node:fs";
import { createRequire } from "node:module";
if (process.argv[2] === "stage") {
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: process.argv[process.argv.indexOf("--state-dir") + 1] });
} else
if (process.argv[2] === "query") {
  writeFileSync(${JSON.stringify(startedMarker)}, "");
  setInterval(() => {}, 1 << 30);
} else {
  process.exit(2);
}
`,
		);
		chmodSync(minsyncBinary, 0o755);
		const method = new MinSyncVectorMethod({ binaryPath: minsyncBinary, root, workspacePath: minsyncWorkspace });
		const controller = new AbortController();

		const query = method.retrieve("renewal cancellation", { topK: 2, signal: controller.signal });
		// Spawning node is slow under a loaded parallel suite: wait on the marker, not the 1s default.
		await vi.waitFor(() => expect(existsSync(startedMarker)).toBe(true), { timeout: 15_000 });
		controller.abort();

		await expect(query).rejects.toThrow(/aborted/i);
	});
});

describe("local embedder preflight", () => {
	it("starts Ollama when its direct OpenAI-compatible endpoint is unavailable", async () => {
		// Given
		let ready = false;
		let starts = 0;

		// When
		await ensureLocalEmbedder({
			baseUrl: "http://127.0.0.1:11434/v1",
			probe: async () => ready,
			start: async () => {
				starts += 1;
				ready = true;
			},
			timeoutMs: 1_000,
		});

		// Then
		expect(starts).toBe(1);
	});

	it("leaves TEI endpoints caller-managed", async () => {
		// Given
		let starts = 0;

		// When
		await ensureLocalEmbedder({
			baseUrl: "http://127.0.0.1:18080",
			start: async () => {
				starts += 1;
			},
		});

		// Then
		expect(starts).toBe(0);
	});
});

describe("MinSyncVectorMethod", () => {
	it("syncs parsed mirror files through minsync sync when a mirror index exists", async () => {
		// Given
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		const result = await method.sync();

		// Then
		expect(result).toMatchObject({ synced: 1 });
		expect(loggedCalls()).toContainEqual(
			JSON.stringify({ args: ["init", "--format", "json", "--language", "multilingual"], cwd: minSyncCwd() }),
		);
		expect(loggedCalls()).toContainEqual(JSON.stringify({ args: ["check", "--format", "json"], cwd: minSyncCwd() }));
		expect(loggedCalls()).toContainEqual(
			JSON.stringify({ args: ["sync", "--full", "--format", "json"], cwd: minSyncCwd() }),
		);
	});

	it("updates only the changed parsed mirror file", async () => {
		// Given
		const guideOutput = join(root, ".autorag", "parsed", "files", "docs", "guide.txt.md");
		writeFileSync(join(source, "guide.txt"), "raw guide source\n");
		writeFileSync(guideOutput, "Parsed guide that does not change.\n");
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/guide.txt": {
					virtualPath: "/docs/guide.txt",
					sourcePath: join(source, "guide.txt"),
					outputPath: guideOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 17,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});
		await method.sync();
		const stagedGuide = join(minsyncWorkspace, "files", "docs", "guide.txt.md");
		const stagedPolicy = join(minsyncWorkspace, "files", "docs", "policy.txt.md");
		const preservedTime = new Date("2020-01-01T00:00:00.000Z");
		utimesSync(stagedGuide, preservedTime, preservedTime);

		// When
		writeFileSync(parsedOutput, "Parsed renewal policy with updated cancellation terms.\n");
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/guide.txt": {
					virtualPath: "/docs/guide.txt",
					sourcePath: join(source, "guide.txt"),
					outputPath: guideOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 17,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 2,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-02T00:00:00.000Z",
				},
			},
		});
		await method.sync();

		// Then
		expect(readFileSync(stagedPolicy, "utf8")).toContain("updated cancellation terms");
		expect(readFileSync(stagedGuide, "utf8")).toBe("Parsed guide that does not change.\n");
		expect(statSync(stagedGuide).mtimeMs).toBe(preservedTime.getTime());
	});

	it("adds and removes only corresponding staged mirror files", async () => {
		// Given
		const archiveOutput = join(root, ".autorag", "parsed", "files", "docs", "archive.txt.md");
		writeFileSync(join(source, "archive.txt"), "raw archive source\n");
		writeFileSync(archiveOutput, "Parsed archive to remove.\n");
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/archive.txt": {
					virtualPath: "/docs/archive.txt",
					sourcePath: join(source, "archive.txt"),
					outputPath: archiveOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 19,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});
		await method.sync();
		const stagedArchive = join(minsyncWorkspace, "files", "docs", "archive.txt.md");
		const stagedPolicy = join(minsyncWorkspace, "files", "docs", "policy.txt.md");
		const preservedTime = new Date("2020-01-01T00:00:00.000Z");
		utimesSync(stagedPolicy, preservedTime, preservedTime);
		const noticeOutput = join(root, ".autorag", "parsed", "files", "docs", "notice.txt.md");
		writeFileSync(join(source, "notice.txt"), "raw notice source\n");
		writeFileSync(noticeOutput, "Parsed new notice.\n");
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/notice.txt": {
					virtualPath: "/docs/notice.txt",
					sourcePath: join(source, "notice.txt"),
					outputPath: noticeOutput,
					parserName: "plain-text",
					sourceMtimeNs: 2,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-02T00:00:00.000Z",
				},
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
			},
		});

		// When
		await method.sync();

		// Then
		expect(existsSync(stagedArchive)).toBe(false);
		expect(readFileSync(join(minsyncWorkspace, "files", "docs", "notice.txt.md"), "utf8")).toBe(
			"Parsed new notice.\n",
		);
		expect(statSync(stagedPolicy).mtimeMs).toBe(preservedTime.getTime());
	});

	it("stages parsed mirror entries whose file names contain '#' or '?'", async () => {
		// Given: real documents whose names carry URL-looking characters.
		// `#` is legal on every host; `?` is reserved on Windows, so that half of
		// the case only exists where such a file can be created.
		const questionMarkHost = process.platform !== "win32";
		const receiptOutput = join(root, ".autorag", "parsed", "files", "docs", "receipt #2832-1476.txt.md");
		const queryOutput = join(root, ".autorag", "parsed", "files", "docs", "what ? query.txt.md");
		writeFileSync(join(source, "receipt #2832-1476.txt"), "raw receipt source\n");
		writeFileSync(receiptOutput, "Parsed receipt #2832-1476 for the July payout.\n");
		if (questionMarkHost) {
			writeFileSync(join(source, "what ? query.txt"), "raw query-marked source\n");
			writeFileSync(queryOutput, "Parsed query-marked note about payouts.\n");
		}
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				"/docs/receipt #2832-1476.txt": {
					virtualPath: "/docs/receipt #2832-1476.txt",
					sourcePath: join(source, "receipt #2832-1476.txt"),
					outputPath: receiptOutput,
					parserName: "plain-text",
					sourceMtimeNs: 1,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-01T00:00:00.000Z",
				},
				...(questionMarkHost
					? {
							"/docs/what ? query.txt": {
								virtualPath: "/docs/what ? query.txt",
								sourcePath: join(source, "what ? query.txt"),
								outputPath: queryOutput,
								parserName: "plain-text",
								sourceMtimeNs: 1,
								sourceSizeBytes: 25,
								updatedAt: "2026-01-01T00:00:00.000Z",
							},
						}
					: {}),
			},
		});
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		await method.sync();

		// Then: every parsed document reaches the MinSync staging mirror.
		expect(readFileSync(join(minsyncWorkspace, "files", "docs", "receipt #2832-1476.txt.md"), "utf8")).toBe(
			"Parsed receipt #2832-1476 for the July payout.\n",
		);
		if (questionMarkHost) {
			expect(readFileSync(join(minsyncWorkspace, "files", "docs", "what ? query.txt.md"), "utf8")).toBe(
				"Parsed query-marked note about payouts.\n",
			);
		}
	});

	it("reports staging exclusions through refresh diagnostics instead of dropping them silently", async () => {
		// Given: a file name that cannot become a canonical source id (a POSIX
		// backslash) — parseable, but not representable as a virtual id.
		// Windows reserves the backslash as a path separator, so such a file
		// name cannot exist there; the exclusion is POSIX-only by nature.
		if (process.platform === "win32") return;
		const trickyName = "policy\\note.txt";
		writeFileSync(join(source, trickyName), "raw policy source\n");
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { binaryPath: minsyncBinary, workspacePath: minsyncWorkspace },
		});

		// When
		const result = await agent.refresh(false, { methods: ["minsync"] });

		// Then: the excluded document is named in the refresh diagnostics.
		const excluded = result.diagnostics.filter((diagnostic) => diagnostic.code === "minsync-staging-excluded");
		expect(excluded.map((diagnostic) => diagnostic.source)).toEqual(["/docs/policy\\note.txt"]);
	});

	it("ignores traversal entries in a corrupt staging state", async () => {
		// Given
		const outsidePath = join(root, "outside.md");
		writeFileSync(outsidePath, "must survive\n");
		mkdirSync(minsyncWorkspace, { recursive: true });
		writeFileSync(
			join(root, ".autorag", ".minsync-autorag-staging.json"),
			`${JSON.stringify({
				version: 1,
				entries: {
					"/../../../outside": {
						outputPath: parsedOutput,
						updatedAt: "2026-01-01T00:00:00.000Z",
					},
				},
			})}\n`,
		);
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		await method.sync();

		// Then
		expect(readFileSync(outsidePath, "utf8")).toBe("must survive\n");
	});

	it("removes staged files missing from the persisted staging state", async () => {
		// Given
		const stagedPolicy = join(minsyncWorkspace, "files", "docs", "policy.txt.md");
		const orphanPath = join(minsyncWorkspace, "files", "docs", "orphan.txt.md");
		mkdirSync(join(minsyncWorkspace, "files", "docs"), { recursive: true });
		writeFileSync(stagedPolicy, "Parsed renewal policy with cancellation and refund terms.\n");
		writeFileSync(orphanPath, "orphan from interrupted staging\n");
		writeFileSync(
			join(root, ".autorag", ".minsync-autorag-staging.json"),
			`${JSON.stringify({
				version: 1,
				entries: {
					"/docs/policy.txt": {
						outputPath: parsedOutput,
						updatedAt: "2026-01-01T00:00:00.000Z",
					},
				},
			})}\n`,
		);
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		await method.sync();

		// Then
		expect(existsSync(orphanPath)).toBe(false);
		expect(readFileSync(stagedPolicy, "utf8")).toBe(readFileSync(parsedOutput, "utf8"));
	});

	it("rebuilds a symlinked staging root without touching its target", async () => {
		// Given
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});
		await method.sync();
		const filesRoot = join(minsyncWorkspace, "files");
		const outsideDirectory = join(root, "outside-staging-target");
		const outsideFile = join(outsideDirectory, "must-survive.txt");
		rmSync(filesRoot, { recursive: true, force: true });
		mkdirSync(outsideDirectory, { recursive: true });
		writeFileSync(outsideFile, "must survive\n");
		symlinkSync(outsideDirectory, filesRoot, "dir");

		// When
		await method.sync();

		// Then
		expect(readFileSync(outsideFile, "utf8")).toBe("must survive\n");
		expect(lstatSync(filesRoot).isSymbolicLink()).toBe(false);
		expect(lstatSync(filesRoot).isDirectory()).toBe(true);
		expect(readFileSync(join(filesRoot, "docs", "policy.txt.md"), "utf8")).toContain("cancellation terms");
	});

	it("replaces a staged file symlink without reading or overwriting its target", async () => {
		// Given
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});
		await method.sync();
		const stagedPolicy = join(minsyncWorkspace, "files", "docs", "policy.txt.md");
		const outsideFile = join(root, "outside-policy.md");
		rmSync(stagedPolicy, { force: true });
		writeFileSync(outsideFile, "must survive\n");
		symlinkSync(outsideFile, stagedPolicy);

		// When
		await method.sync();

		// Then
		expect(readFileSync(outsideFile, "utf8")).toBe("must survive\n");
		expect(lstatSync(stagedPolicy).isSymbolicLink()).toBe(false);
		expect(lstatSync(stagedPolicy).isFile()).toBe(true);
		expect(readFileSync(stagedPolicy, "utf8")).toContain("cancellation terms");
	});

	it("stamps the line range and enclosing article of a chunk found in the parsed mirror (#1692)", async () => {
		// Given
		writeFileSync(
			parsedOutput,
			[
				"# 이용약관",
				"",
				"제5조(회원의 의무)",
				"① 회원은 관계 법령을 준수하여야 한다.",
				"② 회원은 타인의 계정을 도용하여서는 아니 된다.",
				"",
			].join("\n"),
		);
		await seedMinSyncStage(root, {
			version: 1,
			entries: {
				"/docs/policy.txt": {
					virtualPath: "/docs/policy.txt",
					sourcePath: join(source, "policy.txt"),
					outputPath: parsedOutput,
					parserName: "plain-text",
					sourceMtimeNs: 2,
					sourceSizeBytes: 18,
					updatedAt: "2026-01-02T00:00:00.000Z",
				},
			},
		});
		writeFakeMinSync(
			JSON.stringify({
				results: [
					{ path: parsedOutput, score: 0.9, text: "② 회원은 타인의 계정을 도용하여서는 아니 된다." },
					{ path: parsedOutput, score: 0.8, text: "text the mirror does not contain" },
				],
			}),
		);
		const method = new MinSyncVectorMethod({ binaryPath: minsyncBinary, root, workspacePath: minsyncWorkspace });

		// When
		const results = await method.retrieve("계정 도용", { topK: 2 });

		// Then
		expect(results[0]?.metadata).toMatchObject({ lineNumber: 5, lineEnd: 5, section: "제5조(회원의 의무)" });
		expect(results[1]?.metadata).not.toHaveProperty("lineNumber");
		expect(results[1]?.metadata).not.toHaveProperty("section");
	});

	it("returns vector results with the canonical virtual source", async () => {
		// Given
		writeFakeMinSync(
			JSON.stringify({
				results: [
					{
						path: parsedOutput,
						score: 0.91,
						text: "Parsed renewal policy with cancellation terms.",
					},
				],
			}),
		);
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		const results = await method.retrieve("renewal cancellation", { topK: 2 });

		// Then
		expect(results).toHaveLength(1);
		const result = requireValue(results[0], "first vector result");
		expect(result.source).toBe(realpathSync(join(source, "policy.txt")));
		expect(result.content).toBe("Parsed renewal policy with cancellation terms.");
		expect(result.score).toBe(0.91);
		expect(result.metadata).toMatchObject({ method: "minsync", virtualPath: "/docs/policy.txt" });
		expect(loggedCalls()).toContainEqual(
			JSON.stringify({
				args: ["query", "--format", "json", "--mode", "vector", "-k", "2", "renewal cancellation"],
				cwd: minSyncCwd(),
			}),
		);
	});

	it("gives each chunk of one parsed mirror its own evidence id", async () => {
		// Several chunks of one document share MinSync's `path` (the parsed mirror).
		// Keying evidence on that path alone gave every passage the same id, so the
		// merger treated them as one duplicate and only the best chunk survived.
		// MinSync's per-chunk `doc_id` is what keeps them distinct.
		writeFakeMinSync(
			JSON.stringify({
				results: [
					{ path: parsedOutput, score: 0.93, text: "First passage.", doc_id: "chunk-aaa" },
					{ path: parsedOutput, score: 0.9, text: "Second passage.", doc_id: "chunk-bbb" },
					{ path: parsedOutput, score: 0.88, text: "Third passage.", doc_id: "chunk-ccc" },
				],
			}),
		);
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		const results = await method.retrieve("renewal cancellation", { topK: 5 });

		expect(results).toHaveLength(3);
		expect(new Set(results.map((result) => result.id)).size).toBe(3);
		// They remain one document: the source path is shared.
		expect(new Set(results.map((result) => result.source)).size).toBe(1);
		expect(results.map((result) => result.content)).toEqual(["First passage.", "Second passage.", "Third passage."]);
	});

	it("serializes parallel retrievals that share one MinSync workspace", async () => {
		// MinSync allows a single operation per workspace: a second concurrent
		// operation fails with "another sync is in progress". AutoRAG registers the
		// vector and hybrid methods, and the retriever runs them in parallel.
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { closeSync, mkdirSync, openSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";
if (process.argv[2] === "stage") {
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: process.argv[process.argv.indexOf("--state-dir") + 1] });
} else {

const lock = join(process.cwd(), ".minsync", "lock");
mkdirSync(dirname(lock), { recursive: true });
let fd;
try {
  fd = openSync(lock, "wx");
} catch (error) {
  if (error.code === "EEXIST") {
    console.error("another sync is in progress");
    process.exit(2);
  }
  throw error;
}
closeSync(fd);
await new Promise((resolve) => setTimeout(resolve, 120));
unlinkSync(lock);
console.log(JSON.stringify({ results: [{ path: ${JSON.stringify(parsedOutput)}, score: 0.9, text: "Parsed renewal policy with cancellation terms." }] }));
process.exit(0);
}
`,
		);
		chmodSync(minsyncBinary, 0o755);
		const vector = new MinSyncVectorMethod({ binaryPath: minsyncBinary, root, workspacePath: minsyncWorkspace });
		const hybrid = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			mode: "hybrid",
		});

		const [semantic, hybridResults] = await Promise.all([
			vector.retrieve("renewal cancellation", { topK: 2 }),
			hybrid.retrieve("renewal cancellation", { topK: 2 }),
		]);

		expect(semantic).toHaveLength(1);
		expect(hybridResults).toHaveLength(1);
	});

	it("answers a failed hybrid query from BM25 and tells the pipeline the semantic half was down", async () => {
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { createRequire } from "node:module";
const args = process.argv.slice(2);
if (args[0] === "stage") {
  createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: args[args.indexOf("--state-dir") + 1] });
} else {
if (args[args.indexOf("--mode") + 1] === "hybrid") {
  console.error("embedder unreachable at http://127.0.0.1:9");
  process.exit(2);
}
console.log(JSON.stringify({ results: [{ path: ${JSON.stringify(parsedOutput)}, score: 0.9, text: "BM25 answer." }] }));
}
`,
		);
		chmodSync(minsyncBinary, 0o755);
		const hybrid = new MinSyncHybridMethod({ binaryPath: minsyncBinary, root, workspacePath: minsyncWorkspace });
		const events: RetrievalFallbackEvent[] = [];

		const results = await hybrid.retrieve("renewal", { topK: 2, onFallback: (event) => events.push(event) });

		expect(results.map((result) => result.content)).toEqual(["BM25 answer."]);
		expect(events).toHaveLength(1);
		expect(events[0]).toMatchObject({ method: "hybrid", fallback: "bm25" });
		expect(events[0]?.reason).toContain("embedder unreachable at http://127.0.0.1:9");
	});

	it("fails fast instead of waiting behind a refresh sync", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }), false, 200);
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});
		const refresh = method.sync();

		await expect(method.retrieve("renewal cancellation", { topK: 1 })).rejects.toThrow(
			`another sync is in progress (${minsyncWorkspace})`,
		);
		await refresh;
	});

	it("routes lexical retrieval through MinSync BM25 mode", async () => {
		writeFakeMinSync(
			JSON.stringify({
				results: [
					{
						path: "files/docs/policy.txt.md",
						score: 0.88,
						text: "BM25 lexical hit from MinSync.",
					},
				],
			}),
		);
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			mode: "bm25",
		});

		const results = await method.retrieve("renewal cancellation", { topK: 2 });

		expect(results[0]?.metadata.method).toBe("minsync-bm25");
		expect(loggedCalls()).toContainEqual(
			JSON.stringify({
				args: ["query", "--format", "json", "--mode", "bm25", "-k", "2", "renewal cancellation"],
				cwd: minSyncCwd(),
			}),
		);
	});

	it("never auto-installs during retrieval; a missing binary fails with MinSyncRequiredError", async () => {
		// Given: no binary anywhere, and an installer that records any attempt.
		const originalPath = process.env.PATH;
		process.env.PATH = join(root, "empty-path");
		const installAttempts: string[] = [];
		const method = new MinSyncVectorMethod({
			root,
			workspacePath: minsyncWorkspace,
			installer: {
				cargoInstaller: async () => {
					installAttempts.push("cargo");
					throw new Error("mock cargo failure");
				},
				releaseProvider: async () => {
					installAttempts.push("release");
					throw new Error("mock install failure");
				},
			},
			autoInstall: true,
		});
		const engine = new RetrievalEngine();
		engine.register(method);

		try {
			// When / Then: the query fails loudly and never waits on an install.
			await expect(engine.retrieve("renewal cancellation")).rejects.toBeInstanceOf(MinSyncRequiredError);
			expect(installAttempts).toEqual([]);
		} finally {
			process.env.PATH = originalPath;
		}
	});

	it("maps real MinSync relative file paths to original source files", async () => {
		// Given
		writeFakeMinSync(
			JSON.stringify([
				{
					path: "files/docs/policy.txt.md",
					score: 0.77,
					text: "Relative path hit from MinSync.",
				},
			]),
		);
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		const results = await method.retrieve("relative path", { topK: 1 });

		// Then
		expect(results).toHaveLength(1);
		const result = requireValue(results[0], "relative path result");
		expect(result.source).toBe(realpathSync(join(source, "policy.txt")));
		expect(result.metadata.virtualPath).toBe("/docs/policy.txt");
		expect(result.content).toBe("Relative path hit from MinSync.");
	});

	it("fails retrieval with MinSyncRequiredError when the minsync binary is missing", async () => {
		// Given
		const method = new MinSyncVectorMethod({
			binaryPath: join(root, "missing-minsync"),
			root,
			workspacePath: minsyncWorkspace,
		});

		// When / Then: no empty result stands in for the missing binary.
		await expect(method.retrieve("renewal cancellation", { topK: 2 })).rejects.toBeInstanceOf(MinSyncRequiredError);
	});

	it("returns empty vector results when minsync query emits malformed JSON", async () => {
		// Given
		writeFakeMinSync("{not json");
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		const results = await method.retrieve("renewal cancellation", { topK: 2 });

		// Then
		expect(results).toEqual([]);
	});

	it("prefers cargo install over a GitHub release asset when no binary exists", async () => {
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		const order: string[] = [];
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "7350561268bb4e0b9e1621f8557f97e73b43e78e6a09fb2dada54cd413c0c971",
				},
			],
		};

		const resolved = await ensureMinSyncBinary({
			root,
			platform: "darwin",
			arch: "arm64",
			cargoInstaller: async (destination) => {
				order.push("cargo");
				mkdirSync(join(root, ".autorag", "bin"), { recursive: true });
				writeFakeMinSync(JSON.stringify({ results: [] }));
				copyFileSync(minsyncBinary, destination);
				chmodSync(destination, 0o755);
				return { binaryPath: destination, version: MINSYNC_VERSION };
			},
			releaseProvider: async () => {
				order.push("github");
				return release;
			},
			assetInstaller: async () => {
				order.push("github-asset");
				writeFileSync(installedBinary, "#!/usr/bin/env node\n");
				chmodSync(installedBinary, 0o755);
			},
		});

		expect(order).toEqual(["cargo"]);
		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: MINSYNC_VERSION });
	});

	it("does not compile minsync from cargo during NODE_ENV=test without an injected locator", async () => {
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		const order: string[] = [];
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "7350561268bb4e0b9e1621f8557f97e73b43e78e6a09fb2dada54cd413c0c971",
				},
			],
		};

		const resolved = await ensureMinSyncBinary({
			root,
			platform: "darwin",
			arch: "arm64",
			releaseProvider: async () => {
				order.push("github");
				return release;
			},
			assetInstaller: async (_asset, destination) => {
				order.push("github-asset");
				writeFakeMinSync(JSON.stringify({ results: [] }));
				copyFileSync(minsyncBinary, destination);
				chmodSync(destination, 0o755);
			},
		});

		expect(order).toEqual(["github", "github-asset"]);
		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
	});

	it("skips cargo install and uses GitHub when cargo is not on PATH", async () => {
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		const order: string[] = [];
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "7350561268bb4e0b9e1621f8557f97e73b43e78e6a09fb2dada54cd413c0c971",
				},
			],
		};

		const resolved = await ensureMinSyncBinary({
			root,
			platform: "darwin",
			arch: "arm64",
			cargoLocator: () => undefined,
			releaseProvider: async () => {
				order.push("github");
				return release;
			},
			assetInstaller: async (_asset, destination) => {
				order.push("github-asset");
				writeFakeMinSync(JSON.stringify({ results: [] }));
				copyFileSync(minsyncBinary, destination);
				chmodSync(destination, 0o755);
			},
		});

		expect(order).toEqual(["github", "github-asset"]);
		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
	});

	it("falls back to the GitHub release asset when cargo install is unavailable", async () => {
		// Given
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "7350561268bb4e0b9e1621f8557f97e73b43e78e6a09fb2dada54cd413c0c971",
				},
			],
		};

		// When
		const resolved = await ensureMinSyncBinary({
			root,
			platform: "darwin",
			arch: "arm64",
			cargoInstaller: async () => undefined,
			releaseProvider: async () => release,
			assetInstaller: async (asset, destination) => {
				expect(asset.name).toBe("minsync-v0.2.1-aarch64-apple-darwin.tar.gz");
				writeFakeMinSync(JSON.stringify({ results: [] }));
				copyFileSync(minsyncBinary, destination);
				chmodSync(destination, 0o755);
			},
		});

		// Then
		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
	});

	it("falls back to the GitHub release asset when cargo install throws", async () => {
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "7350561268bb4e0b9e1621f8557f97e73b43e78e6a09fb2dada54cd413c0c971",
				},
			],
		};

		const resolved = await ensureMinSyncBinary({
			root,
			platform: "darwin",
			arch: "arm64",
			cargoInstaller: async () => {
				throw new MinSyncReleaseError("cargo missing");
			},
			releaseProvider: async () => release,
			assetInstaller: async (_asset, destination) => {
				writeFakeMinSync(JSON.stringify({ results: [] }));
				copyFileSync(minsyncBinary, destination);
				chmodSync(destination, 0o755);
			},
		});

		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
	});

	it("rejects an incompatible downloaded binary without leaving it cached or opening a stage", async () => {
		const installedBinary = join(root, ".autorag", "bin", "minsync");
		rmSync(join(root, ".autorag", "parsed", "stage"), { recursive: true, force: true });
		await expect(
			ensureMinSyncBinary({
				root,
				platform: "darwin",
				arch: "arm64",
				cargoInstaller: async () => undefined,
				releaseProvider: async () => ({
					tagName: "v0.5.0",
					assets: [
						{
							name: "minsync-v0.5.0-aarch64-apple-darwin.tar.gz",
							downloadUrl: "https://example.test/minsync.tgz",
							sha256: "a".repeat(64),
						},
					],
				}),
				assetInstaller: async (_asset, destination) => {
					writeFileSync(
						destination,
						`#!/usr/bin/env node
process.stdin.once("data", () => console.log(JSON.stringify({ ok: false, op: "unknown", error: { code: "invalid_request", message: "unknown capabilities" } })));
`,
					);
				},
			}),
		).rejects.toMatchObject({ code: "unsupported_protocol" });
		expect(existsSync(installedBinary)).toBe(false);
		expect(existsSync(join(root, ".autorag", "parsed", "stage"))).toBe(false);
	});

	it("rejects release assets without a usable sha256 digest", async () => {
		// Given
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
				},
			],
		};

		// When / Then
		await expect(
			ensureMinSyncBinary({
				root,
				platform: "darwin",
				arch: "arm64",
				cargoInstaller: async () => undefined,
				releaseProvider: async () => release,
			}),
		).rejects.toThrow("sha256");
	});

	it("rejects release assets with malformed sha256 digests", async () => {
		// Given
		const release = {
			tagName: "v0.2.1",
			assets: [
				{
					name: "minsync-v0.2.1-aarch64-apple-darwin.tar.gz",
					downloadUrl: "https://example.test/minsync.tgz",
					sha256: "fixture-digest",
				},
			],
		};

		// When / Then
		await expect(
			ensureMinSyncBinary({
				root,
				platform: "darwin",
				arch: "arm64",
				cargoInstaller: async () => undefined,
				releaseProvider: async () => release,
			}),
		).rejects.toThrow("sha256");
	});
});

describe("MinSyncVectorMethod embedder plumbing", () => {
	it("passes --embedder <id> to init when embedder.id is set", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			embedder: { id: "openai:voyageai/voyage-4-large" },
		});

		const result = await method.sync();

		expect(result).toMatchObject({ synced: 1 });
		const initCall = loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[]; cwd: string })
			.find((call) => call.args[0] === "init");
		expect(initCall?.args).toContain("--embedder");
		const embedderIdx = initCall?.args.indexOf("--embedder");
		expect(initCall?.args[embedderIdx! + 1]).toBe("openai:voyageai/voyage-4-large");
	});

	it("does not pass --embedder when no embedder.id is set", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		await method.sync();

		const initCall = loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[]; cwd: string })
			.find((call) => call.args[0] === "init");
		expect(initCall?.args).not.toContain("--embedder");
	});

	it("fails with MinSyncRequiredError when no binary is available and autoInstall is false", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = "/nonexistent";
		try {
			const method = new MinSyncVectorMethod({
				binaryPath: join(root, "nonexistent-binary"),
				root,
				workspacePath: minsyncWorkspace,
				autoInstall: false,
			});

			await expect(method.sync()).rejects.toBeInstanceOf(MinSyncRequiredError);
		} finally {
			process.env.PATH = savedPath;
		}
	});

	it("auto-installs a verified release when no binary is available", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = savedPath
			?.split(delimiter)
			.filter((directory) => !existsSync(join(directory, process.platform === "win32" ? "minsync.exe" : "minsync")))
			.join(delimiter);
		writeFakeMinSync(JSON.stringify({ results: [] }));
		try {
			const method = new MinSyncVectorMethod({
				root,
				workspacePath: minsyncWorkspace,
				embedder: { id: "openai" },
				installer: {
					platform: "darwin",
					arch: "arm64",
					cargoInstaller: async () => undefined,
					releaseProvider: async () => ({
						tagName: "v0.3.0",
						assets: [
							{
								name: "minsync-v0.3.0-aarch64-apple-darwin.tar.gz",
								downloadUrl: "https://example.test/minsync.tar.gz",
								sha256: "a".repeat(64),
							},
						],
					}),
					assetInstaller: async (_asset, destination) => {
						copyFileSync(minsyncBinary, destination);
					},
				},
			});

			const result = await method.sync();

			expect(result).toMatchObject({ ok: true, synced: 1 });
			expect(existsSync(join(root, ".autorag", "bin", "minsync"))).toBe(true);
		} finally {
			process.env.PATH = savedPath;
		}
	});

	it("degrades with missing-api-key-env when apiKeyEnv is set but env var is empty", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			embedder: { apiKeyEnv: "MINSYNC_TEST_MISSING_KEY" },
		});

		const result = await method.sync();

		expect(result).toMatchObject({ ok: false, synced: 0 });
		expect(result.reason).toContain("missing-api-key-env");
		expect(result.reason).toContain("MINSYNC_TEST_MISSING_KEY");
	});

	it("proceeds with sync when apiKeyEnv is set and env var has a value", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		process.env.MINSYNC_TEST_PRESENT_KEY = "test-key-value";
		try {
			const method = new MinSyncVectorMethod({
				binaryPath: minsyncBinary,
				root,
				workspacePath: minsyncWorkspace,
				embedder: { apiKeyEnv: "MINSYNC_TEST_PRESENT_KEY" },
			});

			const result = await method.sync();

			expect(result).toMatchObject({ ok: true, synced: 1 });
		} finally {
			delete process.env.MINSYNC_TEST_PRESENT_KEY;
		}
	});

	it("projects init stderr to a fixed path-opaque reason", async () => {
		// Fake binary that emits a secret-looking string on stderr for init.
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
const args = process.argv.slice(2);
if (args[0] === "init") {
  console.error("auth failed for key sk-abc123def456 in region us-east-1");
  process.exit(1);
}
process.exit(2);
`,
		);
		chmodSync(minsyncBinary, 0o755);

		const client = new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
		});

		const result = await client.sync();

		expect(result.ok).toBe(false);
		expect(result.reason).not.toContain("sk-abc123def456");
		expect(result.reason).toBe("init-failed");
	});

	it("materializes the managed CLI boundary before native commands", async () => {
		const workspace = mkdtempSync(join(tmpdir(), "autorag-minsync-managed-"));
		const binary = join(workspace, "minsync");
		const callsPath = join(workspace, "managed-calls.jsonl");
		writeFileSync(
			binary,
			`#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(callsPath)}, JSON.stringify({ cwd: process.cwd(), args }) + "\\n");
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
if (args[0] === "init") { mkdirSync(dirname(config), { recursive: true }); writeFileSync(config, "[embedder]\\n"); }
if (args[0] === "check") process.stdout.write('{"embedder_ok":true,"vectorstore_ok":true}');
if (args[0] === "sync") { mkdirSync(dirname(cursor), { recursive: true }); writeFileSync(cursor, "{}"); process.stdout.write('{"synced":1}'); }
`,
		);
		chmodSync(binary, 0o755);

		const result = await new MinSyncClient({ binaryPath: binary, workspacePath: workspace }).sync();

		expect(result).toMatchObject({ ok: true, synced: 1 });
		const calls = readFileSync(callsPath, "utf8")
			.trim()
			.split("\n")
			.map((line) => JSON.parse(line) as { cwd: string; args: string[] });
		expect(calls.every((call) => call.cwd === realpathSync(workspace))).toBe(true);
		expect(calls.map((call) => call.args[0])).toEqual(["init", "check", "sync"]);
		rmSync(workspace, { recursive: true, force: true });
	});

	it("requires init to materialize config.toml before reading it", async () => {
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
if (args[0] === "check") process.stdout.write('{"embedder_ok":true,"vectorstore_ok":true}');
if (args[0] === "sync") process.stdout.write('{"synced":1}');
`,
		);
		chmodSync(minsyncBinary, 0o755);
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: { id: "test" },
		}).sync();

		expect(result).toMatchObject({ ok: false, reason: "init-failed" });
		expect(loggedCalls().map((line) => JSON.parse(line).args[0])).toEqual(["init"]);
	});

	it("rewrites allowlisted embedder fields into .minsync/config.toml after init", async () => {
		// Create a minimal config.toml that init would have produced
		const minsyncConfigDir = join(minsyncWorkspace, ".minsync");
		mkdirSync(minsyncConfigDir, { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			`[embedder]
id = "openai:voyageai/voyage-4-lite"
base_url = "https://api.openai.com/v1"

[vectorstore]
[vectorstore.options]
dimension = 1024
`,
		);

		writeFakeMinSync(JSON.stringify({ results: [] }));

		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			embedder: {
				id: "openai:voyageai/voyage-4-large",
				baseUrl: "https://embed.example.com/v1",
				dimension: 2048,
				queryPrefix: "query:",
				passagePrefix: "passage:",
				batchSize: 64,
				maxRetries: 5,
				maxConcurrent: 4,
				timeoutMs: 30_000,
			},
			maxChunkSize: 1000,
		});

		const result = await method.sync();

		expect(result).toMatchObject({ ok: true, synced: 1 });

		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(rewritten.embedder?.id).toBe("openai:voyageai/voyage-4-large");
		expect(rewritten.embedder?.base_url).toBe("https://embed.example.com/v1");
		expect(rewritten.embedder?.query_prefix).toBe("query:");
		expect(rewritten.embedder?.passage_prefix).toBe("passage:");
		expect(rewritten.embedder?.batch_size).toBe(64);
		expect(rewritten.embedder?.max_retries).toBe(5);
		expect(rewritten.embedder?.max_concurrent).toBe(4);
		expect(rewritten.embedder?.timeout_seconds).toBe(30);
		expect((rewritten.vectorstore?.options as { dimension?: number } | undefined)?.dimension).toBe(2048);
		expect((rewritten.chunker?.options as { max_chunk_size?: number } | undefined)?.max_chunk_size).toBe(1000);
	});

	it("forces a full sync when the configured chunk size changes", async () => {
		const minsyncConfigDir = join(minsyncWorkspace, ".minsync");
		mkdirSync(minsyncConfigDir, { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			`[chunker.options]
max_chunk_size = 4096
`,
		);
		writeFileSync(join(minsyncConfigDir, "cursor.json"), "{}");
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			maxChunkSize: 1000,
		});

		await method.sync();

		const syncCall = loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[] })
			.find((call) => call.args[0] === "sync");
		expect(syncCall?.args).toEqual(["sync", "--full", "--format", "json"]);
	});

	it("restores the previous config when a forced full sync fails", async () => {
		const minsyncConfigDir = join(minsyncWorkspace, ".minsync");
		const originalConfig = "[chunker.options]\nmax_chunk_size = 4096\n";
		mkdirSync(minsyncConfigDir, { recursive: true });
		writeFileSync(minSyncConfigPath(minsyncWorkspace), originalConfig);
		writeFileSync(join(minsyncConfigDir, "cursor.json"), "{}");
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { appendFileSync, existsSync } from "node:fs";
import { join } from "node:path";
const args = process.argv.slice(2);
if (args[0] === "stage") {
  (await import("node:module")).createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: args[args.indexOf("--state-dir") + 1] });
} else {
const cursor = join(process.cwd(), ".minsync", "cursor.json");
const entry = { args, ...(args[0] === "sync" ? { cursorExists: existsSync(cursor) } : {}) };
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify(entry) + "\\n");
if (args[0] === "check") process.stdout.write('{"embedder_ok":true,"vectorstore_ok":true}');
if (args[0] === "sync") process.exit(1);
}
`,
		);
		chmodSync(minsyncBinary, 0o755);

		const result = await new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			maxChunkSize: 1000,
		}).sync();

		expect(result).toMatchObject({ ok: false, reason: "sync-failed" });
		expect(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")).toBe(originalConfig);
		expect(existsSync(join(minsyncConfigDir, "cursor.json"))).toBe(true);
		const syncCall = loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[]; cursorExists?: boolean })
			.find((call) => call.args[0] === "sync");
		expect(syncCall?.cursorExists).toBe(true);
	});

	it("uses the MinSync chunk size for lexical indexing", async () => {
		const minsyncConfigDir = join(minsyncWorkspace, ".minsync");
		mkdirSync(minsyncConfigDir, { recursive: true });
		writeFileSync(minSyncConfigPath(minsyncWorkspace), "[chunker.options]\n");
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			maxChunkSize: 1000,
			mode: "bm25",
		});

		await method.sync();

		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect((rewritten.chunker?.options as { max_chunk_size?: number } | undefined)?.max_chunk_size).toBe(1000);
	});

	it("defaults to in-process native Qwen3 embeddings (1024d) without starting the gateway", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
		});

		const result = await method.sync();
		expect(result.ok).toBe(true);

		const identity = JSON.parse(
			readFileSync(join(minsyncWorkspace, ".minsync", "autorag-embedding-identity.json"), "utf8"),
		);
		expect(identity).toMatchObject({
			provider: "native",
			model: "Qwen/Qwen3-Embedding-0.6B",
			dimension: 1024,
			runtimeBuild: "minsync-native",
		});
	});

	it("throws MinSyncRequiredError on a missing binary during sync instead of returning ok:false", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = "/nonexistent";
		try {
			const method = new MinSyncVectorMethod({
				binaryPath: join(root, "nonexistent"),
				root,
				workspacePath: minsyncWorkspace,
				autoInstall: false,
			});

			await expect(method.sync()).rejects.toBeInstanceOf(MinSyncRequiredError);
		} finally {
			process.env.PATH = savedPath;
		}
	});
});

describe("MinSync embedder selection", () => {
	it("defaults to MinSync's own native Qwen3 and never writes an AutoRAG-owned endpoint", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace }).sync();
		expect(result.ok).toBe(true);
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder).toMatchObject({ id: "native:Qwen/Qwen3-Embedding-0.6B" });
		expect(config.embedder?.base_url).toBeUndefined();
		expect(config.vectorstore?.options).toMatchObject({ dimension: 1024 });
	});

	it("passes a configured remote endpoint through to MinSync untouched", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: {
				id: "openai:text-embedding-3-small",
				baseUrl: "https://router.example/api",
				apiKeyEnv: "REMOTE_EMBED_KEY",
				dimension: 1536,
			},
		}).sync();
		// The key variable is named, not required to exist for this assertion: the fake binary never calls out.
		expect(result).toMatchObject({ ok: false, reason: "missing-api-key-env:REMOTE_EMBED_KEY" });
		process.env.REMOTE_EMBED_KEY = "sk-test";
		try {
			const ok = await new MinSyncClient({
				binaryPath: minsyncBinary,
				workspacePath: minsyncWorkspace,
				embedder: {
					id: "openai:text-embedding-3-small",
					baseUrl: "https://router.example/api",
					apiKeyEnv: "REMOTE_EMBED_KEY",
					dimension: 1536,
				},
			}).sync();
			expect(ok.ok).toBe(true);
		} finally {
			delete process.env.REMOTE_EMBED_KEY;
		}
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder).toMatchObject({
			id: "openai:text-embedding-3-small",
			base_url: "https://router.example/api",
		});
		expect(config.vectorstore?.options).toMatchObject({ dimension: 1536 });
	});

	it("keeps an operator endpoint without an id instead of forcing native embeddings", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			`[embedder]\nid = "tei:google/embeddinggemma-300m"\nbase_url = "http://127.0.0.1:18080"\n\n[vectorstore.options]\ndimension = 768\n`,
		);
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: { baseUrl: "http://127.0.0.1:18080", dimension: 768 },
		}).sync();
		expect(result.ok).toBe(true);
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder?.base_url).toBe("http://127.0.0.1:18080");
		expect(config.embedder?.id).toBe("tei:google/embeddinggemma-300m");
	});

	it("gives the minsync child the key the operator's apiKeyEnv names, because MinSync reads OPENAI_API_KEY only", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		process.env.OPERATOR_EMBED_KEY = "sk-operator";
		const savedOpenAi = process.env.OPENAI_API_KEY;
		delete process.env.OPENAI_API_KEY;
		try {
			const embedder = {
				id: "openai:text-embedding-3-small",
				baseUrl: "https://router.example/api",
				apiKeyEnv: "OPERATOR_EMBED_KEY",
				dimension: 1536,
			};
			const client = new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace, embedder });
			expect((await client.sync()).ok).toBe(true);
			await client.query("q", 3);
			const calls = loggedEnv();
			for (const command of ["check", "sync", "query"]) {
				expect(calls.find((call) => call.command === command)?.openaiKey, command).toBe("sk-operator");
			}
		} finally {
			delete process.env.OPERATOR_EMBED_KEY;
			if (savedOpenAi !== undefined) process.env.OPENAI_API_KEY = savedOpenAi;
		}
	});

	it("does not touch OPENAI_API_KEY for a local native embedder", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const savedOpenAi = process.env.OPENAI_API_KEY;
		delete process.env.OPENAI_API_KEY;
		try {
			const client = new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace });
			await client.sync();
			expect(loggedEnv().find((call) => call.command === "sync")?.openaiKey).toBeNull();
		} finally {
			if (savedOpenAi !== undefined) process.env.OPENAI_API_KEY = savedOpenAi;
		}
	});

	it("keeps operator batching settings next to a remote endpoint", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: {
				id: "tei:BAAI/bge-m3",
				baseUrl: "http://gpu-box:8080",
				dimension: 1024,
				batchSize: 32,
				maxRetries: 5,
				maxConcurrent: 2,
				timeoutMs: 120_000,
			},
		}).sync();
		expect(result.ok).toBe(true);
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder).toMatchObject({
			id: "tei:BAAI/bge-m3",
			base_url: "http://gpu-box:8080",
			batch_size: 32,
			max_retries: 5,
			max_concurrent: 2,
			timeout_seconds: 120,
		});
	});

	it("lets a long sync outlive a small embedder timeout instead of killing the process", async () => {
		// The embedder timeoutMs is a per-batch embedding HTTP timeout (written to
		// config.toml as timeout_seconds). It must not SIGTERM `minsync sync`:
		// a full reindex can legitimately run far longer than any per-request
		// timeout, so an operator tuning batches to fail fast must not kill the
		// sync process mid-run.
		writeFakeMinSync(JSON.stringify({ results: [] }), false, 400);
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			embedder: { id: "openai", timeoutMs: 150 },
		}).sync();
		expect(result).toMatchObject({ ok: true, synced: 1 });
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder).toMatchObject({ id: "openai", timeout_seconds: 1 });
	});

	it("forces full reindex for stale identity without deleting the prior cursor", async () => {
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(minSyncConfigPath(minsyncWorkspace), "[vectorstore.options]\ndimension = 1024\n");
		writeFileSync(join(minsyncWorkspace, ".minsync", "cursor.json"), "{}");
		writeFileSync(
			join(minsyncWorkspace, ".minsync", "autorag-embedding-identity.json"),
			JSON.stringify({ provider: "old" }),
		);
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace }).sync();
		const syncCall = loggedCalls()
			.map((line) => JSON.parse(line))
			.find((call) => call.args[0] === "sync");
		expect(syncCall.args).toEqual(["sync", "--full", "--format", "json"]);
		expect(existsSync(join(minsyncWorkspace, ".minsync", "cursor.json"))).toBe(true);
	});
});

describe("Qwen3 query instruction in MinSync config", () => {
	const QWEN3_QUERY_PREFIX =
		"Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:";

	type Toml = Record<string, Record<string, unknown>>;
	const readConfig = (): Toml => parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Toml;
	const readIdentity = () =>
		JSON.parse(readFileSync(join(minsyncWorkspace, ".minsync", "autorag-embedding-identity.json"), "utf8"));
	const syncCalls = () =>
		loggedCalls()
			.map((line) => JSON.parse(line))
			.filter((call) => call.args[0] === "sync");
	const client = (embedder?: MinSyncEmbedderConfig) =>
		new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			...(embedder === undefined ? {} : { embedder }),
		});

	it("defaults the native Qwen3 embedder to the model card's query instruction", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 }).sync();
		expect(result.ok).toBe(true);
		expect(readConfig().embedder?.query_prefix).toBe(QWEN3_QUERY_PREFIX);
		expect(readConfig().embedder?.passage_prefix).toBeUndefined();
		expect(readIdentity().queryPrefix).toBe(QWEN3_QUERY_PREFIX);
	});

	it("defaults to the instruction when no embedder is configured at all", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await client().sync();
		expect(readConfig().embedder?.query_prefix).toBe(QWEN3_QUERY_PREFIX);
	});

	it("lets an operator-set empty queryPrefix win", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024, queryPrefix: "" }).sync();
		expect(readConfig().embedder?.query_prefix).toBe("");
		expect(readIdentity().queryPrefix).toBe("");
	});

	it("lets an operator-set custom queryPrefix win", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", queryPrefix: "Instruct: custom\nQuery:" }).sync();
		expect(readConfig().embedder?.query_prefix).toBe("Instruct: custom\nQuery:");
	});

	it("does not add the Qwen3 instruction to another native model", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await client({ id: "native:BAAI/bge-small-en-v1.5", dimension: 384 }).sync();
		expect(readConfig().embedder?.query_prefix).toBeUndefined();
		expect(readIdentity().queryPrefix).toBe("");
	});

	it("does not re-embed stored documents when only the query prefix changes", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		// Index built before the instruction existed: empty query prefix.
		await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024, queryPrefix: "" }).sync();
		expect(syncCalls().at(-1).args).toEqual(["sync", "--full", "--format", "json"]);
		// Same documents, new default query prefix: incremental, not --full.
		const result = await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 }).sync();
		expect(result.ok).toBe(true);
		expect(syncCalls().at(-1).args).toEqual(["sync", "--format", "json"]);
		expect(readIdentity().queryPrefix).toBe(QWEN3_QUERY_PREFIX);
		expect(result.diagnostic).toBeUndefined();
	});

	it("still re-embeds documents when the passage prefix changes", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024, passagePrefix: "" }).sync();
		await client({ id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024, passagePrefix: "passage: " }).sync();
		expect(syncCalls().at(-1).args).toEqual(["sync", "--full", "--format", "json"]);
	});
});

describe("rewriteEmbedderConfig", () => {
	it("returns false when config.toml does not exist", () => {
		expect(rewriteEmbedderConfig(minsyncWorkspace, { id: "test-embedder" })).toBe(false);
	});

	it("only writes fields present on the embedder config", () => {
		const minsyncConfigDir = join(minsyncWorkspace, ".minsync");
		mkdirSync(minsyncConfigDir, { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			`[embedder]
id = "old-id"
base_url = "https://old.example.com"

[vectorstore]
[vectorstore.options]
dimension = 1536
`,
		);

		rewriteEmbedderConfig(minsyncWorkspace, { id: "new-id" });

		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(rewritten.embedder?.id).toBe("new-id");
		expect(rewritten.embedder?.base_url).toBe("https://old.example.com");
		expect((rewritten.vectorstore?.options as { dimension?: number } | undefined)?.dimension).toBe(1536);
	});

	it("strips base_url and prefixes when switching to native embedding", () => {
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			`[embedder]
id = "tei:google/embeddinggemma-300m"
base_url = "http://127.0.0.1:8080"
query_prefix = "task: search result | query: "
passage_prefix = "title: none | text: "

[vectorstore.options]
dimension = 768
`,
		);

		rewriteEmbedderConfig(minsyncWorkspace, { id: "native:Qwen/Qwen3-Embedding-0.6B", dimension: 1024 });

		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(rewritten.embedder?.id).toBe("native:Qwen/Qwen3-Embedding-0.6B");
		expect(rewritten.embedder?.base_url).toBeUndefined();
		expect(rewritten.embedder?.query_prefix).toBeUndefined();
		expect(rewritten.embedder?.passage_prefix).toBeUndefined();
		expect((rewritten.vectorstore?.options as { dimension?: number } | undefined)?.dimension).toBe(1024);
	});
});

describe("MinSync lexical (BM25) tokenizer", () => {
	function callArgs(command: string): readonly string[] | undefined {
		return loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[] })
			.find((call) => call.args[0] === command)?.args;
	}

	function languageFlag(args: readonly string[] | undefined): string | undefined {
		const index = args?.indexOf("--language") ?? -1;
		return index === -1 ? undefined : args?.[index + 1];
	}

	/** An initialized and already-synced workspace, as an earlier refresh leaves it. */
	function writeWorkspaceConfig(toml: string): void {
		const dir = join(minsyncWorkspace, ".minsync");
		mkdirSync(dir, { recursive: true });
		writeFileSync(minSyncConfigPath(minsyncWorkspace), toml);
		writeFileSync(join(dir, "cursor.json"), "{}");
	}

	function configuredLanguage(): unknown {
		const parsed = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		return parsed.lexical?.language;
	}

	function method(): MinSyncVectorMethod {
		return new MinSyncVectorMethod({ binaryPath: minsyncBinary, root, workspacePath: minsyncWorkspace });
	}

	function agentWith(languages?: readonly ("ko" | "en" | "ja")[]): AutoRAGAgent {
		return new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			...(languages === undefined ? {} : { languages }),
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { binaryPath: minsyncBinary, workspacePath: minsyncWorkspace },
		});
	}

	afterEach(() => {
		vi.unstubAllEnvs();
	});

	it("initializes MinSync with the multilingual tokenizer", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const result = await method().sync();

		expect(languageFlag(callArgs("init"))).toBe("multilingual");
		expect(configuredLanguage()).toBe("multilingual");
		expect(result.lexicalFallback).toBeUndefined();
	});

	it("uses the multilingual tokenizer whatever languages the agent is configured with", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		await agentWith().refresh(false, { methods: ["minsync"] });
		expect(languageFlag(callArgs("init"))).toBe("multilingual");

		rmSync(join(minsyncWorkspace, ".minsync"), { recursive: true, force: true });
		rmSync(logPath);
		await agentWith(["en"]).refresh(false, { methods: ["minsync"] });
		expect(languageFlag(callArgs("init"))).toBe("multilingual");
	});

	it("falls back to whitespace tokenization when the Kiwi library the tokenizer needs is missing", async () => {
		vi.stubEnv("KIWI_LIBRARY_PATH", join(root, "no-such-libkiwi"));
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const result = await method().sync();

		expect(result).toMatchObject({ ok: true, lexicalFallback: { requested: "multilingual", used: "simple" } });
		expect(languageFlag(callArgs("init"))).toBe("simple");
		expect(configuredLanguage()).toBe("simple");
	});

	it("keeps an index already built with the multilingual tokenizer instead of rebuilding it on a missing Kiwi", async () => {
		vi.stubEnv("KIWI_LIBRARY_PATH", join(root, "no-such-libkiwi"));
		writeWorkspaceConfig('[lexical]\nlanguage = "multilingual"\n');
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const result = await method().sync();

		expect(configuredLanguage()).toBe("multilingual");
		expect(result.lexicalFallback).toBeUndefined();
	});

	it("reports the fallback as a refresh warning naming what to install", async () => {
		vi.stubEnv("KIWI_LIBRARY_PATH", join(root, "no-such-libkiwi"));
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const refresh = await agentWith().refresh(false, { methods: ["minsync"] });

		const warning = refresh.diagnostics.find((diagnostic) => diagnostic.code === "lexical-tokenizer-unavailable");
		expect(warning).toMatchObject({ severity: "warning", source: "minsync" });
		expect(warning?.message).toContain("KIWI_LIBRARY_PATH");
	});

	it("points an existing workspace at the multilingual tokenizer without re-initializing it", async () => {
		writeWorkspaceConfig('[lexical]\nlanguage = "simple"\n');
		writeFakeMinSync(JSON.stringify({ results: [] }));

		const result = await method().sync();

		expect(result).toMatchObject({ ok: true });
		expect(callArgs("init")).toBeUndefined();
		expect(callArgs("sync")).toBeDefined();
		expect(configuredLanguage()).toBe("multilingual");
	});

	it("adds the tokenizer to a config written before MinSync had a lexical section", async () => {
		writeWorkspaceConfig("[chunker.options]\nmax_chunk_size = 4096\n");
		writeFakeMinSync(JSON.stringify({ results: [] }));

		await method().sync();

		expect(configuredLanguage()).toBe("multilingual");
	});

	it("restores the previous tokenizer when the rebuild fails", async () => {
		writeWorkspaceConfig('[lexical]\nlanguage = "simple"\n');
		const original = readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8");
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { appendFileSync } from "node:fs";
const args = process.argv.slice(2);
if (args[0] === "stage") {
  (await import("node:module")).createRequire(import.meta.url)(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir: args[args.indexOf("--state-dir") + 1] });
} else {
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args }) + "\\n");
if (args[0] === "check") process.stdout.write('{"embedder_ok":true,"vectorstore_ok":true}');
if (args[0] === "sync") process.exit(1);
}
`,
		);
		chmodSync(minsyncBinary, 0o755);

		const result = await method().sync();

		expect(result).toMatchObject({ ok: false, reason: "sync-failed" });
		expect(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")).toBe(original);
	});

	it("rewriteEmbedderConfig writes only the lexical language when asked", () => {
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(
			minSyncConfigPath(minsyncWorkspace),
			'[embedder]\nid = "old-id"\n\n[lexical]\nlanguage = "simple"\n',
		);

		rewriteEmbedderConfig(minsyncWorkspace, {}, { lexicalLanguage: "multilingual" });

		const rewritten = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(rewritten.lexical?.language).toBe("multilingual");
		expect(rewritten.embedder?.id).toBe("old-id");
	});
});
