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
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	ensureLocalEmbedder,
	ensureMinSyncBinary,
	MINSYNC_VERSION,
	MinSyncClient,
	MinSyncQueryError,
	MinSyncReleaseError,
	MinSyncVectorMethod,
	minSyncConfigPath,
	rewriteEmbedderConfig,
} from "../../src/minsync/index.ts";
import { saveMirrorIndex } from "../../src/mirror/index.ts";
import { RetrievalEngine } from "../../src/retrieval/engine.ts";

let root: string;
let source: string;
let parsedOutput: string;
let minsyncBinary: string;
let minsyncWorkspace: string;
let logPath: string;

beforeEach(() => {
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
	saveMirrorIndex(root, {
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

function writeFakeMinSync(queryJson: string, strictQuery = false): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, cwd: process.cwd() }) + "\\n");

if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}

if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}

if (args[0] === "sync") {
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
	it("uses the official v0.4.2 query command with its selected mode", async () => {
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
		expect(loggedCalls()).toContainEqual(JSON.stringify({ args: ["init", "--format", "json"], cwd: minSyncCwd() }));
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
		saveMirrorIndex(root, {
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
		saveMirrorIndex(root, {
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
		saveMirrorIndex(root, {
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
		saveMirrorIndex(root, {
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
		saveMirrorIndex(root, {
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
		expect(readFileSync(stagedPolicy, "utf8")).toContain("cancellation and refund terms");
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

	it("serializes parallel retrievals that share one MinSync workspace", async () => {
		// MinSync allows a single operation per workspace: a second concurrent
		// operation fails with "another sync is in progress". AutoRAG registers the
		// vector and hybrid methods, and the retriever runs them in parallel.
		writeFileSync(
			minsyncBinary,
			`#!/usr/bin/env node
import { closeSync, mkdirSync, openSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";

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

	it("exposes an install-failed diagnostic through retrieval after auto-install fails", async () => {
		// Given
		const originalPath = process.env.PATH;
		process.env.PATH = join(root, "empty-path");
		const method = new MinSyncVectorMethod({
			root,
			workspacePath: minsyncWorkspace,
			installer: {
				cargoInstaller: async () => {
					throw new Error("mock cargo failure");
				},
				releaseProvider: async () => {
					throw new Error("mock install failure");
				},
			},
			autoInstall: true,
		});
		const engine = new RetrievalEngine({ isMinSyncBinaryMissing: () => method.isBinaryMissing() });
		engine.register(method);

		try {
			// When
			const { results, diagnostics } = await engine.retrieve("renewal cancellation");

			// Then
			expect(results).toEqual([]);
			expect(diagnostics).toHaveLength(1);
			expect(diagnostics[0]).toMatchObject({
				code: "minsync-unavailable",
				severity: "warning",
				source: "minsync",
			});
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

	it("returns empty vector results when the minsync binary is missing", async () => {
		// Given
		const method = new MinSyncVectorMethod({
			binaryPath: join(root, "missing-minsync"),
			root,
			workspacePath: minsyncWorkspace,
		});

		// When
		const results = await method.retrieve("renewal cancellation", { topK: 2 });

		// Then
		expect(results).toEqual([]);
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
				writeFileSync(destination, "#!/usr/bin/env cargo-minsync\n");
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
		expect(readFileSync(installedBinary, "utf8")).toContain("cargo-minsync");
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
				writeFileSync(destination, "#!/usr/bin/env node\n");
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
				writeFileSync(destination, "#!/usr/bin/env node\n");
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
				writeFileSync(destination, "#!/usr/bin/env node\n");
				chmodSync(destination, 0o755);
			},
		});

		// Then
		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
		expect(readFileSync(installedBinary, "utf8")).toContain("node");
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
				writeFileSync(destination, "#!/usr/bin/env node\n");
				chmodSync(destination, 0o755);
			},
		});

		expect(resolved).toMatchObject({ binaryPath: installedBinary, version: "v0.2.1" });
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
			embedder: { id: "openai:text-embedding-3-large" },
		});

		const result = await method.sync();

		expect(result).toMatchObject({ synced: 1 });
		const initCall = loggedCalls()
			.map((line) => JSON.parse(line) as { args: string[]; cwd: string })
			.find((call) => call.args[0] === "init");
		expect(initCall?.args).toContain("--embedder");
		const embedderIdx = initCall?.args.indexOf("--embedder");
		expect(initCall?.args[embedderIdx! + 1]).toBe("openai:text-embedding-3-large");
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

	it("degrades with missing-binary when no binary is available and autoInstall is false", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = "/nonexistent";
		try {
			const method = new MinSyncVectorMethod({
				binaryPath: join(root, "nonexistent-binary"),
				root,
				workspacePath: minsyncWorkspace,
				autoInstall: false,
			});

			const result = await method.sync();

			expect(result).toMatchObject({ ok: false, synced: 0, reason: "missing-binary" });
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
id = "openai:text-embedding-3-small"
base_url = "https://api.openai.com/v1"

[vectorstore]
[vectorstore.options]
dimension = 1536
`,
		);

		writeFakeMinSync(JSON.stringify({ results: [] }));

		const method = new MinSyncVectorMethod({
			binaryPath: minsyncBinary,
			root,
			workspacePath: minsyncWorkspace,
			embedder: {
				id: "openai:text-embedding-3-large",
				baseUrl: "https://embed.example.com/v1",
				dimension: 3072,
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
		expect(rewritten.embedder?.id).toBe("openai:text-embedding-3-large");
		expect(rewritten.embedder?.base_url).toBe("https://embed.example.com/v1");
		expect(rewritten.embedder?.query_prefix).toBe("query:");
		expect(rewritten.embedder?.passage_prefix).toBe("passage:");
		expect(rewritten.embedder?.batch_size).toBe(64);
		expect(rewritten.embedder?.max_retries).toBe(5);
		expect(rewritten.embedder?.max_concurrent).toBe(4);
		expect(rewritten.embedder?.timeout_seconds).toBe(30);
		expect((rewritten.vectorstore?.options as { dimension?: number } | undefined)?.dimension).toBe(3072);
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
const cursor = join(process.cwd(), ".minsync", "cursor.json");
const entry = { args, ...(args[0] === "sync" ? { cursorExists: existsSync(cursor) } : {}) };
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify(entry) + "\\n");
if (args[0] === "check") process.stdout.write('{"embedder_ok":true,"vectorstore_ok":true}');
if (args[0] === "sync") process.exit(1);
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

	it("does not throw on missing binary during sync; returns ok:false degrade result", async () => {
		const savedPath = process.env.PATH;
		process.env.PATH = "/nonexistent";
		try {
			const method = new MinSyncVectorMethod({
				binaryPath: join(root, "nonexistent"),
				root,
				workspacePath: minsyncWorkspace,
				autoInstall: false,
			});

			const result = await method.sync();

			expect(result.ok).toBe(false);
			expect(result.reason).toBe("missing-binary");
		} finally {
			process.env.PATH = savedPath;
		}
	});
});

describe("AutoRAG embedding runtime integration", () => {
	it("degrades bootstrap failures to the typed embedder-unavailable diagnostic", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			runtime: {
				ensureRuntime: async () => {
					throw new Error("runtime bootstrap failed");
				},
			},
		}).sync();
		expect(result.ok).toBe(false);
		expect(result.diagnostic).toMatchObject({ code: "embedder-unavailable", retryable: true });
	});

	const runtime = {
		ensureRuntime: async () => ({
			baseUrl: "http://127.0.0.1:43123",
			profile: {
				profileId: "qwen3-embedding-0.6b" as const,
				provider: "qwen",
				model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
				dimension: 1024,
				queryPrefix: "",
				passagePrefix: "",
				runtimeBuild: "b10951",
				modelRevision: "rev",
				artifactSha256: "hash",
				backend: "auto" as const,
				protocolCapabilities: { healthz: true, embed: true, openaiCompatible: true, ollamaCompatible: true },
			},
			identity: {
				profileId: "qwen3-embedding-0.6b" as const,
				provider: "qwen",
				model: "Qwen3-Embedding-0.6B-Q8_0.gguf",
				dimension: 1024,
				runtimeBuild: "b10951",
				modelRevision: "rev",
				artifactSha256: "hash",
			},
			supervisor: { state: "ready" as const, backend: "auto" as const, model: "model", port: 43123 },
		}),
	};

	it("writes the deterministic gateway config and embedding identity", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			runtime,
		}).sync();
		expect(result.ok).toBe(true);
		const config = parse(readFileSync(minSyncConfigPath(minsyncWorkspace), "utf8")) as Record<
			string,
			Record<string, unknown>
		>;
		expect(config.embedder).toMatchObject({
			id: "tei:Qwen3-Embedding-0.6B-Q8_0.gguf",
			base_url: "http://127.0.0.1:43123",
		});
		expect(config.vectorstore?.options).toMatchObject({ dimension: 1024 });
		const identity = JSON.parse(
			readFileSync(join(minsyncWorkspace, ".minsync", "autorag-embedding-identity.json"), "utf8"),
		);
		expect(identity).toMatchObject({ provider: "qwen", dimension: 1024, runtimeBuild: "b10951" });
	});

	it("keeps operator batching settings when a profile selects the runtime", async () => {
		writeFakeMinSync(JSON.stringify({ results: [] }));
		const result = await new MinSyncClient({
			binaryPath: minsyncBinary,
			workspacePath: minsyncWorkspace,
			runtime,
			embedder: {
				profile: "qwen3-embedding-0.6b",
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
			id: "tei:Qwen3-Embedding-0.6B-Q8_0.gguf",
			base_url: "http://127.0.0.1:43123",
			batch_size: 32,
			max_retries: 5,
			max_concurrent: 2,
			timeout_seconds: 120,
		});
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
		await new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace, runtime }).sync();
		const syncCall = loggedCalls()
			.map((line) => JSON.parse(line))
			.find((call) => call.args[0] === "sync");
		expect(syncCall.args).toEqual(["sync", "--full", "--format", "json"]);
		expect(existsSync(join(minsyncWorkspace, ".minsync", "cursor.json"))).toBe(true);
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
});
