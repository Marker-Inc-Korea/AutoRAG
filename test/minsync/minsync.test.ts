import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { ensureMinSyncBinary, MinSyncVectorMethod } from "../../src/minsync/index.ts";
import { saveMirrorIndex } from "../../src/mirror/index.ts";

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

function writeFakeMinSync(queryJson: string): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { appendFileSync } from "node:fs";

const args = process.argv.slice(2);
appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ args, cwd: process.cwd() }) + "\\n");

if (args[0] === "init") {
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}

if (args[0] === "sync") {
  console.log(JSON.stringify({ files_processed: 1, files_processed_paths: ["files/docs/policy.txt.md"] }));
  process.exit(0);
}

if (args[0] === "query") {
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
		expect(loggedCalls()).toContainEqual(JSON.stringify({ args: ["sync", "--format", "json"], cwd: minSyncCwd() }));
	});

	it("returns vector results resolved from parsed mirror paths back to virtual paths", async () => {
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
		expect(result.source).toBe("/docs/policy.txt");
		expect(result.content).toBe("Parsed renewal policy with cancellation terms.");
		expect(result.score).toBe(0.91);
		expect(result.metadata.method).toBe("minsync");
		expect(Object.keys(result.metadata)).toEqual(["method"]);
		expect(JSON.stringify(results)).not.toContain(source);
		expect(JSON.stringify(results)).not.toContain(root);
		expect(loggedCalls()).toContainEqual(
			JSON.stringify({
				args: ["query", "--format", "json", "-k", "2", "renewal cancellation"],
				cwd: minSyncCwd(),
			}),
		);
	});

	it("maps real MinSync relative file paths back to virtual paths", async () => {
		// Given
		writeFakeMinSync(
			JSON.stringify({
				results: [
					{
						path: "files/docs/policy.txt.md",
						score: 0.77,
						text: "Relative path hit from MinSync.",
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
		const results = await method.retrieve("relative path", { topK: 1 });

		// Then
		expect(results).toHaveLength(1);
		const result = requireValue(results[0], "relative path result");
		expect(result.source).toBe("/docs/policy.txt");
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

	it("installs the latest MinSync release asset into the AutoRAG bin cache when no binary exists", async () => {
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
				releaseProvider: async () => release,
			}),
		).rejects.toThrow("sha256");
	});
});
