import { spawnSync } from "node:child_process";
import {
	chmodSync,
	existsSync,
	mkdirSync,
	mkdtempSync,
	readdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGRefreshProgress, MinSyncVectorMethod } from "../../src/index.ts";
import { loadMirrorIndex } from "../../src/mirror/index.ts";
import { DEFAULT_MIRROR_BATCH_SIZE } from "../../src/mirror/sync.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-batched-refresh-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(options: {
	readonly binary: string;
	readonly workspacePath: string;
	readonly extra?: Record<string, unknown>;
}) {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { binaryPath: options.binary, workspacePath: options.workspacePath, autoInstall: false },
		...options.extra,
	});
}

function writeSources(count: number): void {
	for (let index = 0; index < count; index += 1) {
		writeFileSync(join(docs, `f${String(index).padStart(2, "0")}.txt`), `content ${index}\n`);
	}
}

/**
 * A MinSync stand-in that stages through the shared simulator, answers queries by
 * walking the staged `files/`, and records every `sync` run (invocation number and
 * the staged paths it saw). `failAtSync` makes the n-th sync exit non-zero.
 */
function writeBatchedMinSync(binaryPath: string, logPath: string, counterPath: string, failAtSync = 0): void {
	writeFileSync(
		binaryPath,
		`#!/usr/bin/env node
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";

const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");

const walk = (dir, rel) => {
  const out = [];
  if (!existsSync(dir)) return out;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const nextRel = rel ? rel + "/" + entry.name : entry.name;
    const nextPath = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...walk(nextPath, nextRel));
    else out.push("files/" + nextRel);
  }
  return out;
};

if (args[0] === "stage") {
  const eqArg = args.find((a) => a.startsWith("--state-dir="));
  const flagAt = args.indexOf("--state-dir");
  const stateDir = eqArg ? eqArg.slice("--state-dir=".length) : flagAt >= 0 ? args[flagAt + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, '[embedder]\\nid = "fixture"\\n');
  console.log(JSON.stringify({ initialized: true }));
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
} else if (args[0] === "sync") {
  let run = 0;
  try { run = Number(readFileSync(${JSON.stringify(counterPath)}, "utf8")); } catch {}
  run += 1;
  writeFileSync(${JSON.stringify(counterPath)}, String(run));
  const staged = walk(join(process.cwd(), "files"), "");
  appendFileSync(${JSON.stringify(logPath)}, JSON.stringify({ run, staged }) + "\\n");
  if (${failAtSync} > 0 && run === ${failAtSync}) {
    console.error("synthetic batch failure");
    process.exit(1);
  }
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ files_processed: staged.length }));
} else if (args[0] === "query") {
  const hits = walk(join(process.cwd(), "files"), "").map((path) => ({
    path,
    score: 0.9,
    text: readFileSync(join(process.cwd(), path), "utf8"),
  }));
  console.log(JSON.stringify({ results: hits }));
} else {
  console.error("unexpected fake minsync command: " + args.join(" "));
  process.exit(2);
}
`,
	);
	chmodSync(binaryPath, 0o755);
}

function syncRuns(logPath: string): { readonly run: number; readonly staged: readonly string[] }[] {
	if (!existsSync(logPath)) return [];
	const runs: { run: number; staged: string[] }[] = [];
	for (const line of readFileSync(logPath, "utf8").split("\n")) {
		if (line.length === 0) continue;
		const parsed: unknown = JSON.parse(line);
		if (
			typeof parsed === "object" &&
			parsed !== null &&
			"run" in parsed &&
			typeof parsed.run === "number" &&
			"staged" in parsed &&
			Array.isArray(parsed.staged)
		) {
			runs.push({
				run: parsed.run,
				staged: parsed.staged.filter((path): path is string => typeof path === "string"),
			});
		}
	}
	return runs;
}

function stagedFiles(workspacePath: string): string[] {
	const out: string[] = [];
	const walk = (dir: string, rel: string): void => {
		if (!existsSync(dir)) return;
		for (const entry of readdirSync(dir, { withFileTypes: true })) {
			const nextRel = rel ? `${rel}/${entry.name}` : entry.name;
			if (entry.isDirectory()) walk(join(dir, entry.name), nextRel);
			else out.push(`files/${nextRel}`);
		}
	};
	walk(join(workspacePath, "files"), "");
	return out.sort();
}

function queryHits(binary: string, workspacePath: string): number {
	const result = spawnSync(binary, ["query", "--format", "json", "-k", "1000", "anything"], {
		cwd: workspacePath,
		encoding: "utf8",
	});
	const parsed: unknown = JSON.parse(result.stdout);
	if (typeof parsed === "object" && parsed !== null && "results" in parsed && Array.isArray(parsed.results)) {
		return parsed.results.length;
	}
	return 0;
}

describe("batched refresh", () => {
	it("makes the first batch searchable before later batches are parsed", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE + 5;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		writeBatchedMinSync(binary, logPath, counterPath);
		const workspacePath = join(root, ".autorag", "minsync");
		const method = new MinSyncVectorMethod({ root, binaryPath: binary, workspacePath, autoInstall: false });
		const agent = makeAgent({ binary, workspacePath });

		let boundary:
			| {
					readonly syncs: number;
					readonly ready: boolean;
					readonly staged: readonly string[];
					readonly queryHits: number;
					readonly currentVirtualPath?: string;
			  }
			| undefined;
		await agent.refresh(true, {
			methods: ["minsync"],
			onProgress: (progress: AutoRAGRefreshProgress) => {
				if (boundary !== undefined) return;
				if (progress.phase !== "parsed" || progress.processed !== DEFAULT_MIRROR_BATCH_SIZE) return;
				// This fires at the top of the first batch-2 file, after batch 1's sync.
				boundary = {
					syncs: syncRuns(logPath).length,
					ready: method.isReady(),
					staged: stagedFiles(workspacePath),
					queryHits: queryHits(binary, workspacePath),
					currentVirtualPath: progress.currentVirtualPath,
				};
			},
		});

		expect(boundary).toBeDefined();
		expect(boundary?.currentVirtualPath).toMatch(/f25\.txt$/);
		expect(boundary?.syncs).toBe(1);
		expect(boundary?.ready).toBe(true);
		expect(boundary?.staged).toHaveLength(DEFAULT_MIRROR_BATCH_SIZE);
		expect(boundary?.staged.some((path) => path.endsWith("f24.txt.md"))).toBe(true);
		expect(boundary?.staged.some((path) => path.endsWith("f25.txt.md"))).toBe(false);
		expect(boundary?.queryHits).toBe(DEFAULT_MIRROR_BATCH_SIZE);

		// The remaining batch ran too, and the whole corpus is queryable afterwards.
		const runs = syncRuns(logPath);
		expect(runs).toHaveLength(2);
		expect(runs[1]?.staged).toHaveLength(total);
		const results = await method.retrieve("content", { topK: 5 });
		expect(results.length).toBeGreaterThan(0);
	});

	it("continues past a failing MinSync batch and reports that batch's files", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE * 2 + 10;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		writeBatchedMinSync(binary, logPath, counterPath, 2);
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });

		const result = await agent.refresh(true, { methods: ["minsync"] });

		// Every batch was attempted: the failure did not stop the ones after it.
		const runs = syncRuns(logPath);
		expect(runs).toHaveLength(3);
		expect(runs[0]?.staged).toHaveLength(DEFAULT_MIRROR_BATCH_SIZE);
		expect(runs[1]?.staged).toHaveLength(DEFAULT_MIRROR_BATCH_SIZE * 2);
		expect(runs[2]?.staged).toHaveLength(total);
		expect(runs[2]?.staged.some((path) => path.endsWith("f59.txt.md"))).toBe(true);

		expect(result.minsync?.ok).toBe(false);
		const failed = result.diagnostics.filter((diagnostic) => diagnostic.code === "minsync-sync-failed");
		const sources = failed.map((diagnostic) => diagnostic.source);
		expect(sources).toContain("/docs/f25.txt");
		expect(sources).toContain("/docs/f49.txt");
		expect(sources).not.toContain("/docs/f00.txt");
		expect(sources).not.toContain("/docs/f50.txt");
	});

	it("keeps not-yet-reached documents mid-refresh and deletes removed sources at the end", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE + 5;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		writeBatchedMinSync(binary, logPath, counterPath);
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });

		await agent.refresh(true, { methods: ["minsync"] });
		// Remove a source from the first batch's range before the incremental refresh.
		rmSync(join(docs, "f00.txt"));

		let boundary:
			| {
					readonly staged: readonly string[];
					readonly mirrorHasRemoved: boolean;
					readonly mirrorHasUnreached: boolean;
			  }
			| undefined;
		const result = await agent.refresh(false, {
			methods: ["minsync"],
			onProgress: (progress: AutoRAGRefreshProgress) => {
				if (boundary !== undefined) return;
				if (progress.phase !== "parsed" || progress.processed !== DEFAULT_MIRROR_BATCH_SIZE) return;
				const index = loadMirrorIndex(root, { minSyncBinaryPath: binary });
				boundary = {
					staged: stagedFiles(workspacePath),
					mirrorHasRemoved: Object.hasOwn(index.entries, "/docs/f00.txt"),
					mirrorHasUnreached: Object.hasOwn(index.entries, "/docs/f29.txt"),
				};
			},
		});

		// The removed source is only deleted with the final batch; not-yet-reached
		// documents keep their staged copy and their mirror entry.
		expect(boundary?.staged.some((path) => path.endsWith("f00.txt.md"))).toBe(true);
		expect(boundary?.staged.some((path) => path.endsWith("f29.txt.md"))).toBe(true);
		expect(boundary?.mirrorHasRemoved).toBe(true);
		expect(boundary?.mirrorHasUnreached).toBe(true);

		// By the end the removed source is gone from both the mirror and staging.
		expect(result.deleted).toBe(1);
		expect(loadMirrorIndex(root, { minSyncBinaryPath: binary }).entries["/docs/f00.txt"]).toBeUndefined();
		expect(stagedFiles(workspacePath).some((path) => path.endsWith("f00.txt.md"))).toBe(false);
	});

	it("syncs only batches whose mirrors changed, and always the final one", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE * 2 + 10;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		writeBatchedMinSync(binary, logPath, counterPath);
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });
		await agent.refresh(true, { methods: ["minsync"] });
		expect(syncRuns(logPath)).toHaveLength(3);

		// Nothing changed: one sync (the final batch) instead of one per batch.
		await agent.refresh(false, { methods: ["minsync"] });
		expect(syncRuns(logPath)).toHaveLength(4);

		// A change in the middle batch syncs that batch and the final one.
		writeFileSync(join(docs, "f30.txt"), "changed content\n");
		await agent.refresh(false, { methods: ["minsync"] });
		expect(syncRuns(logPath)).toHaveLength(6);
	});

	it("still suppresses a mass delete in the batched refresh", async () => {
		for (let index = 0; index < 150; index += 1) writeFileSync(join(docs, `n${index}.txt`), `Note ${index}\n`);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		writeBatchedMinSync(binary, logPath, counterPath);
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({
			binary,
			workspacePath,
			extra: { refresh: { maxDeleteRatio: 0.1 } },
		});

		await agent.refresh(false, { methods: ["minsync"] });
		for (let index = 0; index < 120; index += 1) rmSync(join(docs, `n${index}.txt`));

		const suppressed = await agent.refresh(false, { methods: ["minsync"] });
		expect(suppressed.deleted).toBe(0);
		expect(suppressed.diagnostics.some((diagnostic) => diagnostic.code === "mass-delete-suppressed")).toBe(true);

		const forced = await agent.refresh(true, { methods: ["minsync"] });
		expect(forced.deleted).toBe(120);
	});
});
