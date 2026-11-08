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
 * the staged paths it saw). `failAtSync` makes the n-th sync exit non-zero; `holdFirstSyncPath`
 * keeps the first sync running until that file exists, so a test decides when it may finish.
 */
function writeBatchedMinSync(
	binaryPath: string,
	logPath: string,
	counterPath: string,
	options: { readonly failAtSync?: number; readonly holdFirstSyncPath?: string } = {},
): void {
	const { failAtSync = 0, holdFirstSyncPath = "" } = options;
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
  if (run === 1 && ${JSON.stringify(holdFirstSyncPath)}) {
    // Another process releases this sync, so the wait is on a file the test writes, not on a timer.
    const hold = ${JSON.stringify(holdFirstSyncPath)};
    const deadline = Date.now() + 20000;
    while (!existsSync(hold) && Date.now() < deadline) Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, 25);
    writeFileSync(hold + ".result", existsSync(hold) ? "released" : "timeout");
  }
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

/**
 * Releases a held first sync once parsing reaches its last source. A refresh that parsed and synced one
 * batch at a time would never get there while the sync waits, so a release proves the two overlapped.
 */
function releaseWhenParsingEnds(holdPath: string, total: number): (progress: AutoRAGRefreshProgress) => void {
	return (progress) => {
		if (progress.phase === "parsed" && progress.processed === total - 1) writeFileSync(holdPath, "");
	};
}

describe("batched refresh", () => {
	it("syncs the first batch while the rest is still being parsed", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE + 5;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		const hold = join(root, "hold");
		writeBatchedMinSync(binary, logPath, counterPath, { holdFirstSyncPath: hold });
		const workspacePath = join(root, ".autorag", "minsync");
		const method = new MinSyncVectorMethod({ root, binaryPath: binary, workspacePath, autoInstall: false });
		const agent = makeAgent({ binary, workspacePath });

		await agent.refresh(true, { methods: ["minsync"], onProgress: releaseWhenParsingEnds(hold, total) });

		// The first sync only finished because parsing carried on to the last source.
		expect(readFileSync(`${hold}.result`, "utf8")).toBe("released");
		// Two syncs, not one per batch: the first covered batch 1, the second everything committed meanwhile.
		const runs = syncRuns(logPath);
		expect(runs).toHaveLength(2);
		expect(runs[0]?.staged.length).toBeGreaterThanOrEqual(DEFAULT_MIRROR_BATCH_SIZE);
		expect(runs[1]?.staged).toHaveLength(total);
		expect(method.isReady()).toBe(true);
		expect(queryHits(binary, workspacePath)).toBe(total);
		const results = await method.retrieve("content", { topK: 5 });
		expect(results.length).toBeGreaterThan(0);
	});

	it("continues past a failing MinSync sync and a later sync recovers its files", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE * 2 + 10;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		const hold = join(root, "hold");
		writeBatchedMinSync(binary, logPath, counterPath, { failAtSync: 1, holdFirstSyncPath: hold });
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });

		const result = await agent.refresh(true, {
			methods: ["minsync"],
			onProgress: releaseWhenParsingEnds(hold, total),
		});

		// The failure did not stop the next sync, which indexed the failed sync's still-staged files.
		const runs = syncRuns(logPath);
		expect(runs).toHaveLength(2);
		expect(runs[1]?.staged).toHaveLength(total);
		expect(result.minsync?.ok).toBe(true);
		expect(result.diagnostics.filter((diagnostic) => diagnostic.code === "minsync-sync-failed")).toEqual([]);
	});

	it("reports the files covered by a failed final sync", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE * 2 + 10;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		const hold = join(root, "hold");
		writeBatchedMinSync(binary, logPath, counterPath, { failAtSync: 2, holdFirstSyncPath: hold });
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });

		const result = await agent.refresh(true, {
			methods: ["minsync"],
			onProgress: releaseWhenParsingEnds(hold, total),
		});

		// The first sync covered batch 1 and succeeded; the failed second one covered every later batch.
		expect(result.minsync?.ok).toBe(false);
		const sources = result.diagnostics
			.filter((diagnostic) => diagnostic.code === "minsync-sync-failed" && diagnostic.source !== "minsync")
			.map((diagnostic) => diagnostic.source);
		expect(sources).toContain("/docs/f25.txt");
		expect(sources).toContain("/docs/f59.txt");
		expect(sources).not.toContain("/docs/f24.txt");
		expect(sources).toHaveLength(total - DEFAULT_MIRROR_BATCH_SIZE);
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

	it("starts no sync for batches whose mirrors did not change, and always syncs after the final one", async () => {
		const total = DEFAULT_MIRROR_BATCH_SIZE * 2 + 10;
		writeSources(total);
		const binary = join(root, "batched-minsync.mjs");
		const logPath = join(root, "minsync.log");
		const counterPath = join(root, "minsync.counter");
		const hold = join(root, "hold");
		writeBatchedMinSync(binary, logPath, counterPath, { holdFirstSyncPath: hold });
		const workspacePath = join(root, ".autorag", "minsync");
		const agent = makeAgent({ binary, workspacePath });
		await agent.refresh(true, { methods: ["minsync"], onProgress: releaseWhenParsingEnds(hold, total) });
		expect(syncRuns(logPath)).toHaveLength(2);

		// Nothing changed: only the final batch syncs.
		await agent.refresh(false, { methods: ["minsync"] });
		expect(syncRuns(logPath)).toHaveLength(3);

		// A change in the middle batch syncs it, and the final batch syncs once more.
		writeFileSync(join(docs, "f30.txt"), "changed content\n");
		await agent.refresh(false, { methods: ["minsync"] });
		expect(syncRuns(logPath)).toHaveLength(5);
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
