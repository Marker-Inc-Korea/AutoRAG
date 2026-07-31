import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { MinSyncClient } from "../../src/minsync/client.ts";

let root: string;
let docs: string;
let minsyncBinary: string;
let minsyncWorkspace: string;
let commandLog: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-first-sync-"));
	docs = join(root, "docs");
	minsyncBinary = join(root, "fake-minsync.mjs");
	minsyncWorkspace = join(root, ".autorag", "minsync");
	commandLog = join(root, "minsync-commands.jsonl");
	mkdirSync(docs, { recursive: true });
	mkdirSync(minsyncWorkspace, { recursive: true });
	writeFileSync(join(docs, "handbook.txt"), "Refund decisions require manager review.\n");
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFaithfulMinSync(
	options: {
		readonly checkFails?: boolean;
		readonly checkFailureReason?: string;
		readonly checkUnhealthyJson?: boolean;
		readonly checkUnhealthyOnce?: boolean;
		readonly fullSyncCreatesCursor?: boolean;
	} = {},
): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { appendFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";

const args = process.argv.slice(2);
const cwd = process.cwd();
const config = join(cwd, ".minsync", "config.toml");
const cursor = join(cwd, ".minsync", "cursor.json");
const unhealthyOnceMarker = join(cwd, ".minsync", "check-failed-once");
const log = ${JSON.stringify(commandLog)};
appendFileSync(log, JSON.stringify(args) + "\\n");

if (args[0] === "init") {
  if (existsSync(config)) {
    console.error("AlreadyInitialized");
    process.exit(1);
  }
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true, baseline_files: 1 }));
  process.exit(0);
}

if (args[0] === "check") {
  if (${options.checkFails === true ? "true" : "false"}) {
    console.error(${JSON.stringify(options.checkFailureReason ?? "embedder unavailable")});
    process.exit(1);
  }
  if (${options.checkUnhealthyOnce === true ? "true" : "false"} && !existsSync(unhealthyOnceMarker)) {
    writeFileSync(unhealthyOnceMarker, "failed");
    console.log(JSON.stringify({ all_passed: false, vectorstore_ok: true, embedder_ok: false }));
    process.exit(0);
  }
  if (${options.checkUnhealthyJson === true ? "true" : "false"}) {
    console.log(JSON.stringify({ all_passed: false, vectorstore_ok: true, embedder_ok: false }));
    process.exit(0);
  }
  console.log(JSON.stringify({ all_passed: true, vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
}

if (args[0] === "sync") {
  if (args.includes("--full")) {
    if (${options.fullSyncCreatesCursor === false ? "false" : "true"}) {
      mkdirSync(dirname(cursor), { recursive: true });
      writeFileSync(cursor, JSON.stringify({ ready: true }));
    }
    console.log(JSON.stringify({ files_processed: 1, embedded_texts: 1 }));
    process.exit(0);
  }
  if (existsSync(cursor)) {
    console.log(JSON.stringify({ files_processed: 0, embedded_texts: 0, already_up_to_date: true }));
    process.exit(0);
  }
  console.log(JSON.stringify({ files_processed: 0, embedded_texts: 0, already_up_to_date: true }));
  process.exit(0);
}

if (args[0] === "query") {
  if (!existsSync(cursor)) {
    console.error("never synced");
    process.exit(1);
  }
  console.log(JSON.stringify({
    results: [{
      path: ${JSON.stringify(join(root, ".autorag", "parsed", "files", "docs", "handbook.txt.md"))},
      score: 0.91,
      text: "Refund decisions require manager review."
    }]
  }));
  process.exit(0);
}

process.exit(2);
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function createAgent(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		bm25: false,
		minSync: { binaryPath: minsyncBinary, workspacePath: minsyncWorkspace },
	});
}

function loggedCommands(): string[][] {
	return readFileSync(commandLog, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line) as string[]);
}

describe("AutoRAGAgent MinSync first-sync contract (#1366)", () => {
	it("uses full sync and creates a cursor for a fresh MinSync client workspace", async () => {
		writeFaithfulMinSync();
		const client = new MinSyncClient({ binaryPath: minsyncBinary, workspacePath: minsyncWorkspace });

		const result = await client.sync();

		expect(result).toMatchObject({ ok: true, synced: 1 });
		expect(existsSync(join(minsyncWorkspace, ".minsync", "cursor.json"))).toBe(true);
		expect(loggedCommands()).toEqual([
			["init", "--format", "json"],
			["check", "--format", "json"],
			["sync", "--full", "--format", "json"],
		]);
	});

	it("full-syncs a fresh workspace, queries immediately, then refreshes incrementally", async () => {
		writeFaithfulMinSync();
		const agent = createAgent();

		const firstRefresh = await agent.refresh(true);
		const hits = await agent.retrieve("refund approval", { topK: 1 });
		const secondRefresh = await agent.refresh(false);

		expect(firstRefresh.minsync).toMatchObject({ ok: true, synced: 1 });
		expect(firstRefresh.minsync).not.toHaveProperty("workspacePath");
		expect(secondRefresh.minsync).toMatchObject({ ok: true, synced: 0 });
		expect(existsSync(join(minsyncWorkspace, ".minsync", "cursor.json"))).toBe(true);
		expect(hits.map((hit) => hit.source)).toEqual(["/docs/handbook.txt"]);
		expect(loggedCommands()).toEqual([
			["init", "--format", "json"],
			["check", "--format", "json"],
			["sync", "--full", "--format", "json"],
			["query", "--format", "json", "-k", "1", "refund approval"],
			["check", "--format", "json"],
			["sync", "--format", "json"],
		]);
	});

	it("recovers a config-only workspace with a full sync", async () => {
		writeFaithfulMinSync();
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(join(minsyncWorkspace, ".minsync", "config.toml"), '[embedder]\nid = "openai"\n');
		const agent = createAgent();

		const refresh = await agent.refresh(false);

		expect(refresh.minsync).toMatchObject({ ok: true, synced: 1 });
		expect(loggedCommands()).toEqual([
			["check", "--format", "json"],
			["sync", "--full", "--format", "json"],
		]);
	});

	it("reports exit-zero full sync without a cursor as unready", async () => {
		writeFaithfulMinSync({ fullSyncCreatesCursor: false });
		mkdirSync(join(minsyncWorkspace, ".minsync"), { recursive: true });
		writeFileSync(join(minsyncWorkspace, ".minsync", "config.toml"), '[embedder]\nid = "openai"\n');
		const agent = createAgent();

		const refresh = await agent.refresh(false);

		expect(refresh.minsync).toMatchObject({ ok: false, synced: 0 });
		expect(refresh.minsync?.reason).toContain("not-ready");
		expect(loggedCommands()).toEqual([
			["check", "--format", "json"],
			["sync", "--full", "--format", "json"],
		]);
		expect((await agent.getRefreshStatus()).components.minsync).toBe("degraded");
	});

	it("surfaces embedder preflight failure without claiming readiness", async () => {
		const privateFailure = `cannot open ${join(minsyncWorkspace, ".minsync", "config.toml")}`;
		writeFaithfulMinSync({ checkFails: true, checkFailureReason: privateFailure });
		const agent = createAgent();

		const refresh = await agent.refresh(true);

		expect(refresh.minsync).toMatchObject({ ok: false, synced: 0, reason: "check-failed" });
		expect(JSON.stringify(refresh.minsync)).not.toContain(root);
		expect(loggedCommands()).toEqual([
			["init", "--format", "json"],
			["check", "--format", "json"],
		]);
		expect((await agent.getRefreshStatus()).components.minsync).toBe("degraded");
	});

	it("rejects exit-zero unhealthy preflight JSON before sync", async () => {
		writeFaithfulMinSync({ checkUnhealthyJson: true });
		const agent = createAgent();

		const refresh = await agent.refresh(true);

		expect(refresh.minsync).toMatchObject({
			ok: false,
			synced: 0,
			reason: "check-failed: embedder unavailable",
		});
		expect(loggedCommands()).toEqual([
			["init", "--format", "json"],
			["check", "--format", "json"],
		]);
	});

	it("retries a transient preflight failure with a full sync", async () => {
		writeFaithfulMinSync({ checkUnhealthyOnce: true });
		const agent = createAgent();

		const first = await agent.refresh(true);
		const second = await agent.refresh(false);

		expect(first.minsync).toMatchObject({ ok: false, synced: 0 });
		expect(second.minsync).toMatchObject({ ok: true, synced: 1 });
		expect(loggedCommands()).toEqual([
			["init", "--format", "json"],
			["check", "--format", "json"],
			["check", "--format", "json"],
			["sync", "--full", "--format", "json"],
		]);
	});
});
