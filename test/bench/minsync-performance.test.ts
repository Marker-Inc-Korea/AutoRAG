import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { MinSyncVectorMethod } from "../../src/minsync/method.ts";
import { acquireRefreshLock } from "../../src/mirror/refresh-lock.ts";
import { syncParsedMirrors } from "../../src/mirror/sync.ts";

let root: string;
let docs: string;
let binary: string;
let log: string;
let workspace: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-performance-"));
	docs = join(root, "docs");
	binary = join(root, "fake-minsync.mjs");
	log = join(root, "calls.jsonl");
	workspace = join(root, ".autorag", "minsync");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(docs, "a.txt"), "alpha policy\n");
	writeFileSync(
		binary,
		`#!/usr/bin/env node
import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
const args = process.argv.slice(2);
const log = ${JSON.stringify(log)};
const config = join(process.cwd(), ".minsync", "config.toml");
const cursor = join(process.cwd(), ".minsync", "cursor.json");
appendFileSync(log, JSON.stringify(args) + "\\n");
if (args[0] === "init") { mkdirSync(dirname(config), { recursive: true }); writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n"); process.exit(0); }
if (args[0] === "check") { console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true })); process.exit(0); }
if (args[0] === "sync") { mkdirSync(dirname(cursor), { recursive: true }); writeFileSync(cursor, "{}\\n"); console.log(JSON.stringify({ files_processed: 1 })); process.exit(0); }
if (args[0] === "query") { console.log(JSON.stringify({ results: [{ path: "files/docs/a.txt.md", score: 0.9, text: "alpha policy" }] })); process.exit(0); }
process.exit(2);
`,
	);
	chmodSync(binary, 0o755);
});

afterEach(() => rmSync(root, { recursive: true, force: true }));

function calls(): string[][] {
	if (!existsSync(log)) return [];
	return readFileSync(log, "utf8")
		.trim()
		.split("\n")
		.filter(Boolean)
		.map((line) => JSON.parse(line) as string[]);
}

describe("MinSync unchanged-workspace trust and query cache", () => {
	it("excludes a second refresh process through the workspace lock", () => {
		const first = acquireRefreshLock(root);
		expect(first).toBeDefined();
		expect(acquireRefreshLock(root)).toBeUndefined();
		first?.release();
		const afterRelease = acquireRefreshLock(root);
		expect(afterRelease).toBeDefined();
		afterRelease?.release();
	});

	it("coalesces identical refreshes and refuses a different concurrent refresh", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			workspacePath: root,
			minSync: { binaryPath: binary, workspacePath: workspace, autoInstall: false },
		});
		let release: (() => void) | undefined;
		const blocked = new Promise<void>((resolve) => {
			release = resolve;
		});
		const original = agent.syncParsedMirrors.bind(agent);
		let calls = 0;
		agent.syncParsedMirrors = async (force) => {
			calls += 1;
			await blocked;
			return original(force);
		};

		const first = agent.refresh(true);
		const joined = agent.refresh(true);
		const busy = await agent.refresh(false);
		expect(busy.outcome).toBe("busy");
		release?.();
		expect(await joined).toBe(await first);
		expect(calls).toBe(1);
	});

	it("skips the external sync when parsed mirror content is unchanged", async () => {
		await syncParsedMirrors({ root, searchPaths: [docs] });
		const method = new MinSyncVectorMethod({
			root,
			binaryPath: binary,
			workspacePath: workspace,
			autoInstall: false,
		});

		const first = await method.sync();
		const second = await method.sync();

		expect(first).toMatchObject({ ok: true, synced: 1 });
		expect(second).toMatchObject({ ok: true, synced: 0, skipped: true });
		expect(calls().filter((args) => args[0] === "sync")).toHaveLength(1);

		const forced = await method.sync({ force: true });
		expect(forced).toMatchObject({ ok: true, synced: 1 });
		expect(calls().filter((args) => args[0] === "sync")).toHaveLength(2);
	});

	it("reuses same-fingerprint query results and invalidates them after a mirror change", async () => {
		await syncParsedMirrors({ root, searchPaths: [docs] });
		const method = new MinSyncVectorMethod({
			root,
			binaryPath: binary,
			workspacePath: workspace,
			autoInstall: false,
		});
		await method.sync();

		const first = await method.retrieve("alpha", { topK: 1 });
		const second = await method.retrieve("alpha", { topK: 1 });
		expect(second).toEqual(first);
		expect(calls().filter((args) => args[0] === "query")).toHaveLength(1);

		writeFileSync(join(docs, "a.txt"), "beta policy changed\n");
		await syncParsedMirrors({ root, searchPaths: [docs], force: true });
		const refreshed = await method.sync();
		const after = await method.retrieve("alpha", { topK: 1 });

		expect(refreshed.ok).toBe(true);
		expect(refreshed.skipped).toBeUndefined();
		expect(after).toHaveLength(1);
		expect(calls().filter((args) => args[0] === "query")).toHaveLength(2);
	});
});
