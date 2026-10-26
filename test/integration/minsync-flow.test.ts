import { chmodSync, mkdirSync, mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { MinSyncRequiredError } from "../../src/minsync/errors.ts";
import { fakeMinSyncStageSimulatorPath } from "../helpers/fake-minsync.ts";

let root: string;
let docs: string;
let minsyncBinary: string;
let minsyncWorkspace: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-minsync-flow-"));
	docs = join(root, "docs");
	minsyncBinary = join(root, "fake-minsync.mjs");
	minsyncWorkspace = join(root, ".autorag", "minsync");
	mkdirSync(docs, { recursive: true });
	mkdirSync(minsyncWorkspace, { recursive: true });
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

function writeFakeMinSync(): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const cursor = join(process.cwd(), ".minsync", "cursor.json");

// A parsed refresh drives MinSync's materialization stage in a long-lived
// session: hand the state directory to the shared simulator and let it keep
// reading this process's stdin instead of exiting.
if (args[0] === "stage") {
  const eqArg = args.find((a) => a.startsWith("--state-dir="));
  const flagAt = args.indexOf("--state-dir");
  const stateDir = eqArg ? eqArg.slice("--state-dir=".length) : flagAt >= 0 ? args[flagAt + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  const config = join(process.cwd(), ".minsync", "config.toml");
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
} else if (args[0] === "sync") {
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ synced: 1 }));
  process.exit(0);
} else if (args[0] === "query") {
  console.log(JSON.stringify({
    results: [
      {
        path: ${JSON.stringify(join(root, ".autorag", "parsed", "files", "docs", "handbook.txt.md"))},
        score: 0.83,
        text: "Parsed handbook says refunds are approved after manager review."
      }
    ]
  }));
  process.exit(0);
} else {
  process.exit(2);
}
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function writeFakeScopedMinSync(): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
import { mkdirSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
const require = createRequire(import.meta.url);
const args = process.argv.slice(2);
const cursor = join(process.cwd(), ".minsync", "cursor.json");

// A parsed refresh drives MinSync's materialization stage in a long-lived
// session: hand the state directory to the shared simulator and let it keep
// reading this process's stdin instead of exiting.
if (args[0] === "stage") {
  const eqArg = args.find((a) => a.startsWith("--state-dir="));
  const flagAt = args.indexOf("--state-dir");
  const stateDir = eqArg ? eqArg.slice("--state-dir=".length) : flagAt >= 0 ? args[flagAt + 1] : args[1];
  require(${JSON.stringify(fakeMinSyncStageSimulatorPath)}).runStage({ stateDir });
} else if (args[0] === "init") {
  const config = join(process.cwd(), ".minsync", "config.toml");
  mkdirSync(dirname(config), { recursive: true });
  writeFileSync(config, "[embedder]\\nid = \\"openai\\"\\n");
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
} else if (args[0] === "check") {
  console.log(JSON.stringify({ vectorstore_ok: true, embedder_ok: true }));
  process.exit(0);
} else if (args[0] === "sync") {
  mkdirSync(dirname(cursor), { recursive: true });
  writeFileSync(cursor, JSON.stringify({ ready: true }));
  console.log(JSON.stringify({ synced: 2 }));
  process.exit(0);
} else if (args[0] === "query") {
  const k = Number(args[args.indexOf("-k") + 1]);
  const results = [
    {
      path: ${JSON.stringify(join(root, ".autorag", "parsed", "files", "docs", "outside.txt.md"))},
      score: 0.99,
      text: "Out of scope semantic hit."
    }
  ];
  if (k > 1) {
    results.push({
      path: ${JSON.stringify(join(root, ".autorag", "parsed", "files", "docs", "sub", "inside.txt.md"))},
      score: 0.72,
      text: "Scoped semantic hit inside the requested folder."
    });
  }
  console.log(JSON.stringify({ results }));
  process.exit(0);
} else {
  process.exit(2);
}
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function requireValue<T>(value: T | undefined, label: string): T {
	if (value === undefined) throw new Error(`missing ${label}`);
	return value;
}

describe("AutoRAGAgent MinSync integration", () => {
	it("includes MinSync hybrid results in retrieve() with original source paths", async () => {
		// Given
		writeFileSync(join(docs, "handbook.txt"), "Refund decisions require manager review.\n");
		writeFakeMinSync();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: {
				binaryPath: minsyncBinary,
				workspacePath: minsyncWorkspace,
			},
		});
		await agent.refresh(true);

		// When
		const results = await agent.retrieve("semantic refund approval", { topK: 1 });

		// Then
		expect(results).toHaveLength(1);
		const result = requireValue(results[0], "first retrieval result");
		expect(result.source).toBe(realpathSync(join(docs, "handbook.txt")));
		expect(readFileSync(result.source, "utf8")).toBe("Refund decisions require manager review.\n");
		expect(result.content).toContain("refunds are approved");
		expect(result.metadata).toMatchObject({ method: "minsync-hybrid", virtualPath: "/docs/handbook.txt" });
	});

	it("over-queries before filtering scoped vector results", async () => {
		writeFileSync(join(docs, "outside.txt"), "Outside original content.\n");
		mkdirSync(join(docs, "sub"), { recursive: true });
		writeFileSync(join(docs, "sub", "inside.txt"), "Inside original content.\n");
		writeFakeScopedMinSync();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: {
				binaryPath: minsyncBinary,
				workspacePath: minsyncWorkspace,
			},
		});
		await agent.refresh(true);

		const results = await agent.retrieve("semantic marker", { topK: 1, scope: "/docs/sub" });

		expect(results).toHaveLength(1);
		expect(results[0]?.source).toBe(realpathSync(join(docs, "sub", "inside.txt")));
		expect(results[0]?.content).toContain("Scoped semantic hit");
	});
	it("fails the search with a path-free MinSyncRequiredError when the binary is missing (#21)", async () => {
		writeFileSync(join(docs, "handbook.txt"), "Refund decisions require manager review.\n");
		const missingBinary = join(root, "missing-minsync");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			jikji: false,
			everything: false,
			fsearch: false,
			minSync: { binaryPath: missingBinary, workspacePath: minsyncWorkspace },
		});

		const failure = await agent.retrieveWithDiagnostics("manager", { topK: 5 }).catch((error: unknown) => error);

		expect(failure).toBeInstanceOf(MinSyncRequiredError);
		expect((failure as Error).message).not.toContain(missingBinary);
		expect((failure as Error).message).not.toContain(root);
	});
});
