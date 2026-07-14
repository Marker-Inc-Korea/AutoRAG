import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

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
const args = process.argv.slice(2);

if (args[0] === "init") {
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}

if (args[0] === "sync") {
  console.log(JSON.stringify({ synced: 1 }));
  process.exit(0);
}

if (args[0] === "query") {
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
}

process.exit(2);
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function writeFakeScopedMinSync(): void {
	writeFileSync(
		minsyncBinary,
		`#!/usr/bin/env node
const args = process.argv.slice(2);

if (args[0] === "init") {
  console.log(JSON.stringify({ initialized: true }));
  process.exit(0);
}

if (args[0] === "sync") {
  console.log(JSON.stringify({ synced: 2 }));
  process.exit(0);
}

if (args[0] === "query") {
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
}

process.exit(2);
`,
	);
	chmodSync(minsyncBinary, 0o755);
}

function requireValue<T>(value: T | undefined, label: string): T {
	if (value === undefined) throw new Error(`missing ${label}`);
	return value;
}

describe("AutoRAGAgent MinSync integration", () => {
	it("includes MinSync vector results in retrieve() and exposes only virtual paths", async () => {
		// Given
		writeFileSync(join(docs, "handbook.txt"), "Refund decisions require manager review.\n");
		writeFakeMinSync();
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			bm25: false,
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
		expect(result.source).toBe("/docs/handbook.txt");
		expect(result.content).toContain("refunds are approved");
		expect(result.metadata.method).toBe("minsync");
		expect(JSON.stringify(results)).not.toContain(docs);
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
			bm25: false,
			minSync: {
				binaryPath: minsyncBinary,
				workspacePath: minsyncWorkspace,
			},
		});
		await agent.refresh(true);

		const results = await agent.retrieve("semantic marker", { topK: 1, scope: "/docs/sub" });

		expect(results).toHaveLength(1);
		expect(results[0]?.source).toBe("/docs/sub/inside.txt");
		expect(results[0]?.content).toContain("Scoped semantic hit");
	});
	it("surfaces a path-free minsync-unavailable diagnostic when the binary is missing (#21)", async () => {
		writeFileSync(join(docs, "handbook.txt"), "Refund decisions require manager review.\n");
		const missingBinary = join(root, "missing-minsync");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
			bm25: false,
			minSync: { binaryPath: missingBinary, workspacePath: minsyncWorkspace },
		});

		const { results, diagnostics } = await agent.retrieveWithDiagnostics("manager", { topK: 5 });

		expect(results).toEqual([]);
		const minsync = diagnostics.find((d) => d.source === "minsync");
		expect(minsync?.code).toBe("minsync-unavailable");
		expect(minsync?.message).not.toContain(missingBinary);
		expect(minsync?.message).not.toContain(root);
	});
});
