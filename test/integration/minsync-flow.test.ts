import { chmodSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { clearWorkspaceCache } from "../../src/agentdir/workspace.ts";

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
	clearWorkspaceCache();
});

afterEach(() => {
	clearWorkspaceCache();
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
});
