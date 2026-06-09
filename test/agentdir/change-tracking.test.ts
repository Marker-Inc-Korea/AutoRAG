import { mkdirSync, mkdtempSync, rmSync, statSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import {
	bootstrapMappings,
	clearWorkspaceCache,
	getWorkspace,
	refreshWorkspace,
} from "../../src/agentdir/workspace.ts";

let root: string;
let source: string;
const FIXED_SECONDS = 1_700_000_000; // whole seconds => exact mtime, no sub-second drift

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-change-"));
	clearWorkspaceCache();
	source = join(root, "docs");
	mkdirSync(source, { recursive: true });
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

/** Replace a file's content with same-length bytes and restore the exact original mtime. */
function spoofSameSizeSameMtime(file: string, newContent: string): void {
	const before = statSync(file, { bigint: true });
	writeFileSync(file, newContent);
	const after = statSync(file, { bigint: true });
	if (after.size !== before.size) throw new Error("test setup: replacement must be the same byte length");
	utimesSync(file, FIXED_SECONDS, FIXED_SECONDS);
}

describe("issue #2 — hash-verified change detection", () => {
	it("detects same-size/same-mtime content swaps only when hash verification is enabled (AC-7)", async () => {
		const file = join(source, "x.txt");
		writeFileSync(file, "AAAA\n");
		utimesSync(file, FIXED_SECONDS, FIXED_SECONDS);

		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);

		// Replace content with the same byte length and restore the exact mtime.
		spoofSameSizeSameMtime(file, "BBBB\n");

		// mtime+size are unchanged, so the default refresh misses the swap...
		const plain = await refreshWorkspace(ws, { verifyHashes: false });
		expect(plain.refreshed).toBe(0);

		// ...but hash verification catches it.
		const verified = await refreshWorkspace(ws, { verifyHashes: true });
		expect(verified.refreshed).toBeGreaterThanOrEqual(1);
	});

	it("AutoRAGAgent.refresh(true) surfaces hash-verified detection", async () => {
		const file = join(source, "y.txt");
		writeFileSync(file, "1234\n");
		utimesSync(file, FIXED_SECONDS, FIXED_SECONDS);

		const agent = new AutoRAGAgent({
			searchPaths: [source],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		await agent.refresh(false); // establish catalog baseline

		spoofSameSizeSameMtime(file, "5678\n");

		expect((await agent.refresh(false)).refreshed).toBe(0);
		expect((await agent.refresh(true)).refreshed).toBeGreaterThanOrEqual(1);
	});

	it("a genuine size change is detected even without hash verification", async () => {
		const file = join(source, "z.txt");
		writeFileSync(file, "short\n");
		utimesSync(file, FIXED_SECONDS, FIXED_SECONDS);
		const ws = getWorkspace(root);
		await bootstrapMappings(ws, [source]);

		writeFileSync(file, "a much longer line of content\n");
		utimesSync(file, FIXED_SECONDS, FIXED_SECONDS);

		const plain = await refreshWorkspace(ws, { verifyHashes: false });
		expect(plain.refreshed).toBeGreaterThanOrEqual(1);
	});
});
