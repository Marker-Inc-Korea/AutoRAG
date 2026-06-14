import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { clearWorkspaceCache } from "../../src/agentdir/workspace.ts";
import { loadMirrorIndex } from "../../src/mirror/index.ts";

let root: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-agent-mirror-"));
	clearWorkspaceCache();
});

afterEach(() => {
	clearWorkspaceCache();
	rmSync(root, { recursive: true, force: true });
});

describe("AutoRAGAgent parsed mirror integration", () => {
	it("refresh() syncs parsed markdown mirrors for supported files", async () => {
		const docs = join(root, "docs");
		mkdirSync(docs, { recursive: true });
		writeFileSync(join(docs, "note.txt"), "Mirror me\n");
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});

		await agent.refresh(true);
		const index = loadMirrorIndex(root);
		const outputPath = index.entries["/docs/note.txt"]?.outputPath;

		expect(outputPath).toBeDefined();
		if (!outputPath) throw new Error("expected parsed mirror output path");
		expect(existsSync(outputPath)).toBe(true);
		expect(readFileSync(outputPath, "utf8")).toBe("Mirror me\n");
	});
});
