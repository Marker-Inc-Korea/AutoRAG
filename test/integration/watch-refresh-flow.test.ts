import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGWatchRefreshHandle } from "../../src/index.ts";
import { parsedOutputPath } from "../../src/mirror/paths.ts";

let root: string;
let docs: string;
let handle: AutoRAGWatchRefreshHandle | undefined;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-watch-flow-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
});

afterEach(() => {
	handle?.stop();
	handle = undefined;
	rmSync(root, { recursive: true, force: true });
});

async function waitFor(predicate: () => boolean, timeoutMs = 4000): Promise<boolean> {
	const start = Date.now();
	while (Date.now() - start < timeoutMs) {
		if (predicate()) return true;
		await new Promise((resolve) => setTimeout(resolve, 40));
	}
	return predicate();
}

describe("AutoRAGAgent watch refresh (real fs)", () => {
	it("updates parsed mirrors when a watched source file is created, then stops cleanly", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		await agent.refresh(true);

		handle = agent.startWatchRefresh({ debounceMs: 30 });

		// Create a new source file in the watched directory.
		writeFileSync(join(docs, "new-note.txt"), "Freshly added note about invoices.\n");
		const mirrorPath = parsedOutputPath(root, "/docs/new-note.txt");
		const appeared = await waitFor(() => existsSync(mirrorPath));
		expect(appeared).toBe(true);

		// Stop the watcher; further changes must NOT trigger a refresh.
		handle.stop();
		const statusAfterStop = await agent.getRefreshStatus();
		const finishedAt = statusAfterStop.lastFinishedAt;

		writeFileSync(join(docs, "after-stop.txt"), "Should not be indexed by the watcher.\n");
		await new Promise((resolve) => setTimeout(resolve, 300));
		const afterStopMirror = parsedOutputPath(root, "/docs/after-stop.txt");

		expect(existsSync(afterStopMirror)).toBe(false);
		// No refresh ran after stop, so the last-finished timestamp is unchanged.
		expect((await agent.getRefreshStatus()).lastFinishedAt).toBe(finishedAt);
	});
});
