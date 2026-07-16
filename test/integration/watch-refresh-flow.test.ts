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

// Real recursive fs.watch delivery latency varies under load; poll generously
// and retry so an occasional delayed/dropped OS event does not flake the suite.
async function waitFor(predicate: () => boolean | Promise<boolean>, timeoutMs = 10000): Promise<boolean> {
	const start = Date.now();
	while (Date.now() - start < timeoutMs) {
		if (await predicate()) return true;
		await new Promise((resolve) => setTimeout(resolve, 40));
	}
	return await predicate();
}

describe("AutoRAGAgent watch refresh (real fs)", () => {
	it(
		"updates parsed mirrors when a watched source file is created, then stops cleanly",
		{ retry: 3, timeout: 30000 },
		async () => {
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

			// An in-flight refresh may still be completing after stop(). Wait for it
			// to finish before capturing the baseline timestamp.
			await waitFor(async () => !(await agent.getRefreshStatus()).inFlight, 10000);
			const finishedAt = (await agent.getRefreshStatus()).lastFinishedAt;

			writeFileSync(join(docs, "after-stop.txt"), "Should not be indexed by the watcher.\n");
			await new Promise((resolve) => setTimeout(resolve, 300));
			const afterStopMirror = parsedOutputPath(root, "/docs/after-stop.txt");

			expect(existsSync(afterStopMirror)).toBe(false);
			// No refresh ran after stop, so the last-finished timestamp is unchanged.
			expect((await agent.getRefreshStatus()).lastFinishedAt).toBe(finishedAt);
		},
	);
});
