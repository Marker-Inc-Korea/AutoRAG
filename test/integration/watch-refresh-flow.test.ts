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

describe("AutoRAGAgent watch refresh", () => {
	it("updates parsed mirrors when a watched source file is created, then stops cleanly", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [docs],
			memoryPath: join(root, "memory.json"),
			workspacePath: root,
		});
		await agent.refresh(true);

		let emitChange: ((filename: string | null) => void) | undefined;
		let resolveRefreshCompleted: (() => void) | undefined;
		const refreshCompleted = new Promise<void>((resolve) => {
			resolveRefreshCompleted = resolve;
		});
		const originalRefresh = agent.refresh.bind(agent);
		agent.refresh = async (...args) => {
			const result = await originalRefresh(...args);
			resolveRefreshCompleted?.();
			return result;
		};
		handle = agent.startWatchRefresh({
			debounceMs: 0,
			watcherFactory: (_dir, onChange) => {
				emitChange = onChange;
				return { close: () => {} };
			},
		});

		// Create a new source file in the watched directory.
		writeFileSync(join(docs, "new-note.txt"), "Freshly added note about invoices.\n");
		emitChange?.("new-note.txt");
		const mirrorPath = parsedOutputPath(root, "/docs/new-note.txt");
		let timeout: NodeJS.Timeout | undefined;
		try {
			await Promise.race([
				refreshCompleted,
				new Promise<never>((_resolve, reject) => {
					timeout = setTimeout(() => reject(new Error("Timed out waiting for watch refresh")), 10_000);
				}),
			]);
		} finally {
			if (timeout !== undefined) clearTimeout(timeout);
		}
		expect(existsSync(mirrorPath)).toBe(true);

		// Stop the watcher; further changes must NOT trigger a refresh.
		handle.stop();
		const finishedAt = (await agent.getRefreshStatus()).lastFinishedAt;

		writeFileSync(join(docs, "after-stop.txt"), "Should not be indexed by the watcher.\n");
		emitChange?.("after-stop.txt");
		const afterStopMirror = parsedOutputPath(root, "/docs/after-stop.txt");

		expect(existsSync(afterStopMirror)).toBe(false);
		// No refresh ran after stop, so the last-finished timestamp is unchanged.
		expect((await agent.getRefreshStatus()).lastFinishedAt).toBe(finishedAt);
	});
});
