import type * as Fs from "node:fs";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { readRefreshProgress } from "../../src/agent/refresh-progress.ts";
import { AutoRAGAgent } from "../../src/index.ts";

const PROGRESS_FILE = /refresh-progress\.json$/;
const PROGRESS_TEMP_FILE = /refresh-progress\.json\.[^/\\]+\.tmp$/;

const fsMock = vi.hoisted(() => ({
	renameHook: undefined as ((from: string, to: string) => void) | undefined,
	writeFileSyncHook: undefined as ((path: string) => void) | undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof Fs>("node:fs");
	return {
		...actual,
		renameSync: (from: Fs.PathLike, to: Fs.PathLike) => {
			fsMock.renameHook?.(String(from), String(to));
			return actual.renameSync(from, to);
		},
		writeFileSync: (...args: Parameters<typeof actual.writeFileSync>) => {
			fsMock.writeFileSyncHook?.(String(args[0]));
			return actual.writeFileSync(...args);
		},
	};
});

let root: string;
let docs: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-refresh-progress-write-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
});

afterEach(() => {
	fsMock.renameHook = undefined;
	fsMock.writeFileSyncHook = undefined;
	rmSync(root, { recursive: true, force: true });
});

function makeAgent(): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		jikji: false,
		everything: false,
		fsearch: false,
		minSync: { autoInstall: false },
	});
}

function writeNotes(count: number): void {
	for (let index = 0; index < count; index += 1) writeFileSync(join(docs, `note-${index}.md`), `# Note ${index}\n`);
}

function errno(code: string): NodeJS.ErrnoException {
	const error: NodeJS.ErrnoException = new Error(`${code}: injected failure`);
	error.code = code;
	return error;
}

describe("refresh-progress.json writes", () => {
	it("does not rewrite the marker once per source file", async () => {
		writeNotes(60);
		let progressRenames = 0;
		fsMock.renameHook = (_from, to) => {
			if (PROGRESS_FILE.test(to)) progressRenames += 1;
		};

		await makeAgent().refresh(true, { methods: ["parsed"] });

		// Start + a handful of per-phase writes + one per 25-file checkpoint (0, 25, 50, 60).
		expect(progressRenames).toBeLessThanOrEqual(15);
		expect(readRefreshProgress(root)).toMatchObject({
			state: "success",
			parsedCounts: { scanned: 60, written: 60 },
		});
	});

	it("keeps the running counts observable at the checkpoint cadence", async () => {
		writeNotes(60);
		const persistedScanned: number[] = [];
		await makeAgent().refresh(true, {
			methods: ["parsed"],
			onProgress: (progress) => {
				if (progress.phase === "parsed")
					persistedScanned.push(readRefreshProgress(root)?.parsedCounts?.scanned ?? -1);
			},
		});

		expect(persistedScanned).toHaveLength(61);
		expect(persistedScanned[0]).toBe(0);
		expect(persistedScanned[24]).toBe(0);
		expect(persistedScanned[25]).toBe(25);
		expect(persistedScanned[49]).toBe(25);
		expect(persistedScanned[50]).toBe(50);
		expect(persistedScanned[60]).toBe(60);
	});

	it("completes and reports success when the marker rename hits a held destination", async () => {
		writeNotes(3);
		let attempts = 0;
		fsMock.renameHook = (_from, to) => {
			if (!PROGRESS_FILE.test(to)) return;
			attempts += 1;
			// First write exhausts every retry (in-place fallback); the second recovers on its third rename.
			if (attempts <= 8) throw errno("EPERM");
		};

		await makeAgent().refresh(true, { methods: ["parsed"] });

		expect(attempts).toBeGreaterThan(8);
		expect(readRefreshProgress(root)).toMatchObject({ state: "success", parsedCounts: { scanned: 3 } });
	});

	it("records state:failed when the very first marker write fails", async () => {
		writeNotes(3);
		let failed = false;
		fsMock.writeFileSyncHook = (path) => {
			if (failed || !PROGRESS_TEMP_FILE.test(path)) return;
			failed = true;
			throw errno("ENOSPC");
		};
		const agent = makeAgent();

		await expect(agent.refresh(true, { methods: ["parsed"] })).rejects.toMatchObject({ code: "ENOSPC" });

		expect(readRefreshProgress(root)).toMatchObject({ state: "failed", phase: "parsed" });
		expect((await agent.getRefreshStatus()).inFlight).toBe(false);
	});
});
