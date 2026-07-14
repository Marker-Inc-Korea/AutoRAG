import { type ChildProcess, fork } from "node:child_process";
import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, utimesSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { acquireFileLock } from "../../src/filesystem/file-lock.ts";
import { normalizeSessionEvidenceRef, RetrievalMemory } from "../../src/memory/memory.ts";

const MEMORY_PROCESS_FIXTURE = fileURLToPath(new URL("./memory-process-fixture.ts", import.meta.url));

const fsMock = vi.hoisted(() => ({
	realRenameSync: undefined as typeof import("node:fs").renameSync | undefined,
	realRmdirSync: undefined as typeof import("node:fs").rmdirSync | undefined,
	renameSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").renameSync>) => ReturnType<typeof import("node:fs").renameSync>)
		| undefined,
	rmdirSyncHook: undefined as
		| ((...args: Parameters<typeof import("node:fs").rmdirSync>) => ReturnType<typeof import("node:fs").rmdirSync>)
		| undefined,
}));

vi.mock("node:fs", async () => {
	const actual = await vi.importActual<typeof import("node:fs")>("node:fs");
	fsMock.realRenameSync = actual.renameSync;
	fsMock.realRmdirSync = actual.rmdirSync;
	return {
		...actual,
		renameSync: (...args: Parameters<typeof actual.renameSync>) => {
			if (fsMock.renameSyncHook) return fsMock.renameSyncHook(...args);
			return actual.renameSync(...args);
		},
		rmdirSync: (...args: Parameters<typeof actual.rmdirSync>) => {
			if (fsMock.rmdirSyncHook) return fsMock.rmdirSyncHook(...args);
			return actual.rmdirSync(...args);
		},
	};
});

let tmpDir: string;
let memoryPath: string;

interface WorkerMessage {
	readonly type: "ready" | "saved";
	readonly workerId: string;
}

interface MemoryWorker {
	readonly child: ChildProcess;
	readonly ready: Promise<void>;
	readonly saved: Promise<void>;
	readonly exited: Promise<void>;
}

function isWorkerMessage(value: unknown): value is WorkerMessage {
	return (
		typeof value === "object" &&
		value !== null &&
		"type" in value &&
		(value.type === "ready" || value.type === "saved") &&
		"workerId" in value &&
		typeof value.workerId === "string"
	);
}

function waitForWorkerMessage(
	child: ChildProcess,
	expectedType: WorkerMessage["type"],
	readStderr: () => string,
): Promise<void> {
	return new Promise((resolve, reject) => {
		const timeout = setTimeout(() => {
			cleanup();
			reject(new Error(`Timed out waiting for child message ${expectedType}: ${readStderr()}`));
		}, 5_000);
		const onMessage = (message: unknown): void => {
			if (!isWorkerMessage(message) || message.type !== expectedType) return;
			cleanup();
			resolve();
		};
		const onError = (error: Error): void => {
			cleanup();
			reject(error);
		};
		const onExit = (code: number | null, signal: NodeJS.Signals | null): void => {
			cleanup();
			reject(new Error(`Child exited before ${expectedType}: code=${code} signal=${signal} ${readStderr()}`));
		};
		const cleanup = (): void => {
			clearTimeout(timeout);
			child.off("message", onMessage);
			child.off("error", onError);
			child.off("exit", onExit);
		};
		child.on("message", onMessage);
		child.once("error", onError);
		child.once("exit", onExit);
	});
}

function spawnMemoryWorker(workerId: string): MemoryWorker {
	const child = fork(MEMORY_PROCESS_FIXTURE, [memoryPath, workerId], {
		execArgv: ["--experimental-strip-types", "--disable-warning=ExperimentalWarning"],
		silent: true,
	});
	let stderr = "";
	child.stderr?.on("data", (chunk: Buffer | string) => {
		stderr += String(chunk);
	});
	const readStderr = (): string => stderr;
	const exited = new Promise<void>((resolve, reject) => {
		child.once("error", reject);
		child.once("exit", (code, signal) => {
			if (code === 0) resolve();
			else reject(new Error(`Child failed: code=${code} signal=${signal} ${stderr}`));
		});
	});
	return {
		child,
		ready: waitForWorkerMessage(child, "ready", readStderr),
		saved: waitForWorkerMessage(child, "saved", readStderr),
		exited,
	};
}

async function stopMemoryWorker(worker: MemoryWorker): Promise<void> {
	if (worker.child.exitCode !== null || worker.child.signalCode !== null) return;
	await new Promise<void>((resolve) => {
		worker.child.once("exit", () => resolve());
		worker.child.kill();
	});
}

beforeEach(() => {
	tmpDir = join(tmpdir(), `autorag-memory-test-${Date.now()}`);
	mkdirSync(tmpDir, { recursive: true });
	memoryPath = join(tmpDir, "memory.json");
});

afterEach(() => {
	fsMock.renameSyncHook = undefined;
	fsMock.rmdirSyncHook = undefined;
	vi.restoreAllMocks();
	rmSync(tmpDir, { recursive: true, force: true });
});

function recordSession(memory: RetrievalMemory): void {
	memory.recordCuratedResultsSession({
		sessionId: "s1",
		query: "typescript handbook",
		results: [
			{
				number: 1,
				title: "Handbook",
				summary: "TypeScript handbook summary",
				content: "TypeScript handbook content",
				method: "posix",
				source: "/docs/handbook.md",
				evidenceRefs: [
					normalizeSessionEvidenceRef({
						method: "posix",
						source: "/docs/handbook.md",
						excerpt: "TypeScript handbook content",
						lineNumber: 4,
					}),
				],
			},
		],
	});
}

describe("RetrievalMemory", () => {
	it("starts with empty v4 state when file does not exist", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getSchema().version).toBe(4);
		expect(memory.getMethodHints("test query")).toEqual([]);
		expect(memory.getEntries()).toEqual([]);
	});

	it("records explicit method feedback as advisory hints", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("search code files", "posix", true);
		memory.recordFeedback("search code files", "vector", false);
		const hints = memory.getMethodHints("search code files");
		expect(hints[0].method).toBe("posix");
		expect(hints[0].score).toBeGreaterThan(0);
		expect(hints.find((hint) => hint.method === "vector")?.score).toBeLessThan(0);
		expect(hints[0].reason).toContain("advisory");
	});

	it("persists v4 data to disk with save()", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript functions", "posix", true);
		memory.save();
		expect(existsSync(memoryPath)).toBe(true);
		const raw = JSON.parse(readFileSync(memoryPath, "utf-8"));
		expect(raw.version).toBe(4);
		expect(raw.feedbackSignals).toHaveLength(1);
	});

	it("merges feedback and session results saved by independent processes", { timeout: 10_000 }, async () => {
		const workers = [spawnMemoryWorker("alpha"), spawnMemoryWorker("beta")];
		try {
			await Promise.all(workers.map((worker) => worker.ready));
			for (const worker of workers) worker.child.send("save");
			await Promise.all(workers.map((worker) => worker.saved));
			await Promise.all(workers.map((worker) => worker.exited));

			const raw = JSON.parse(readFileSync(memoryPath, "utf-8"));
			expect(raw.feedbackSignals).toHaveLength(2);
			expect(raw.feedbackSignals.map((signal: { query: string }) => signal.query).sort()).toEqual([
				"feedback-alpha",
				"feedback-beta",
			]);
			expect(raw.curatedResults).toHaveLength(2);
			expect(raw.curatedResults.map((result: { sessionId: string }) => result.sessionId).sort()).toEqual([
				"session-alpha",
				"session-beta",
			]);
			expect(raw.evidenceChunks).toHaveLength(2);
			expect(readdirSync(tmpDir).filter((name) => name.includes(".tmp") || name.includes(".lock"))).toEqual([]);
		} finally {
			await Promise.all(workers.map(stopMemoryWorker));
		}
	});

	it("uses a unique temporary path for each save attempt", () => {
		const tempPaths: string[] = [];
		fsMock.renameSyncHook = (...args) => {
			if (String(args[1]) === memoryPath) tempPaths.push(String(args[0]));
			return fsMock.realRenameSync?.(...args);
		};
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("first save", "posix", true);
		memory.save();
		memory.recordFeedback("second save", "posix", true);
		memory.save();

		expect(tempPaths).toHaveLength(2);
		expect(new Set(tempPaths).size).toBe(2);
		expect(tempPaths.every((path) => path.startsWith(`${memoryPath}.`))).toBe(true);
		expect(tempPaths.every((path) => path.endsWith(".tmp"))).toBe(true);
	});

	it("cleans a unique temporary file after a failed rename without replacing existing memory", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("existing memory", "posix", true);
		memory.save();
		const existingMemory = readFileSync(memoryPath, "utf-8");
		fsMock.renameSyncHook = (...args) => {
			if (String(args[1]) === memoryPath) throw new Error("rename failed");
			return fsMock.realRenameSync?.(...args);
		};

		memory.recordFeedback("new memory", "posix", true);
		expect(() => memory.save()).toThrow("rename failed");
		expect(readFileSync(memoryPath, "utf-8")).toBe(existingMemory);
		expect(readdirSync(tmpDir).filter((name) => name.endsWith(".tmp") || name.includes(".lock"))).toEqual([]);
	});

	it("reclaims an abandoned stale lock and removes its cleanup artifacts", () => {
		const lockPath = `${memoryPath}.lock`;
		writeFileSync(lockPath, JSON.stringify({ token: "abandoned", pid: 999_999, createdAt: 0 }), "utf-8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(lockPath, staleTime, staleTime);
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("after stale lock", "posix", true);

		expect(() => memory.save()).not.toThrow();
		expect(JSON.parse(readFileSync(memoryPath, "utf-8")).feedbackSignals).toHaveLength(1);
		expect(readdirSync(tmpDir).filter((name) => name.includes(".lock") || name.endsWith(".tmp"))).toEqual([]);
	});

	it("keeps a fresh owner valid while merging its update during stale-lock turnover", () => {
		const ownerPath = join(tmpDir, "turnover-owner.json");
		const ownerMemory = new RetrievalMemory({ storagePath: ownerPath });
		ownerMemory.load();
		ownerMemory.recordFeedback("fresh owner update", "posix", true);
		ownerMemory.save();
		const ownerBytes = readFileSync(ownerPath);
		rmSync(ownerPath, { force: true });

		const lockPath = `${memoryPath}.lock`;
		const staleContents = `${JSON.stringify({ token: "stale-owner", pid: 999_999, createdAt: 0 })}\n`;
		mkdirSync(lockPath, { mode: 0o700 });
		const staleMarkerPath = join(lockPath, "owner-stale-owner.json");
		writeFileSync(staleMarkerPath, staleContents, "utf-8");
		const staleTime = new Date(Date.now() - 60_000);
		utimesSync(staleMarkerPath, staleTime, staleTime);
		let turnoverInjected = false;
		let freshOwnerAssertions = 0;
		let freshOwnerCommitted = false;
		let staleReaperBlocked = false;
		let competingOwnerRejected = false;
		fsMock.rmdirSyncHook = (...args) => {
			if (!turnoverInjected && String(args[0]) === lockPath) {
				turnoverInjected = true;
				if (!fsMock.realRmdirSync) throw new Error("real rmdirSync is unavailable");
				fsMock.realRmdirSync(...args);
				const freshOwner = acquireFileLock(lockPath, {
					timeoutMs: 1_000,
					staleMs: 30_000,
					retryMs: 1,
					timeoutError: () => new Error("fresh turnover owner could not acquire the memory lock"),
				});
				let delayedReaperError: unknown;
				try {
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					try {
						fsMock.realRmdirSync(lockPath);
					} catch (error) {
						if (
							!(
								error instanceof Error &&
								"code" in error &&
								(error.code === "ENOTEMPTY" || error.code === "EEXIST")
							)
						) {
							throw error;
						}
						staleReaperBlocked = true;
						delayedReaperError = error;
					}
					freshOwner.assertOwned();
					freshOwnerAssertions++;
					expect(() =>
						acquireFileLock(lockPath, {
							timeoutMs: 0,
							staleMs: 30_000,
							retryMs: 1,
							timeoutError: () => {
								competingOwnerRejected = true;
								return new Error("turnover competitor could not acquire the memory lock");
							},
						}),
					).toThrow("turnover competitor could not acquire the memory lock");
					writeFileSync(memoryPath, ownerBytes);
					freshOwnerCommitted = true;
				} finally {
					freshOwner.release();
				}
				if (delayedReaperError !== undefined) throw delayedReaperError;
				throw new Error("delayed stale reaper unexpectedly removed the fresh memory lock");
			}
			return fsMock.realRmdirSync?.(...args);
		};

		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("contending update", "minsync", true);
		memory.save();

		expect(turnoverInjected).toBe(true);
		expect(freshOwnerAssertions).toBe(2);
		expect(freshOwnerCommitted).toBe(true);
		expect(staleReaperBlocked).toBe(true);
		expect(competingOwnerRejected).toBe(true);
		const persisted = JSON.parse(readFileSync(memoryPath, "utf-8")) as {
			feedbackSignals: Array<{ query: string }>;
		};
		expect(persisted.feedbackSignals.map((signal) => signal.query).sort()).toEqual([
			"contending update",
			"fresh owner update",
		]);
		expect(readdirSync(tmpDir).filter((name) => name.includes(".lock") || name.includes(".quarantine"))).toEqual([]);
	});

	it("bounds waiting for a live lock without deleting another process's lock", () => {
		const lockPath = `${memoryPath}.lock`;
		const realNow = Date.now();
		writeFileSync(lockPath, JSON.stringify({ token: "live", pid: process.pid, createdAt: realNow }), "utf-8");
		let clock = realNow;
		vi.spyOn(Date, "now").mockImplementation(() => {
			clock += 20_000;
			return clock;
		});
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("blocked save", "posix", true);

		expect(() => memory.save()).toThrow("Timed out waiting for retrieval memory lock");
		expect(existsSync(lockPath)).toBe(true);
		expect(readdirSync(tmpDir).filter((name) => name.endsWith(".tmp") || name.includes(".stale"))).toEqual([]);
	});

	it("loads persisted v4 data after restart", () => {
		const memory1 = new RetrievalMemory({ storagePath: memoryPath });
		memory1.load();
		memory1.recordFeedback("code search query", "posix", true);
		memory1.recordFeedback("code search query", "posix", true);
		memory1.save();

		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		const hints = memory2.getMethodHints("code search query");
		expect(hints[0].method).toBe("posix");
		expect(hints[0].score).toBeGreaterThan(0);
	});

	it("resets corrupted memory file with non-path warning", () => {
		writeFileSync(memoryPath, "not valid json {{{", "utf-8");
		const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		expect(() => memory.load()).not.toThrow();
		expect(memory.getSchema().version).toBe(4);
		expect(memory.getSchema().warnings[0].message).not.toContain("/");
		expect(warn).toHaveBeenCalledWith("[AutoRAG] Retrieval memory is not v4-compatible; starting fresh");
	});

	it("resets v1-v3 memory instead of migrating", () => {
		writeFileSync(memoryPath, JSON.stringify({ version: 3, entries: [{ query: "q", method: "posix" }] }), "utf-8");
		vi.spyOn(console, "warn").mockImplementation(() => {});
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getSchema().version).toBe(4);
		expect(memory.getEntries()).toEqual([]);
		expect(memory.getSignalCount()).toBe(0);
		expect(memory.getSchema().warnings[0].code).toBe("memory-reset");
	});

	it("records curated result and evidence records for a session", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		recordSession(memory);
		const schema = memory.getSchema();
		expect(schema.curatedResults).toHaveLength(1);
		expect(schema.evidenceChunks).toHaveLength(1);
		expect(schema.curatedResults[0].evidenceIds).toEqual([schema.evidenceChunks[0].stableEvidenceId]);
		expect(schema.evidenceChunks[0].method).toBe("posix");
	});

	it("distributes explicit numbered feedback to result and evidence without full-strength double-counting", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		recordSession(memory);
		expect(
			memory.recordNumberedFeedback({
				sessionId: "s1",
				query: "typescript handbook",
				feedback: [{ number: 1, useful: true }],
			}),
		).toBe(true);
		const signals = memory.getSchema().feedbackSignals;
		expect(signals).toHaveLength(2);
		expect(signals[0].target.type).toBe("curated_result");
		expect(signals[0].weight).toBe(1);
		expect(signals[1].target.type).toBe("evidence_chunk");
		expect(signals[1].weight).toBe(1);
		expect(memory.getMethodHints("typescript handbook")[0].score).toBe(1);
	});

	it("does not duplicate repeated feedback for the same result and sentiment", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		recordSession(memory);
		memory.recordNumberedFeedback({
			sessionId: "s1",
			query: "typescript handbook",
			feedback: [{ number: 1, useful: true }],
		});
		memory.recordNumberedFeedback({
			sessionId: "s1",
			query: "typescript handbook",
			feedback: [{ number: 1, useful: true }],
		});
		expect(memory.getSchema().feedbackSignals).toHaveLength(2);
		expect(memory.getMethodHints("typescript handbook")[0].score).toBe(1);
	});

	it("recomputes caller-provided path-like stable evidence IDs", () => {
		const ref = normalizeSessionEvidenceRef({
			method: "grep",
			source: "/docs/a.md",
			content: "safe content",
			stableEvidenceId: "/Users/me/docs/a.md:1",
		});
		expect(ref.stableEvidenceId).toMatch(/^grep:[0-9a-f]{24}$/u);
		expect(ref.stableEvidenceId).not.toContain("/Users");
		const driveRef = normalizeSessionEvidenceRef({
			method: "grep",
			source: "/docs/a.md",
			content: "safe content",
			stableEvidenceId: "C:docs-file",
		});
		expect(driveRef.stableEvidenceId).toMatch(/^grep:[0-9a-f]{24}$/u);
	});

	it("splits evidence signal weight across multiple evidence chunks", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordCuratedResultsSession({
			sessionId: "s2",
			query: "q",
			results: [
				{
					number: 1,
					title: "T",
					summary: "S",
					content: "C",
					method: "grep",
					source: "/a",
					evidenceRefs: [
						normalizeSessionEvidenceRef({ method: "grep", source: "/a", excerpt: "one" }),
						normalizeSessionEvidenceRef({ method: "grep", source: "/b", excerpt: "two" }),
					],
				},
			],
		});
		memory.recordNumberedFeedback({ sessionId: "s2", query: "q", feedback: [{ number: 1, useful: false }] });
		const evidenceSignals = memory
			.getSchema()
			.feedbackSignals.filter((signal) => signal.target.type === "evidence_chunk");
		expect(evidenceSignals.map((signal) => signal.weight)).toEqual([-0.5, -0.5]);
	});

	it("advisory hints remain fallback-eligible for negative methods", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.recordFeedback("find typescript files", "posix", true);
		memory.recordFeedback("find typescript files", "minsync", false);
		const methods = memory.getMethodHints("typescript files").map((hint) => hint.method);
		expect(methods).toContain("posix");
		expect(methods).toContain("minsync");
	});

	it("compat append/registerAttempt resolves in-memory pending entries", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		const entry = memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.registerAttempt({
			id: entry.id,
			query: "q",
			method: "posix",
			sources: ["file.ts"],
			timestamp: Date.now(),
		});
		memory.recordResultFeedback([{ source: "file.ts", useful: true }]);
		expect(memory.getEntries().find((e) => e.id === entry.id)?.outcome).toBe("useful");
	});

	it("compat pending entries are not persisted as legacy v3 state", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		memory.append({ query: "q", method: "posix", outcome: "pending" });
		memory.save();
		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		expect(memory2.getEntries()).toEqual([]);
		expect(memory2.getSchema().version).toBe(4);
	});

	it("caps v4 feedback signals at 500", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 510; i++) memory.recordFeedback(`query-${i}`, "posix", true);
		memory.save();
		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		expect(memory2.getSignalCount()).toBe(500);
	});

	it("normalizes current v4 files without insights", () => {
		writeFileSync(
			memoryPath,
			JSON.stringify({
				version: 4,
				curatedResults: [],
				evidenceChunks: [],
				feedbackSignals: [],
				signalDefaults: { explicitWeight: 1, followupWeight: 0.25, retryWeight: -0.25, implicitCap: 0.5 },
				warnings: [],
			}),
			"utf-8",
		);
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		expect(memory.getSchema().insights).toEqual([]);
	});

	it("extracts durable insights from complete 100-signal evicted batches", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 600; i++) memory.recordFeedback("photo archive lookup", "posix", true);
		memory.save();

		const memory2 = new RetrievalMemory({ storagePath: memoryPath });
		memory2.load();
		expect(memory2.getSignalCount()).toBe(500);
		const insights = memory2.getInsights("photo archive lookup");
		expect(insights).toHaveLength(1);
		expect(insights[0].domain).toBe("photo archive lookup");
		expect(insights[0].recommendedMethods).toEqual(["posix"]);
		expect(insights[0].supportingSignalCount).toBe(100);
	});

	it("accumulates evicted insight batches across incremental saves", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 500; i++) memory.recordFeedback("photo archive lookup", "posix", true);
		memory.save();

		for (let i = 0; i < 99; i++) {
			memory.recordFeedback("photo archive lookup", "posix", true);
			memory.save();
		}
		expect(memory.getSignalCount()).toBe(500);
		expect(memory.getInsights("photo archive lookup")).toEqual([]);

		memory.recordFeedback("photo archive lookup", "posix", true);
		memory.save();
		expect(memory.getSignalCount()).toBe(500);
		const insights = memory.getInsights("photo archive lookup");
		expect(insights).toHaveLength(1);
		expect(insights[0].supportingSignalCount).toBe(100);
	});

	it("does not create insights from under-sized or noisy evictions", () => {
		const undersized = new RetrievalMemory({ storagePath: memoryPath });
		undersized.load();
		for (let i = 0; i < 510; i++) undersized.recordFeedback("photo archive lookup", "posix", true);
		undersized.save();
		expect(undersized.getSchema().feedbackSignals).toHaveLength(500);
		expect(undersized.getInsights("photo archive lookup")).toEqual([]);
		expect(undersized.getSchema().pendingInsightSignals).toHaveLength(10);

		const noisyPath = join(tmpDir, "noisy-memory.json");
		const noisy = new RetrievalMemory({ storagePath: noisyPath });
		noisy.load();
		for (let i = 0; i < 600; i++) noisy.recordWeakSignal("weak photo lookup", "posix", "followup");
		noisy.save();
		expect(noisy.getSignalCount()).toBe(500);
		expect(noisy.getInsights("weak photo lookup")).toEqual([]);
	});

	it("merges repeated insight batches instead of duplicating them", () => {
		const memory = new RetrievalMemory({ storagePath: memoryPath });
		memory.load();
		for (let i = 0; i < 600; i++) memory.recordFeedback("insurance claim forms", "minsync", true);
		memory.save();
		for (let i = 0; i < 100; i++) memory.recordFeedback("insurance claim forms", "minsync", true);
		memory.save();

		const insights = memory.getInsights("insurance claim forms");
		expect(insights).toHaveLength(1);
		expect(insights[0].supportingSignalCount).toBe(200);
		expect(insights[0].recommendedMethods).toEqual(["minsync"]);
	});

	it("keeps save fail-open when insight extraction fails", () => {
		const memory = new RetrievalMemory({
			storagePath: memoryPath,
			insightExtractor: () => {
				throw new Error("extractor failed");
			},
		});
		memory.load();
		for (let i = 0; i < 600; i++) memory.recordFeedback("photo archive lookup", "posix", true);
		expect(() => memory.save()).not.toThrow();
		expect(memory.getSignalCount()).toBe(500);
		expect(memory.getSchema().warnings.some((warning) => warning.code === "insight-extraction-failed")).toBe(true);
	});
});
