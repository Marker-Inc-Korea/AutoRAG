import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { ParsedMirrorSyncResult } from "../../src/mirror/sync.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
const fakeSummary: ParsedMirrorSyncResult = {
	scanned: 0,
	written: 0,
	deleted: 0,
	skipped: 0,
	indexPath: "index.json",
	diagnostics: [],
};

let tmpDir: string;
let agent: AutoRAGAgent;

beforeEach(() => {
	vi.useFakeTimers();
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-auto-refresh-test-"));
	agent = new AutoRAGAgent({
		searchPaths: [FIXTURE_DIR],
		memoryPath: join(tmpDir, "memory.json"),
		workspacePath: tmpDir,
	});
});

afterEach(() => {
	agent.stopAutoRefresh();
	vi.useRealTimers();
	vi.restoreAllMocks();
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("AutoRAGAgent auto-refresh scheduler", () => {
	it("calls refresh(false) on each interval tick", async () => {
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(fakeSummary);
		agent.startAutoRefresh(1000);

		await vi.advanceTimersByTimeAsync(1000);

		expect(refresh).toHaveBeenCalledTimes(1);
		expect(refresh).toHaveBeenCalledWith(false);
	});

	it("skips overlapping ticks while a refresh is still in flight", async () => {
		let release: (() => void) | undefined;
		const refresh = vi.spyOn(agent, "refresh").mockImplementation(
			() =>
				new Promise<ParsedMirrorSyncResult>((resolve) => {
					release = () => resolve(fakeSummary);
				}),
		);
		agent.startAutoRefresh(1000);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);

		// Further ticks fire while the first refresh is still pending — they must be skipped.
		await vi.advanceTimersByTimeAsync(3000);
		expect(refresh).toHaveBeenCalledTimes(1);

		release?.();
		await Promise.resolve();
		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(2);
	});

	it("keeps ticking after a refresh rejects", async () => {
		const refresh = vi
			.spyOn(agent, "refresh")
			.mockRejectedValueOnce(new Error("boom"))
			.mockResolvedValue(fakeSummary);
		agent.startAutoRefresh(1000);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(2);
	});

	it("stopAutoRefresh halts further ticks and is idempotent", async () => {
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(fakeSummary);
		agent.startAutoRefresh(1000);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);

		agent.stopAutoRefresh();
		agent.stopAutoRefresh();

		await vi.advanceTimersByTimeAsync(5000);
		expect(refresh).toHaveBeenCalledTimes(1);
	});

	it("runs once immediately when immediate is set", async () => {
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(fakeSummary);
		agent.startAutoRefresh(1000, { immediate: true });

		await Promise.resolve();
		expect(refresh).toHaveBeenCalledTimes(1);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(2);
	});

	it("a second startAutoRefresh replaces the prior interval", async () => {
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(fakeSummary);
		agent.startAutoRefresh(1000);
		agent.startAutoRefresh(2000);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(0);

		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);
	});

	it("the constructor option starts the scheduler", () => {
		const start = vi.spyOn(AutoRAGAgent.prototype, "startAutoRefresh").mockImplementation(() => {});
		const scheduled = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory-2.json"),
			workspacePath: tmpDir,
			autoRefresh: { intervalMs: 5000, immediate: true },
		});

		expect(start).toHaveBeenCalledWith(5000, { immediate: true });
		scheduled.stopAutoRefresh();
		start.mockRestore();
	});
	it("captures a background refresh failure in getRefreshStatus instead of swallowing it (#22)", async () => {
		// Let the real refresh() run its status lifecycle, but force a step to fail.
		vi.spyOn(agent, "syncParsedMirrors").mockRejectedValue(new Error("boom /Users/secret"));
		agent.startAutoRefresh(1000);

		await vi.advanceTimersByTimeAsync(1000);
		// Give the swallowed rejection a tick to settle.
		await Promise.resolve();

		const status = await agent.getRefreshStatus();
		expect(status.state).toBe("failed");
		expect(status.lastError).toBeDefined();
		expect(status.lastError).not.toContain("/Users/");
	});
});
