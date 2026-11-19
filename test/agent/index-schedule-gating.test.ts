import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import * as power from "../../src/agent/index-schedule.ts";
import type { ParsedMirrorSyncResult } from "../../src/mirror/sync.ts";

const summary: ParsedMirrorSyncResult = {
	scanned: 0,
	written: 0,
	deleted: 0,
	skipped: 0,
	indexPath: "i",
	diagnostics: [],
};
let tmpDir: string;

function makeAgent(schedule?: power.IndexSchedule): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: ["test/fixtures/sample-project"],
		memoryPath: join(tmpDir, "memory.json"),
		workspacePath: tmpDir,
		everything: false,
		fsearch: false,
		...(schedule ? { refresh: { schedule } } : {}),
	});
}

beforeEach(() => {
	vi.useFakeTimers();
	tmpDir = mkdtempSync(join(tmpdir(), "autorag-schedule-"));
});
afterEach(() => {
	vi.useRealTimers();
	vi.restoreAllMocks();
	rmSync(tmpDir, { recursive: true, force: true });
});

describe("background refresh honours the index schedule", () => {
	it("auto-refresh skips ticks on battery with power=ac and resumes on AC", async () => {
		const source = vi.spyOn(power, "detectPowerSource").mockResolvedValue("battery");
		const agent = makeAgent({ power: "ac" });
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(summary);
		agent.startAutoRefresh(1000);
		await vi.advanceTimersByTimeAsync(2000);
		expect(refresh).not.toHaveBeenCalled();
		source.mockResolvedValue("ac");
		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);
		agent.stopAutoRefresh();
	});

	it("explicit refresh() is never gated", async () => {
		vi.spyOn(power, "detectPowerSource").mockResolvedValue("battery");
		const agent = makeAgent({ power: "ac", windows: ["03:00-03:01"] });
		const decision = await agent.indexScheduleDecision();
		expect(decision.allowed).toBe(false);
		await expect(agent.refresh(false, { methods: ["parsed"] })).resolves.toBeDefined();
	});

	it("watcher defers a change while closed and refreshes once the schedule opens", async () => {
		const source = vi.spyOn(power, "detectPowerSource").mockResolvedValue("battery");
		const agent = makeAgent({ power: "ac" });
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(summary);
		let emit: (f: string | null) => void = () => {};
		const handle = agent.startWatchRefresh({
			debounceMs: 10,
			scheduleRecheckMs: 1000,
			watcherFactory: (_dir, onChange) => {
				emit = onChange;
				return { close() {} };
			},
		});
		emit("a.md");
		await vi.advanceTimersByTimeAsync(50);
		expect(refresh).not.toHaveBeenCalled();
		await vi.advanceTimersByTimeAsync(2000);
		expect(refresh).not.toHaveBeenCalled();
		source.mockResolvedValue("ac");
		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).toHaveBeenCalledTimes(1);
		await vi.advanceTimersByTimeAsync(5000);
		expect(refresh).toHaveBeenCalledTimes(1);
		handle.stop();
	});

	it("stop() cancels the deferred recheck", async () => {
		vi.spyOn(power, "detectPowerSource").mockResolvedValue("battery");
		const agent = makeAgent({ power: "ac" });
		const refresh = vi.spyOn(agent, "refresh").mockResolvedValue(summary);
		const handle = agent.startWatchRefresh({
			debounceMs: 10,
			scheduleRecheckMs: 100,
			watcherFactory: () => ({ close() {} }),
		});
		handle.trigger();
		await vi.advanceTimersByTimeAsync(50);
		handle.stop();
		vi.spyOn(power, "detectPowerSource").mockResolvedValue("ac");
		await vi.advanceTimersByTimeAsync(1000);
		expect(refresh).not.toHaveBeenCalled();
	});

	it("rejects an invalid schedule at construction", () => {
		expect(() => makeAgent({ windows: ["bad"] })).toThrow(/windows/);
	});
});
