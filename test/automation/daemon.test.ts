import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Automation, AutomationEvent } from "../../src/automation/config.ts";
import {
	AutomationDaemonAlreadyRunningError,
	type AutomationSubscription,
	type AutomationWatcherBackend,
	isAutomationDaemonRunning,
	startAutomationDaemon,
} from "../../src/automation/daemon.ts";
import type { AutomationRunRecord } from "../../src/automation/run.ts";

let root: string;
let home: string;
let inbox: string;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-automation-daemon-"));
	home = join(root, "home");
	inbox = join(root, "inbox");
	mkdirSync(home, { recursive: true });
	mkdirSync(inbox, { recursive: true });
});

afterEach(() => {
	vi.useRealTimers();
	rmSync(root, { recursive: true, force: true });
});

function writeAutomations(automations: unknown[]): void {
	writeFileSync(join(home, "automations.json"), JSON.stringify({ automations }), "utf8");
}

function fakeBackend() {
	const handlers = new Map<string, Set<(events: readonly AutomationEvent[]) => void>>();
	let activeCount = 0;
	const backend: AutomationWatcherBackend = async (dir, onEvents): Promise<AutomationSubscription> => {
		let set = handlers.get(dir);
		if (!set) {
			set = new Set();
			handlers.set(dir, set);
		}
		set.add(onEvents);
		activeCount++;
		return {
			unsubscribe: async () => {
				if (set?.delete(onEvents)) activeCount--;
			},
		};
	};
	return {
		backend,
		activeCount: () => activeCount,
		emit(dir: string, events: readonly AutomationEvent[]): void {
			for (const fn of handlers.get(dir) ?? []) fn(events);
		},
	};
}

function okRecord(id: string, events: readonly AutomationEvent[]): AutomationRunRecord {
	return {
		id,
		startedAt: new Date().toISOString(),
		durationMs: 1,
		eventCount: events.length,
		filesOverflow: false,
		exitCode: 0,
		signal: null,
		timedOut: false,
		stdout: "",
		stderr: "",
	};
}

describe("startAutomationDaemon (unit)", () => {
	it("debounces bursts, coalesces create+update to create, and drops create+delete transients", async () => {
		vi.useFakeTimers();
		writeAutomations([
			{
				id: "pdf",
				paths: [inbox],
				events: ["create"],
				include: ["**/*.pdf"],
				debounceMs: 200,
				run: ["node", "-e", "0"],
			},
		]);
		const fb = fakeBackend();
		const runs: Array<{ id: string; events: readonly AutomationEvent[] }> = [];
		const daemon = await startAutomationDaemon({
			home,
			watcher: fb.backend,
			runner: async (a, evs) => {
				runs.push({ id: a.id, events: evs });
				return okRecord(a.id, evs);
			},
		});

		try {
			const keep = join(inbox, "keep.pdf");
			const transient = join(inbox, "transient.pdf");
			const ignored = join(inbox, "note.txt");
			fb.emit(inbox, [
				{ type: "create", path: keep },
				{ type: "update", path: keep },
				{ type: "create", path: transient },
				{ type: "delete", path: transient },
				{ type: "create", path: ignored },
			]);

			await vi.advanceTimersByTimeAsync(199);
			expect(runs).toHaveLength(0);
			await vi.advanceTimersByTimeAsync(1);
			expect(runs).toEqual([{ id: "pdf", events: [{ type: "create", path: keep }] }]);
		} finally {
			await daemon.stop();
		}
	});

	it("runs at most one instance per automation in flight and batches events that arrive mid-run", async () => {
		vi.useFakeTimers();
		writeAutomations([{ id: "a", paths: [inbox], debounceMs: 100, run: ["node", "-e", "0"] }]);
		const fb = fakeBackend();
		let releaseFirst!: () => void;
		const firstGate = new Promise<void>((r) => {
			releaseFirst = r;
		});
		const batches: Array<readonly AutomationEvent[]> = [];
		let call = 0;

		const daemon = await startAutomationDaemon({
			home,
			watcher: fb.backend,
			runner: async (a, evs) => {
				batches.push(evs);
				if (++call === 1) await firstGate;
				return okRecord(a.id, evs);
			},
		});

		try {
			fb.emit(inbox, [{ type: "create", path: join(inbox, "1.pdf") }]);
			await vi.advanceTimersByTimeAsync(100);
			expect(batches).toHaveLength(1);

			fb.emit(inbox, [{ type: "create", path: join(inbox, "2.pdf") }]);
			fb.emit(inbox, [{ type: "create", path: join(inbox, "3.pdf") }]);
			await vi.advanceTimersByTimeAsync(500);
			expect(batches).toHaveLength(1);

			releaseFirst();
			await vi.advanceTimersByTimeAsync(100);
			expect(batches).toHaveLength(2);
			expect(batches[1]?.map((e) => e.path)).toEqual([join(inbox, "2.pdf"), join(inbox, "3.pdf")]);
		} finally {
			await daemon.stop();
		}
	});

	it("trips the runaway loop guard after maxRunsPerMinute and re-arms on reload", async () => {
		vi.useFakeTimers();
		writeAutomations([{ id: "loop", paths: [inbox], debounceMs: 10, run: ["node", "-e", "0"] }]);
		const fb = fakeBackend();
		const errors: string[] = [];
		let count = 0;

		const daemon = await startAutomationDaemon({
			home,
			watcher: fb.backend,
			maxRunsPerMinute: 3,
			onError: (msg) => errors.push(msg),
			runner: async (a, evs) => {
				count++;
				return okRecord(a.id, evs);
			},
		});

		try {
			for (let i = 0; i < 5; i++) {
				fb.emit(inbox, [{ type: "update", path: join(inbox, "self.txt") }]);
				await vi.advanceTimersByTimeAsync(10);
			}
			expect(count).toBe(3);
			expect(errors.some((e) => e.includes("loop") && e.includes("disabled"))).toBe(true);

			await daemon.reload();
			fb.emit(inbox, [{ type: "update", path: join(inbox, "self.txt") }]);
			await vi.advanceTimersByTimeAsync(10);
			expect(count).toBe(4);
		} finally {
			await daemon.stop();
		}
	});

	it("enforces a single daemon instance per home and keeps previous rules when reload hits invalid JSON", async () => {
		writeAutomations([{ id: "a", paths: [inbox], run: ["node", "-e", "0"] }]);
		const fb = fakeBackend();
		const errors: string[] = [];
		const daemon = await startAutomationDaemon({ home, watcher: fb.backend, onError: (m) => errors.push(m) });

		try {
			expect(isAutomationDaemonRunning(home)).toBe(true);
			await expect(startAutomationDaemon({ home, watcher: fb.backend })).rejects.toThrow(
				AutomationDaemonAlreadyRunningError,
			);

			writeFileSync(join(home, "automations.json"), "{ broken", "utf8");
			await daemon.reload();
			expect(errors.some((e) => e.includes("automations.json"))).toBe(true);
			expect(fb.activeCount()).toBe(1);
		} finally {
			await daemon.stop();
		}
		expect(isAutomationDaemonRunning(home)).toBe(false);
	});
});

describe("startAutomationDaemon (real @parcel/watcher)", () => {
	it("watches a real folder and runs the configured script when a matching file is created", async () => {
		const outFile = join(root, "out.json");
		const script = join(root, "worker.mjs");
		writeFileSync(
			script,
			`import { writeFileSync } from "node:fs";
let s = "";
for await (const c of process.stdin) s += c;
writeFileSync(${JSON.stringify(outFile)}, s);
`,
		);
		writeAutomations([
			{
				id: "real",
				paths: [inbox],
				events: ["create"],
				include: ["**/*.pdf"],
				debounceMs: 50,
				run: ["node", script],
			},
		]);

		let notifyRun!: (record: AutomationRunRecord) => void;
		const ran = new Promise<AutomationRunRecord>((r) => {
			notifyRun = r;
		});
		const daemon = await startAutomationDaemon({
			home,
			onRun: (_automation: Automation, record: AutomationRunRecord) => notifyRun(record),
		});

		try {
			writeFileSync(join(inbox, "ignore.txt"), "no");
			writeFileSync(join(inbox, "invoice.pdf"), "%PDF-1.7");
			const record = await ran;
			expect(record.exitCode).toBe(0);
			const written = JSON.parse(readFileSync(outFile, "utf8")) as AutomationEvent[];
			expect(written).toHaveLength(1);
			expect(written[0]?.type).toBe("create");
			expect(written[0]?.path.endsWith("invoice.pdf")).toBe(true);
		} finally {
			await daemon.stop();
		}
	});
});
