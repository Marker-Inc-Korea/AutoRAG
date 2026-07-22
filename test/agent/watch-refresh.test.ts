import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createWatchRefresh, type WatchWatcher } from "../../src/agent/watch-refresh.ts";

interface FakeWatcher extends WatchWatcher {
	dir: string;
	emit: (filename: string | null) => void;
	closed: boolean;
}

let watchers: FakeWatcher[];

function factory() {
	return (dir: string, onChange: (filename: string | null) => void): FakeWatcher => {
		const w: FakeWatcher = {
			dir,
			closed: false,
			emit: (filename) => onChange(filename),
			close: () => {
				w.closed = true;
			},
		};
		watchers.push(w);
		return w;
	};
}

beforeEach(() => {
	watchers = [];
	vi.useFakeTimers();
});

afterEach(() => {
	vi.useRealTimers();
});

describe("createWatchRefresh", () => {
	it("creates one watcher per directory", () => {
		createWatchRefresh({
			dirs: ["/a", "/b"],
			debounceMs: 50,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: async () => {},
		});
		expect(watchers.map((w) => w.dir)).toEqual(["/a", "/b"]);
	});

	it("coalesces a burst of events into a single debounced refresh", async () => {
		let calls = 0;
		createWatchRefresh({
			dirs: ["/a"],
			debounceMs: 50,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: async () => {
				calls += 1;
			},
		});
		watchers[0]?.emit("f1.txt");
		watchers[0]?.emit("f2.txt");
		watchers[0]?.emit("f3.txt");
		await vi.advanceTimersByTimeAsync(60);
		expect(calls).toBe(1);
	});

	it("allows at most one in-flight refresh plus one coalesced rerun", async () => {
		let calls = 0;
		let release: (() => void) | undefined;
		createWatchRefresh({
			dirs: ["/a"],
			debounceMs: 10,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: () => {
				calls += 1;
				return new Promise<void>((resolve) => {
					release = resolve;
				});
			},
		});
		// First burst starts a refresh.
		watchers[0]?.emit("f1");
		await vi.advanceTimersByTimeAsync(15);
		expect(calls).toBe(1);
		// While in-flight, more events collapse into a single pending rerun.
		watchers[0]?.emit("f2");
		watchers[0]?.emit("f3");
		await vi.advanceTimersByTimeAsync(15);
		expect(calls).toBe(1);
		// Completing the first refresh triggers exactly one rerun.
		release?.();
		await vi.advanceTimersByTimeAsync(15);
		expect(calls).toBe(2);
	});

	it("ignores events and schedules no refresh after stop(); closes all watchers", async () => {
		let calls = 0;
		const handle = createWatchRefresh({
			dirs: ["/a", "/b"],
			debounceMs: 50,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: async () => {
				calls += 1;
			},
		});
		handle.stop();
		expect(watchers.every((w) => w.closed)).toBe(true);
		watchers[0]?.emit("f1");
		await vi.advanceTimersByTimeAsync(100);
		expect(calls).toBe(0);
	});

	it("does not schedule a refresh if stop() happens during the debounce window", async () => {
		let calls = 0;
		const handle = createWatchRefresh({
			dirs: ["/a"],
			debounceMs: 50,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: async () => {
				calls += 1;
			},
		});
		watchers[0]?.emit("f1");
		handle.stop();
		await vi.advanceTimersByTimeAsync(100);
		expect(calls).toBe(0);
	});

	it("ignores events under excluded directories (.autorag/.git/node_modules)", async () => {
		let calls = 0;
		createWatchRefresh({
			dirs: ["/a"],
			debounceMs: 50,
			maxWatchers: 10,
			watcherFactory: factory(),
			runRefresh: async () => {
				calls += 1;
			},
		});
		watchers[0]?.emit(".autorag/parsed/x.md");
		watchers[0]?.emit(".git/HEAD");
		watchers[0]?.emit("node_modules/pkg/index.js");
		await vi.advanceTimersByTimeAsync(100);
		expect(calls).toBe(0);
	});

	it("caps watchers at maxWatchers and reports the limit", () => {
		let limited = 0;
		createWatchRefresh({
			dirs: ["/a", "/b", "/c"],
			debounceMs: 50,
			maxWatchers: 2,
			watcherFactory: factory(),
			runRefresh: async () => {},
			onLimit: () => {
				limited += 1;
			},
		});
		expect(watchers).toHaveLength(2);
		expect(limited).toBe(1);
	});
});
