import type { EventEmitter as EventEmitterType } from "node:events";
import type { PassThrough as PassThroughType } from "node:stream";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { JikjiClient } from "../../src/jikji/index.ts";

interface FakeChild {
	readonly args: readonly string[];
	killed: boolean;
	exit(code: number): void;
}

const state = vi.hoisted(() => ({
	spawned: [] as FakeChild[],
	waiters: [] as Array<(child: FakeChild) => void>,
}));

// Upstream keeps every root in ONE central SQLite file and sets no busy
// timeout, so two jikji processes writing at once fail with "database is
// locked". `prepare` always writes; a `find` that returns no paths reindexes
// synchronously, so it writes too. The mock lets each test decide exactly when
// a spawned process finishes, so ordering is driven by events, not by a clock.
vi.mock("node:child_process", async () => {
	// vi.mock is hoisted above the imports, so the node builtins come from
	// importActual (a module-boundary seam, not a source-level dynamic import).
	const events = await vi.importActual<{ EventEmitter: typeof EventEmitterType }>("node:events");
	const stream = await vi.importActual<{ PassThrough: typeof PassThroughType }>("node:stream");
	class MockChild extends events.EventEmitter implements FakeChild {
		readonly stdout = new stream.PassThrough();
		readonly stderr = new stream.PassThrough();
		readonly args: readonly string[];
		killed = false;
		constructor(args: readonly string[]) {
			super();
			this.args = args;
		}
		kill(): boolean {
			this.killed = true;
			this.emit("close", null);
			return true;
		}
		exit(code: number): void {
			this.emit("close", code);
		}
	}
	return {
		spawn: (_command: string, args: readonly string[]) => {
			const child = new MockChild(args);
			state.spawned.push(child);
			state.waiters.shift()?.(child);
			return child;
		},
	};
});

// #1905 kills a signal-bearing run through `terminateProcessTree`, which needs a
// real pid and signals a process group. The mock records the request and closes
// the fake child instead, so no real process group is ever signalled.
vi.mock("../../src/process/terminate-tree.ts", () => ({
	terminateProcessTree: (child: FakeChild & { kill(): boolean }) => child.kill(),
}));

let consumed = 0;

/** Resolves with the next spawned process the test has not looked at yet. */
function nextChild(): Promise<FakeChild> {
	const ready = state.spawned[consumed];
	if (ready !== undefined) {
		consumed += 1;
		return Promise.resolve(ready);
	}
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	return new Promise((resolve) => {
		state.waiters.push((child) => {
			consumed += 1;
			resolve(child);
		});
	});
}

const spawnedRoots = (): (string | undefined)[] => state.spawned.map((child) => child.args[1]);
const newClient = (): JikjiClient => new JikjiClient({ binaryPath: "/fake/jikji" });

beforeEach(() => {
	state.spawned.length = 0;
	state.waiters.length = 0;
	consumed = 0;
});

describe("JikjiClient process serialization", () => {
	it("runs concurrent prepares one at a time, in submission order", async () => {
		const client = newClient();
		const results = [client.prepare("/r/a"), client.prepare("/r/b"), client.prepare("/r/c")];

		const first = await nextChild();
		expect(spawnedRoots()).toEqual(["/r/a"]);
		first.exit(0);

		const second = await nextChild();
		expect(spawnedRoots()).toEqual(["/r/a", "/r/b"]);
		second.exit(0);

		(await nextChild()).exit(0);
		expect(spawnedRoots()).toEqual(["/r/a", "/r/b", "/r/c"]);
		for (const result of await Promise.all(results)) expect(result).toMatchObject({ ok: true });
	});

	it("runs concurrent finds one at a time, in submission order", async () => {
		const client = newClient();
		const results = [client.find("/r/a", "q"), client.find("/r/b", "q")];

		const first = await nextChild();
		expect(spawnedRoots()).toEqual(["/r/a"]);
		first.exit(0);

		(await nextChild()).exit(0);
		expect(spawnedRoots()).toEqual(["/r/a", "/r/b"]);
		await Promise.all(results);
	});

	it("does not make a query wait behind a running refresh prepare", async () => {
		const client = newClient();
		const refreshing = client.prepare("/r/a");
		const prepareChild = await nextChild();

		const found = client.find("/r/b", "q");
		const findChild = await nextChild();

		expect(state.spawned).toHaveLength(2);
		findChild.exit(0);
		prepareChild.exit(0);
		await Promise.all([found, refreshing]);
	});

	it("resolves a queued call as aborted without ever spawning it, and keeps the queue moving", async () => {
		const client = newClient();
		const running = client.prepare("/r/a");
		const runningChild = await nextChild();
		const controller = new AbortController();
		const queued = client.prepare("/r/b", { signal: controller.signal });

		controller.abort();

		await expect(queued).resolves.toMatchObject({ ok: false, reason: "aborted" });
		expect(state.spawned).toHaveLength(1);

		const later = client.prepare("/r/c");
		runningChild.exit(0);
		await expect(running).resolves.toMatchObject({ ok: true });
		(await nextChild()).exit(0);
		await expect(later).resolves.toMatchObject({ ok: true });
		expect(spawnedRoots()).toEqual(["/r/a", "/r/c"]);
	});

	it("kills the running process when its own signal aborts", async () => {
		const client = newClient();
		const controller = new AbortController();
		const pending = client.prepare("/r/a", { signal: controller.signal });
		const child = await nextChild();

		controller.abort();

		await expect(pending).resolves.toMatchObject({ ok: false, reason: "aborted" });
		expect(child.killed).toBe(true);
	});

	it("keeps serving later calls after an earlier queued call failed", async () => {
		const client = newClient();
		const bad = client.prepare("/r/bad");
		const good = client.prepare("/r/good");

		(await nextChild()).exit(3);
		(await nextChild()).exit(0);

		await expect(bad).resolves.toMatchObject({ ok: false, reason: "nonzero-exit" });
		await expect(good).resolves.toMatchObject({ ok: true });
	});
});
