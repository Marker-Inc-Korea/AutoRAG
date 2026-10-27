import { describe, expect, it } from "vitest";
import { SharedExclusiveQueue } from "../../src/filesystem/shared-exclusive-queue.ts";

function deferred() {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	let release!: () => void;
	const promise = new Promise<void>((resolve) => {
		release = resolve;
	});
	return { promise, release };
}

/** Lets every already-queued promise continuation run; no wall-clock wait. */
async function tick(): Promise<void> {
	for (let turn = 0; turn < 20; turn++) await Promise.resolve();
}

describe("SharedExclusiveQueue", () => {
	it("runs shared tasks at the same time", async () => {
		const queue = new SharedExclusiveQueue();
		const gate = deferred();
		let inFlight = 0;
		let peak = 0;
		const task = async () => {
			inFlight += 1;
			peak = Math.max(peak, inFlight);
			await gate.promise;
			inFlight -= 1;
		};
		const all = Promise.all([queue.shared(task), queue.shared(task), queue.shared(task)]);
		await tick();
		expect(peak).toBe(3);
		gate.release();
		await all;
	});

	it("holds an exclusive task until running shared tasks finish, and later shared tasks until it finishes", async () => {
		const queue = new SharedExclusiveQueue();
		const sharedGate = deferred();
		const exclusiveGate = deferred();
		const events: string[] = [];
		const first = queue.shared(async () => {
			events.push("shared-1 start");
			await sharedGate.promise;
			events.push("shared-1 end");
		});
		const exclusive = queue.exclusive(async () => {
			events.push("exclusive start");
			await exclusiveGate.promise;
			events.push("exclusive end");
		});
		const later = queue.shared(async () => {
			events.push("shared-2 start");
		});
		await tick();
		expect(events).toEqual(["shared-1 start"]);
		sharedGate.release();
		await first;
		await tick();
		expect(events).toEqual(["shared-1 start", "shared-1 end", "exclusive start"]);
		exclusiveGate.release();
		await Promise.all([exclusive, later]);
		expect(events).toEqual(["shared-1 start", "shared-1 end", "exclusive start", "exclusive end", "shared-2 start"]);
	});

	it("runs exclusive tasks one at a time in call order", async () => {
		const queue = new SharedExclusiveQueue();
		const gate = deferred();
		const events: string[] = [];
		const a = queue.exclusive(async () => {
			events.push("a start");
			await gate.promise;
			events.push("a end");
		});
		const b = queue.exclusive(async () => {
			events.push("b");
		});
		await tick();
		expect(events).toEqual(["a start"]);
		gate.release();
		await Promise.all([a, b]);
		expect(events).toEqual(["a start", "a end", "b"]);
	});

	it("keeps serving tasks after one rejects, and still rejects that caller", async () => {
		const queue = new SharedExclusiveQueue();
		await expect(queue.shared(async () => Promise.reject(new Error("boom")))).rejects.toThrow("boom");
		await expect(queue.exclusive(async () => Promise.reject(new Error("bang")))).rejects.toThrow("bang");
		await expect(queue.shared(async () => "ok")).resolves.toBe("ok");
		await expect(queue.exclusive(async () => "fine")).resolves.toBe("fine");
	});
});
