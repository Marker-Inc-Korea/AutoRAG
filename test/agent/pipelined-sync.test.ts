import { describe, expect, it } from "vitest";
import { PipelinedSync, type PipelinedSyncRun } from "../../src/agent/pipelined-sync.ts";

/** A promise the test resolves from outside. */
function deferred(): { readonly promise: Promise<void>; readonly resolve: () => void } {
	let resolve: () => void = () => undefined;
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	const promise = new Promise<void>((done) => {
		resolve = done;
	});
	return { promise, resolve };
}

/** A sync whose runs finish only when the test releases them; `started(n)` resolves once run n has begun. */
function gatedSync(outcomes: ReadonlyMap<number, { readonly stop?: boolean } | Error> = new Map()) {
	const runs: PipelinedSyncRun[] = [];
	const gates = new Map<number, ReturnType<typeof deferred>>();
	const startSignals = new Map<number, ReturnType<typeof deferred>>();
	const gate = (map: Map<number, ReturnType<typeof deferred>>, index: number) => {
		let entry = map.get(index);
		if (entry === undefined) {
			entry = deferred();
			map.set(index, entry);
		}
		return entry;
	};
	const pipeline = new PipelinedSync({
		sync: async (run) => {
			runs.push(run);
			gate(startSignals, run.index).resolve();
			await gate(gates, run.index).promise;
			const outcome = outcomes.get(run.index);
			if (outcome instanceof Error) throw outcome;
			return outcome;
		},
	});
	return {
		pipeline,
		runs,
		started: (index: number) => gate(startSignals, index).promise,
		release: (index: number) => gate(gates, index).resolve(),
	};
}

describe("PipelinedSync", () => {
	it("starts a sync on the first batch without making the parse loop wait for it", async () => {
		const { pipeline, runs } = gatedSync();

		const outcome = await pipeline.batch({ virtualPaths: ["/a", "/b"], final: false });

		expect(outcome).toBeUndefined();
		expect(runs).toEqual([{ index: 1, virtualPaths: ["/a", "/b"] }]);
	});

	it("folds every batch committed during a sync into the one sync that follows it", async () => {
		const { pipeline, runs, started, release } = gatedSync();
		await pipeline.batch({ virtualPaths: ["/a"], final: false });
		await pipeline.batch({ virtualPaths: ["/b"], final: false });
		await pipeline.batch({ virtualPaths: ["/c"], final: false });
		expect(runs).toHaveLength(1);

		const last = pipeline.batch({ virtualPaths: ["/d"], final: true });
		release(1);
		await started(2);
		release(2);
		await last;

		expect(runs).toEqual([
			{ index: 1, virtualPaths: ["/a"] },
			{ index: 2, virtualPaths: ["/b", "/c", "/d"] },
		]);
	});

	it("makes the final batch wait for a sync that began after every batch was committed", async () => {
		const { pipeline, runs, started, release } = gatedSync();
		await pipeline.batch({ virtualPaths: ["/a"], final: false });
		release(1);

		let finished = false;
		const last = pipeline.batch({ virtualPaths: ["/b"], final: true }).then(() => {
			finished = true;
		});
		await started(2);
		expect(finished).toBe(false);
		release(2);
		await last;

		expect(finished).toBe(true);
		expect(runs.flatMap((run) => run.virtualPaths)).toEqual(["/a", "/b"]);
	});

	it("runs one sync for a final batch that covers no sources", async () => {
		const { pipeline, runs, release } = gatedSync();

		const last = pipeline.batch({ virtualPaths: [], final: true });
		release(1);
		await last;

		expect(runs).toEqual([{ index: 1, virtualPaths: [] }]);
	});

	it("raises a failed sync at the next batch instead of dropping it", async () => {
		const failure = new Error("sync exploded");
		const { pipeline, release } = gatedSync(new Map([[1, failure]]));
		await pipeline.batch({ virtualPaths: ["/a"], final: false });
		release(1);
		await pipeline.settle();

		await expect(pipeline.batch({ virtualPaths: ["/b"], final: false })).rejects.toBe(failure);
	});

	it("raises a failed sync from the final batch", async () => {
		const failure = new Error("sync exploded");
		const { pipeline, release } = gatedSync(new Map([[1, failure]]));
		const last = pipeline.batch({ virtualPaths: ["/a"], final: true });
		release(1);

		await expect(last).rejects.toBe(failure);
	});

	it("stops syncing once a run asks to stop, and still lets the final batch return", async () => {
		const { pipeline, runs, release } = gatedSync(new Map([[1, { stop: true }]]));
		await pipeline.batch({ virtualPaths: ["/a"], final: false });

		// Committed while run 1 is in flight, so a run 2 would follow if run 1 did not stop.
		const last = pipeline.batch({ virtualPaths: ["/b"], final: true });
		release(1);

		expect(await last).toEqual({ stop: true });
		expect(runs).toHaveLength(1);
	});

	it("settles by waiting for the run in flight and starting no further one", async () => {
		const { pipeline, runs, release } = gatedSync();
		await pipeline.batch({ virtualPaths: ["/a"], final: false });
		await pipeline.batch({ virtualPaths: ["/b"], final: false });

		const settled = pipeline.settle();
		release(1);
		await settled;

		expect(runs).toHaveLength(1);
	});
});
