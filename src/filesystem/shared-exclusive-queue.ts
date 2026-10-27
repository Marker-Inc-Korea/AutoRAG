/**
 * Orders async tasks like a read/write lock, in arrival order.
 *
 * `shared` tasks (searches) overlap each other. An `exclusive` task (index,
 * stop) waits for every earlier task and holds back every later one, so a
 * reindex never runs under a search and a search never reads a half-built
 * index. A rejected task rejects its own caller only; the queue keeps going.
 */
export class SharedExclusiveQueue {
	/** Shared tasks queued since the last exclusive one; the next exclusive task waits for all of them. */
	private sharedSinceExclusive: Promise<unknown>[] = [];
	/** Settles when the last exclusive task has; every shared task waits for it. */
	private lastExclusive: Promise<unknown> = Promise.resolve();

	shared<T>(task: () => Promise<T>): Promise<T> {
		const run = this.lastExclusive.then(task);
		const settled = run.then(
			() => undefined,
			() => undefined,
		);
		this.sharedSinceExclusive.push(settled);
		return run;
	}

	exclusive<T>(task: () => Promise<T>): Promise<T> {
		const earlier = Promise.all([this.lastExclusive, ...this.sharedSinceExclusive]);
		const run = earlier.then(task);
		const settled = run.then(
			() => undefined,
			() => undefined,
		);
		this.sharedSinceExclusive = [];
		this.lastExclusive = settled;
		return run;
	}
}
