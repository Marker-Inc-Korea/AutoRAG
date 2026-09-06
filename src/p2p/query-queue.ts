export class QueryQueueFullError extends Error {
	constructor(message = "The peer query queue is full.") {
		super(message);
		this.name = "QueryQueueFullError";
	}
}

export class QueryQueueClosedError extends Error {
	constructor(message = "The peer query queue is closed.") {
		super(message);
		this.name = "QueryQueueClosedError";
	}
}

type QueueTask<T> = () => T | PromiseLike<T>;

interface QueueEntry<T> {
	readonly task: QueueTask<T>;
	readonly resolve: (value: T | PromiseLike<T>) => void;
	readonly reject: (reason?: unknown) => void;
}

/**
 * A bounded FIFO queue whose tasks are always executed one at a time.
 *
 * The depth counts waiting tasks, not the task currently running. This lets a
 * caller reserve a small amount of useful backlog while preserving the
 * single-flight invariant of AutoRAGAgent.searchDocuments().
 */
export class QueryQueue<T> {
	private readonly depth: number;
	private readonly pending: QueueEntry<T>[] = [];
	private running = false;
	private closed = false;

	constructor(depth = 8) {
		if (!Number.isSafeInteger(depth) || depth < 1) {
			throw new RangeError("Query queue depth must be a positive integer.");
		}
		this.depth = depth;
	}

	get pendingCount(): number {
		return this.pending.length;
	}

	get active(): boolean {
		return this.running;
	}

	get size(): number {
		return this.pending.length + (this.running ? 1 : 0);
	}

	enqueue(task: QueueTask<T>): Promise<T> {
		if (this.closed) throw new QueryQueueClosedError();
		if (this.pending.length >= this.depth) throw new QueryQueueFullError();

		const result = new Promise<T>((resolve, reject) => {
			this.pending.push({ task, resolve, reject });
		});
		void this.drain();
		return result;
	}

	/** Stop accepting work and reject tasks which have not started yet. */
	close(reason: unknown = new QueryQueueClosedError()): void {
		if (this.closed) return;
		this.closed = true;
		const pending = this.pending.splice(0);
		for (const entry of pending) entry.reject(reason);
	}

	private async drain(): Promise<void> {
		if (this.running) return;
		this.running = true;
		try {
			while (this.pending.length > 0) {
				const entry = this.pending.shift();
				if (entry === undefined) continue;
				try {
					entry.resolve(await entry.task());
				} catch (error) {
					entry.reject(error);
				}
			}
		} finally {
			this.running = false;
		}
	}
}

export const P2PQueryQueue = QueryQueue;
