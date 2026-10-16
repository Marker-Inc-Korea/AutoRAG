/** What a queued task resolves to when its signal aborts before its turn. */
export interface SerialQueueSkip<S> {
	readonly signal: AbortSignal | undefined;
	readonly value: S;
}

/**
 * Runs async tasks strictly one after another, in submission order. A task that
 * throws rejects only its own caller; the queue keeps serving later tasks.
 *
 * A caller whose signal aborts while it is still waiting resolves immediately
 * with `skip.value` and its task never runs. Once a task has started, aborting
 * is the task's own business.
 */
export class SerialQueue {
	private tail: Promise<void> = Promise.resolve();

	run<T, S = never>(task: () => Promise<T>, skip?: SerialQueueSkip<S>): Promise<T | S> {
		return new Promise<T | S>((resolve, reject) => {
			let skipped = false;
			const skipNow = (): void => {
				skipped = true;
				resolve(skip?.value as S);
			};
			const signal = skip?.signal;
			if (signal?.aborted) skipNow();
			else signal?.addEventListener("abort", skipNow, { once: true });

			this.tail = this.tail.then(async () => {
				signal?.removeEventListener("abort", skipNow);
				if (skipped) return;
				try {
					resolve(await task());
				} catch (error) {
					reject(error);
				}
			});
		});
	}
}
