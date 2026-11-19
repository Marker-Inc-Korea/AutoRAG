import type { ParsedMirrorBatch, ParsedMirrorBatchOutcome } from "../mirror/sync.ts";

/** One sync run: the 1-based run number and the sources of every batch committed since the previous run began. */
export interface PipelinedSyncRun {
	readonly index: number;
	readonly virtualPaths: readonly string[];
}

export interface PipelinedSyncOptions {
	/**
	 * Index everything committed so far. Resolving with `stop` ends the pipeline (a held lock means another
	 * process owns the index, so every further run would fail the same way); a rejection is raised from the
	 * next {@link PipelinedSync.batch} call.
	 */
	readonly sync: (run: PipelinedSyncRun) => Promise<ParsedMirrorBatchOutcome | undefined>;
}

/**
 * Runs index syncs alongside the parse loop instead of between its batches.
 *
 * A sync only sees what is committed when it starts, and the indexer batches and commits on its own, so the
 * first batch starts a sync right away while parsing carries on. At most one sync is in flight; when it ends,
 * the next one covers every batch committed meanwhile. A slow sync therefore turns into few large ones
 * instead of one per parse batch, and each sync pays its start-up and end-of-run maintenance once.
 */
export class PipelinedSync {
	readonly #sync: PipelinedSyncOptions["sync"];
	#covered: string[] = [];
	#dirty = false;
	#running: Promise<void> | undefined;
	#failure: { readonly error: unknown } | undefined;
	#stopped = false;
	#runs = 0;

	constructor(options: PipelinedSyncOptions) {
		this.#sync = options.sync;
	}

	/**
	 * Hand over a committed batch. Returns at once, except for the final batch: it waits until a sync that
	 * started after every batch was committed has finished, and always runs one, so a sync that failed
	 * earlier is retried even when nothing changed.
	 */
	async batch(
		batch: Pick<ParsedMirrorBatch, "virtualPaths" | "final">,
	): Promise<ParsedMirrorBatchOutcome | undefined> {
		this.#raise();
		this.#covered.push(...batch.virtualPaths);
		this.#dirty = true;
		if (!this.#stopped) this.#running ??= this.#loop();
		if (batch.final) await this.#running;
		this.#raise();
		return this.#stopped ? { stop: true } : undefined;
	}

	/** Wait for the sync in flight, if any, and start no further one. Never rejects. */
	async settle(): Promise<void> {
		this.#stopped = true;
		await this.#running;
	}

	#raise(): void {
		if (this.#failure !== undefined) throw this.#failure.error;
	}

	async #loop(): Promise<void> {
		try {
			while (this.#dirty && !this.#stopped) {
				this.#dirty = false;
				const virtualPaths = this.#covered;
				this.#covered = [];
				this.#runs += 1;
				const outcome = await this.#sync({ index: this.#runs, virtualPaths });
				if (outcome?.stop === true) this.#stopped = true;
			}
		} catch (error) {
			this.#failure = { error };
		} finally {
			this.#running = undefined;
		}
	}
}
