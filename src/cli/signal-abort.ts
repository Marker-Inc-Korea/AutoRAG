/** Exit codes a shell reports for a process that stops on a signal: 128 plus the signal number. */
export const SIGNAL_EXIT_CODES = { SIGINT: 130, SIGTERM: 143 } as const;

export type StopSignal = keyof typeof SIGNAL_EXIT_CODES;

export interface SignalAbort {
	/** Aborts, with an error naming the signal, when the process is told to stop. */
	readonly signal: AbortSignal;
	/** The first signal received, or `undefined` while none has arrived. */
	readonly received: StopSignal | undefined;
	/** Stop listening. Call once the guarded work has settled. */
	dispose(): void;
}

/**
 * Turn SIGINT and SIGTERM into an abort the long-running work already understands. Without a listener of its
 * own a command is left to the host-exit cleanup, which kills the child processes but lets the run carry on
 * through its other phases against a dead index; with one, the refresh stops, records itself as failed and the
 * command exits with the signal's conventional code.
 */
export function abortOnSignal(): SignalAbort {
	const controller = new AbortController();
	let received: StopSignal | undefined;
	const listeners = (Object.keys(SIGNAL_EXIT_CODES) as StopSignal[]).map((name) => {
		const listener = (): void => {
			received ??= name;
			controller.abort(new Error(`Interrupted by ${name}.`));
		};
		process.on(name, listener);
		return { name, listener };
	});
	return {
		signal: controller.signal,
		get received() {
			return received;
		},
		dispose() {
			for (const { name, listener } of listeners) process.off(name, listener);
		},
	};
}
