export interface ToolBurstGuardOptions {
	/** Calls of one tool inside `windowMs` that count as a runaway loop. */
	readonly maxCalls: number;
	/** Sliding window, in milliseconds. */
	readonly windowMs: number;
}

/** One tool called 60 times inside 60 seconds is a loop, not work. */
export const DEFAULT_TOOL_BURST_GUARD: ToolBurstGuardOptions = { maxCalls: 60, windowMs: 60_000 };

/**
 * Detects a runaway tool loop: the same tool called `maxCalls` times inside
 * `windowMs`. It limits the rate of one tool, never a run's total length or
 * duration, so a long indexing job or a many-step run is never cut short.
 * It reports every call that is part of a burst; `reset()` clears it for the next run.
 */
export class ToolBurstGuard {
	private readonly options: ToolBurstGuardOptions | undefined;
	private readonly calls = new Map<string, number[]>();

	constructor(options: ToolBurstGuardOptions | false = DEFAULT_TOOL_BURST_GUARD) {
		if (options !== false) {
			if (!Number.isInteger(options.maxCalls) || options.maxCalls <= 0) {
				throw new Error("toolCallBurstGuard.maxCalls must be a positive integer");
			}
			if (!Number.isFinite(options.windowMs) || options.windowMs <= 0) {
				throw new Error("toolCallBurstGuard.windowMs must be a positive finite number");
			}
		}
		this.options = options === false ? undefined : options;
	}

	/**
	 * Record one call. Returns true while the tool is inside a burst: on the call that
	 * reaches `maxCalls` and on every further call in the window. An abort issued
	 * mid-tool does not always stop the next model turn, so the caller keeps aborting.
	 */
	record(tool: string, now: number = Date.now()): boolean {
		if (this.options === undefined) return false;
		const { maxCalls, windowMs } = this.options;
		const recent = (this.calls.get(tool) ?? []).filter((at) => now - at < windowMs);
		recent.push(now);
		this.calls.set(tool, recent);
		return recent.length >= maxCalls;
	}

	reset(): void {
		this.calls.clear();
	}
}
