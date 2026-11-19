/**
 * Query time window: `since` (inclusive) and `until` (exclusive) in epoch ms.
 *
 * No store can filter by time natively (MinSync's `query` takes only `-k`, `--mode`, `--format`), so a window is a
 * post-filter over an over-fetched candidate list, the same way `scope` already over-fetches. A result's time is
 * `metadata.publishedAt` (the document's own timestamp) or, for local files, `metadata.sourceMtimeMs` (the source's
 * last modification, not its authorship). A result with neither is excluded and reported; it is never passed
 * through as if it had been filtered.
 */

import type { RetrievalOptions, RetrievalResult } from "./types.ts";

/** Half-open `[since, until)` in epoch ms. A missing bound is open. */
export interface TimeWindow {
	readonly since?: number;
	readonly until?: number;
}

/** Thrown for a malformed or empty window. The message is the text shown to the caller, verbatim. */
export class TimeWindowError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "TimeWindowError";
	}
}

/** Over-fetch ceiling for a windowed query: never ask a backend for more than this on the window's account. */
export const TIME_WINDOW_FETCH_CEILING = 300;
/** Depth a windowed query asks for when the caller gave no `topK`. */
const TIME_WINDOW_DEFAULT_BASE = 50;

const ISO = /^(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?(Z|[+-]\d{2}:?\d{2})?)?$/u;

function daysInMonth(year: number, month: number): number {
	return new Date(Date.UTC(year, month, 0)).getUTCDate();
}

/**
 * Parse one ISO-8601 bound. A date-only value or a datetime without an offset is read in the host's local time
 * zone (the user's); `Z` or `±HH:MM` is honored exactly. Calendar-invalid values (`2026-02-31`) are rejected
 * instead of rolling into March.
 */
function parseBound(field: "since" | "until", raw: string): number {
	const value = raw.trim();
	const fail = (): never => {
		throw new TimeWindowError(
			`Invalid ${field} "${raw}": expected an ISO-8601 date or datetime such as 2026-09-01, 2026-09-01T09:30, or 2026-09-01T00:00:00+09:00.`,
		);
	};
	const match = ISO.exec(value);
	if (match === null) return fail();
	const year = Number(match[1]);
	const month = Number(match[2]);
	const day = Number(match[3]);
	const hour = match[4] === undefined ? 0 : Number(match[4]);
	const minute = match[5] === undefined ? 0 : Number(match[5]);
	const second = match[6] === undefined ? 0 : Number(match[6]);
	const millis = match[7] === undefined ? 0 : Math.floor(Number(`0.${match[7]}`) * 1000);
	if (month < 1 || month > 12 || day < 1 || day > daysInMonth(year, month)) return fail();
	if (hour > 23 || minute > 59 || second > 59) return fail();
	const offset = match[8];
	if (offset === undefined) {
		const local = new Date(0);
		local.setFullYear(year, month - 1, day);
		local.setHours(hour, minute, second, millis);
		return local.getTime();
	}
	const utc = new Date(0);
	utc.setUTCFullYear(year, month - 1, day);
	utc.setUTCHours(hour, minute, second, millis);
	if (offset === "Z") return utc.getTime();
	const sign = offset.startsWith("-") ? -1 : 1;
	const digits = offset.slice(1).replace(":", "");
	const offsetHours = Number(digits.slice(0, 2));
	const offsetMinutes = Number(digits.slice(2));
	if (offsetHours > 23 || offsetMinutes > 59) return fail();
	return utc.getTime() - sign * (offsetHours * 60 + offsetMinutes) * 60_000;
}

/**
 * Parse the model- or operator-supplied ISO strings into a window. Returns undefined when neither bound is set.
 * A malformed bound or a window that is empty by construction throws {@link TimeWindowError}.
 */
export function parseTimeWindow(input: {
	readonly since?: string | undefined;
	readonly until?: string | undefined;
}): TimeWindow | undefined {
	const sinceRaw = input.since?.trim();
	const untilRaw = input.until?.trim();
	const since = sinceRaw === undefined || sinceRaw === "" ? undefined : parseBound("since", sinceRaw);
	const until = untilRaw === undefined || untilRaw === "" ? undefined : parseBound("until", untilRaw);
	if (since === undefined && until === undefined) return undefined;
	if (since !== undefined && until !== undefined && since >= until) {
		throw new TimeWindowError(
			`Invalid time window: since "${sinceRaw}" must be before until "${untilRaw}" (the window is [since, until)).`,
		);
	}
	return { ...(since === undefined ? {} : { since }), ...(until === undefined ? {} : { until }) };
}

/** The window carried by retrieval options, or undefined when neither bound is set. */
export function windowFromOptions(options: RetrievalOptions): TimeWindow | undefined {
	if (options.since === undefined && options.until === undefined) return undefined;
	return {
		...(options.since === undefined ? {} : { since: options.since }),
		...(options.until === undefined ? {} : { until: options.until }),
	};
}

/** Whether `timeMs` lies in `[since, until)`. */
export function inTimeWindow(timeMs: number, window: TimeWindow): boolean {
	if (window.since !== undefined && timeMs < window.since) return false;
	if (window.until !== undefined && timeMs >= window.until) return false;
	return true;
}

/** A finite number, or undefined. */
function finite(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

/**
 * The time of a result: `metadata.publishedAt` (the document's own timestamp, epoch ms), else the local source file's
 * `metadata.sourceMtimeMs` (last modified, not authored; cloud placeholders may carry a sync time).
 */
export function resultTimeMs(result: RetrievalResult): number | undefined {
	return finite(result.metadata.publishedAt) ?? finite(result.metadata.sourceMtimeMs);
}

/** Epoch values below this are seconds: as milliseconds they would be before 1973. */
const MIN_EPOCH_MS = 100_000_000_000;

/**
 * Epoch ms from what a store reported: a number or digits-only string (seconds or milliseconds), or an ISO / RFC 2822
 * string. Undefined when it cannot be placed.
 */
export function parseTimestampMs(value: unknown): number | undefined {
	const trimmed = typeof value === "string" ? value.trim() : value;
	const epoch = typeof trimmed === "string" && /^\d+(?:\.\d+)?$/u.test(trimmed) ? Number(trimmed) : finite(trimmed);
	if (epoch !== undefined) return Math.round(epoch < MIN_EPOCH_MS ? epoch * 1000 : epoch);
	if (typeof trimmed !== "string" || trimmed === "") return undefined;
	return finite(Date.parse(trimmed));
}

/** `{ publishedAt }` for a store-reported time that can be placed, else `{}`: spread into result metadata. */
export function publishedAtFrom(value: unknown): { publishedAt?: number } {
	const publishedAt = parseTimestampMs(value);
	return publishedAt === undefined ? {} : { publishedAt };
}

/**
 * How deep a backend must be asked so a window can be post-filtered: the same shape as the scope over-fetch
 * (`max(5x, +20)`), bounded by {@link TIME_WINDOW_FETCH_CEILING}, and never below what the caller asked for.
 */
export function expandedWindowTopK(topK: number | undefined): number {
	if (topK === undefined) return TIME_WINDOW_DEFAULT_BASE * 5;
	return Math.min(Math.max(topK * 5, topK + 20), Math.max(TIME_WINDOW_FETCH_CEILING, topK));
}

/** `[since, until)` as ISO instants; an open bound reads `unbounded`. */
export function describeTimeWindow(window: TimeWindow): string {
	const since = window.since === undefined ? "unbounded" : new Date(window.since).toISOString();
	const until = window.until === undefined ? "unbounded" : new Date(window.until).toISOString();
	return `[${since}, ${until})`;
}

/** Results of one method split by a window: those inside it, and how many had no time to judge by. */
export interface WindowedResults {
	readonly kept: RetrievalResult[];
	readonly untimed: number;
}

/** Keep the results whose time lies in `window`; results with no time are dropped and counted, never passed through. */
export function applyTimeWindow(results: readonly RetrievalResult[], window: TimeWindow): WindowedResults {
	const kept: RetrievalResult[] = [];
	let untimed = 0;
	for (const result of results) {
		const time = resultTimeMs(result);
		if (time === undefined) untimed += 1;
		else if (inTimeWindow(time, window)) kept.push(result);
	}
	return { kept, untimed };
}
