const pad = (value: number): string => String(value).padStart(2, "0");

/**
 * The prompt block that tells the model what "today" is. AutoRAG replaces pi's whole
 * system prompt, so nothing else gives the model a clock; without it "last month",
 * "this year", "is the deadline past", and "the most recent" have no anchor.
 *
 * Built per prompt from the host's local clock and time zone, so a session left open
 * for days never shows yesterday's date. The system prompt is the first thing in every
 * request, so anything in it that changes also rewrites the provider's prompt-cache
 * prefix for the whole conversation after it. The block therefore carries the date,
 * which changes once a day, and not the time of day, which would change every minute.
 * The time comes from the `calculator` tool's `now` operation when a question needs it.
 */
export function currentDateAddendum(now: Date = new Date()): string {
	const weekday = new Intl.DateTimeFormat("en-US", { weekday: "long" }).format(now);
	const offsetMinutes = -now.getTimezoneOffset();
	const sign = offsetMinutes < 0 ? "-" : "+";
	const absolute = Math.abs(offsetMinutes);
	const offset = `UTC${sign}${pad(Math.floor(absolute / 60))}:${pad(absolute % 60)}`;
	const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
	const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
	return `\n\n<current_date>\nToday's date: ${date} (${weekday}), ${zone} (${offset}). The time of day is not part of these instructions; use the calculator \`now\` operation when a question depends on it.\nResolve every relative date the user gives ("today", "last month", "this year", "by Friday") and every judgment of recency or whether a date has passed against this date.\n</current_date>`;
}
