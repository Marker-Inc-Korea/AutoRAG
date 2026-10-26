const pad = (value: number): string => String(value).padStart(2, "0");

/**
 * The prompt block that tells the model what "now" is. AutoRAG replaces pi's whole
 * system prompt, so nothing else gives the model a clock; without it "last month",
 * "this year", "is the deadline past", and "the most recent" have no anchor.
 *
 * Built per turn from the host's local clock and time zone, so a session left open
 * for days never shows yesterday's date. It sits last in the prompt so the stable
 * part before it keeps its prompt-cache prefix.
 */
export function currentDateAddendum(now: Date = new Date()): string {
	const weekday = new Intl.DateTimeFormat("en-US", { weekday: "long" }).format(now);
	const offsetMinutes = -now.getTimezoneOffset();
	const sign = offsetMinutes < 0 ? "-" : "+";
	const absolute = Math.abs(offsetMinutes);
	const offset = `UTC${sign}${pad(Math.floor(absolute / 60))}:${pad(absolute % 60)}`;
	const zone = Intl.DateTimeFormat().resolvedOptions().timeZone;
	const date = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
	const time = `${pad(now.getHours())}:${pad(now.getMinutes())}`;
	return `\n\n<current_date>\nCurrent date and time: ${date} (${weekday}) ${time}, ${zone} (${offset}).\nResolve every relative date the user gives ("today", "last month", "this year", "by Friday") and every judgment of recency or whether a date has passed against this moment.\n</current_date>`;
}
