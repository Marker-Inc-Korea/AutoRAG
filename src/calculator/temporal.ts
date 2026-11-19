import { daysFromCivil, floorDiv, isRealDate, MS_PER_DAY } from "./civil.ts";
import { CalculatorError } from "./errors.ts";
import { fixedZone, fromNaive, parseZone, toNaive, type Zone } from "./timezone.ts";

/** A date with no time of day, or one instant on the timeline. */
export type Temporal =
	| { readonly kind: "date"; readonly days: number }
	/** `zoneless` marks a wall-clock time that was placed in the zone by assumption, not by its own offset. */
	| { readonly kind: "instant"; readonly epochMs: number; readonly zoneless: boolean };

const DATE_FORMS = [
	/^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})/,
	/^(\d{4})(\d{2})(\d{2})(?!\d)/,
	/^(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일/,
] as const;

const TIME_FORM = /^(?:T|\s+)\s*(\d{1,2}):(\d{2})(?::(\d{2})(?:\.(\d{1,9}))?)?\s*(Z|[+-]\d{1,2}(?::?\d{2})?)?$/i;

const UNREADABLE =
	'Use YYYY-MM-DD (also YYYY/MM/DD, YYYY.MM.DD, YYYYMMDD, YYYY년 M월 D일), YYYY-MM-DDTHH:MM:SS with an optional offset such as +09:00 or Z, 10-digit epoch seconds, "today" or "now".';

export function todayIn(zone: Zone, now: Date): number {
	return floorDiv(toNaive(zone, now.getTime()), MS_PER_DAY);
}

export function parseTemporal(input: string, zone: Zone, now: Date): Temporal {
	const text = input.trim();
	const lower = text.toLowerCase();
	if (lower === "today") return { kind: "date", days: todayIn(zone, now) };
	if (lower === "now") return { kind: "instant", epochMs: now.getTime(), zoneless: false };
	if (/^-?\d{10}$/.test(text)) return { kind: "instant", epochMs: Number(text) * 1000, zoneless: false };
	if (/^-?\d{13}$/.test(text)) return { kind: "instant", epochMs: Number(text), zoneless: false };

	for (const form of DATE_FORMS) {
		const date = form.exec(text);
		if (date === null) continue;
		const [y, m, d] = [Number(date[1]), Number(date[2]), Number(date[3])] as const;
		if (!isRealDate(y, m, d)) throw new CalculatorError(`"${date[0]}" is not a real date.`);
		const days = daysFromCivil(y, m, d);
		const rest = text.slice(date[0].length);
		if (rest.trim() === "") return { kind: "date", days };
		return parseClock(text, rest, days, zone);
	}
	throw new CalculatorError(`Could not read "${text}" as a date or time. ${UNREADABLE}`);
}

function parseClock(text: string, rest: string, days: number, zone: Zone): Temporal {
	const time = TIME_FORM.exec(rest);
	if (time === null) throw new CalculatorError(`Could not read "${text}" as a date or time. ${UNREADABLE}`);
	const [hour, minute, second] = [Number(time[1]), Number(time[2]), Number(time[3] ?? 0)] as const;
	if (hour > 23 || minute > 59 || second > 59) {
		throw new CalculatorError(
			`"${time[1]}:${time[2]}${time[3] === undefined ? "" : `:${time[3]}`}" is not a real time.`,
		);
	}
	const millis = Number(`${time[4] ?? ""}000`.slice(0, 3));
	const naive = days * MS_PER_DAY + ((hour * 60 + minute) * 60 + second) * 1000 + millis;
	if (time[5] === undefined) return { kind: "instant", epochMs: fromNaive(zone, naive), zoneless: true };
	const offset = fixedZone(0);
	const explicit = time[5].toUpperCase() === "Z" ? offset : parseZone(time[5]);
	return { kind: "instant", epochMs: naive - (explicit.fixed as number) * 1000, zoneless: false };
}

/** Calendar day of a value: a date as is, an instant as the date on the wall clock in `zone`. */
export function dayOf(value: Temporal, zone: Zone): number {
	return value.kind === "date" ? value.days : floorDiv(toNaive(zone, value.epochMs), MS_PER_DAY);
}

/** The instant of a value: a date means midnight in `zone`. */
export function instantOf(value: Temporal, zone: Zone): number {
	return value.kind === "instant" ? value.epochMs : fromNaive(zone, value.days * MS_PER_DAY);
}
