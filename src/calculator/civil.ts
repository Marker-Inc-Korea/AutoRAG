/** Proleptic Gregorian calendar arithmetic on whole days since 1970-01-01. No time zones, no `Date`. */
export const MS_PER_DAY = 86_400_000;
export const MIN_YEAR = 1;
export const MAX_YEAR = 9999;

export const WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] as const;

export interface Civil {
	readonly y: number;
	readonly m: number;
	readonly d: number;
}

export function pad(value: number, width = 2): string {
	return String(value).padStart(width, "0");
}

export function floorDiv(a: number, b: number): number {
	return Math.floor(a / b);
}

export function mod(a: number, b: number): number {
	return ((a % b) + b) % b;
}

export function isLeapYear(y: number): boolean {
	return (y % 4 === 0 && y % 100 !== 0) || y % 400 === 0;
}

export function daysInMonth(y: number, m: number): number {
	return m === 2 ? (isLeapYear(y) ? 29 : 28) : [4, 6, 9, 11].includes(m) ? 30 : 31;
}

export function isRealDate(y: number, m: number, d: number): boolean {
	return y >= MIN_YEAR && y <= MAX_YEAR && m >= 1 && m <= 12 && d >= 1 && d <= daysInMonth(y, m);
}

/** Days since 1970-01-01 (Howard Hinnant's days_from_civil). */
export function daysFromCivil(y: number, m: number, d: number): number {
	const shifted = m <= 2 ? y - 1 : y;
	const era = floorDiv(shifted, 400);
	const yearOfEra = shifted - era * 400;
	const dayOfYear = floorDiv(153 * (m + (m > 2 ? -3 : 9)) + 2, 5) + d - 1;
	const dayOfEra = yearOfEra * 365 + floorDiv(yearOfEra, 4) - floorDiv(yearOfEra, 100) + dayOfYear;
	return era * 146_097 + dayOfEra - 719_468;
}

export function civilFromDays(days: number): Civil {
	const z = days + 719_468;
	const era = floorDiv(z, 146_097);
	const dayOfEra = z - era * 146_097;
	const yearOfEra = floorDiv(
		dayOfEra - floorDiv(dayOfEra, 1460) + floorDiv(dayOfEra, 36_524) - floorDiv(dayOfEra, 146_096),
		365,
	);
	const dayOfYear = dayOfEra - (365 * yearOfEra + floorDiv(yearOfEra, 4) - floorDiv(yearOfEra, 100));
	const monthIndex = floorDiv(5 * dayOfYear + 2, 153);
	const d = dayOfYear - floorDiv(153 * monthIndex + 2, 5) + 1;
	const m = monthIndex < 10 ? monthIndex + 3 : monthIndex - 9;
	return { y: yearOfEra + era * 400 + (m <= 2 ? 1 : 0), m, d };
}

/** 1 = Monday … 7 = Sunday. 1970-01-01 was a Thursday. */
export function isoWeekday(days: number): number {
	return mod(days + 3, 7) + 1;
}

export function dayOfYear(c: Civil): number {
	return daysFromCivil(c.y, c.m, c.d) - daysFromCivil(c.y, 1, 1) + 1;
}

function weeksInIsoYear(y: number): 52 | 53 {
	const p = (year: number) => mod(year + floorDiv(year, 4) - floorDiv(year, 100) + floorDiv(year, 400), 7);
	return p(y) === 4 || p(y - 1) === 3 ? 53 : 52;
}

/** ISO 8601 week date, e.g. `2026-W45-7`. */
export function isoWeekDate(days: number): string {
	const c = civilFromDays(days);
	const weekday = isoWeekday(days);
	let year = c.y;
	let week = floorDiv(dayOfYear(c) - weekday + 10, 7);
	if (week < 1) {
		year -= 1;
		week = weeksInIsoYear(year);
	} else if (week > weeksInIsoYear(year)) {
		year += 1;
		week = 1;
	}
	return `${pad(year, 4)}-W${pad(week)}-${weekday}`;
}

/** Adds calendar months, clamping the day to the end of a shorter month (Jan 31 + 1 month = Feb 28). */
export function addMonths(c: Civil, months: number): Civil {
	const index = c.y * 12 + (c.m - 1) + months;
	const y = floorDiv(index, 12);
	const m = mod(index, 12) + 1;
	return { y, m, d: Math.min(c.d, daysInMonth(y, m)) };
}

export function formatDate(days: number): string {
	const c = civilFromDays(days);
	return `${pad(c.y, 4)}-${pad(c.m)}-${pad(c.d)}`;
}
