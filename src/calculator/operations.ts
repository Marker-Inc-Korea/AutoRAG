import {
	addMonths,
	civilFromDays,
	dayOfYear,
	daysFromCivil,
	daysInMonth,
	floorDiv,
	formatDate,
	isLeapYear,
	isoWeekDate,
	isoWeekday,
	MAX_YEAR,
	MIN_YEAR,
	MS_PER_DAY,
	mod,
	WEEKDAYS,
} from "./civil.ts";
import { CalculatorError } from "./errors.ts";
import { dayOf, instantOf, parseTemporal, type Temporal } from "./temporal.ts";
import { formatIso, formatOffset, fromNaive, offsetAt, parseZone, toNaive, type Zone } from "./timezone.ts";

export type FieldValue = string | number | boolean;
export type Fields = Record<string, FieldValue>;

export interface CalculatorRequest {
	readonly operation: string;
	readonly expression?: string;
	/** Zone a date or time is read in and shown in; for convert_timezone, the source zone. */
	readonly timeZone?: string;
	readonly toTimeZone?: string;
	readonly date?: string;
	readonly datetime?: string;
	readonly from?: string;
	readonly to?: string;
	readonly years?: number;
	readonly months?: number;
	readonly weeks?: number;
	readonly days?: number;
	readonly hours?: number;
	readonly minutes?: number;
	readonly seconds?: number;
	readonly holidays?: readonly string[];
	readonly inclusive?: boolean;
}

export interface Clock {
	readonly now: Date;
	readonly zone: Zone;
	readonly zoneSource: "requested" | "host default";
}

const MAX_UNIT = 1_000_000;
const MAX_BUSINESS_STEP = 100_000;
const MAX_HOLIDAYS = 1000;
const MS_PER_SECOND = 1000;

export function need<T>(operation: string, name: string, value: T | undefined): T {
	if (value === undefined || (typeof value === "string" && value.trim() === "")) {
		throw new CalculatorError(`${operation} requires "${name}".`);
	}
	return value;
}

function wholeNumber(name: string, value: number): number {
	if (!Number.isInteger(value) || Math.abs(value) > MAX_UNIT) {
		throw new CalculatorError(`${name} must be a whole number between -${MAX_UNIT} and ${MAX_UNIT}.`);
	}
	return value;
}

function plural(count: number, unit: string): string {
	return `${count} ${unit}${count === 1 ? "" : "s"}`;
}

function describeParts(parts: readonly (readonly [number, string])[], zeroUnit: string): string {
	const shown = parts.filter(([count]) => count !== 0).map(([count, unit]) => plural(count, unit));
	return shown.length === 0 ? plural(0, zeroUnit) : shown.join(", ");
}

function checkYear(days: number): number {
	const year = civilFromDays(days).y;
	if (year < MIN_YEAR || year > MAX_YEAR) {
		throw new CalculatorError(`The result is outside the supported years ${MIN_YEAR}-${MAX_YEAR}.`);
	}
	return days;
}

function weekdayName(days: number): string {
	return WEEKDAYS[isoWeekday(days) - 1] as string;
}

function zoneFields(clock: Clock, zone: Zone = clock.zone): Fields {
	return { timeZone: zone.name, ...(zone === clock.zone ? { timeZoneSource: clock.zoneSource } : {}) };
}

export function now(clock: Clock): Fields {
	const epochMs = clock.now.getTime();
	const days = floorDiv(toNaive(clock.zone, epochMs), MS_PER_DAY);
	const iso = formatIso(clock.zone, epochMs);
	return {
		iso: iso.replace(/\.\d{3}(?=[+-])/, ""),
		date: formatDate(days),
		time: iso.slice(11, 19),
		weekday: weekdayName(days),
		...zoneFields(clock),
		utcOffset: formatOffset(offsetAt(clock.zone, epochMs)),
		epochSeconds: Math.floor(epochMs / MS_PER_SECOND),
		isoWeek: isoWeekDate(days),
	};
}

interface Span {
	readonly years: number;
	readonly months: number;
	readonly days: number;
	readonly hours: number;
	readonly minutes: number;
	readonly seconds: number;
}

/** Whole calendar months first (day clamped like a calendar), then the days and clock time left over. */
function calendarSpan(naiveFrom: number, naiveTo: number): Span {
	const startDay = floorDiv(naiveFrom, MS_PER_DAY);
	const startClock = naiveFrom - startDay * MS_PER_DAY;
	const start = civilFromDays(startDay);
	const end = civilFromDays(floorDiv(naiveTo, MS_PER_DAY));
	const anchorOf = (months: number) => {
		const c = addMonths(start, months);
		return daysFromCivil(c.y, c.m, c.d) * MS_PER_DAY + startClock;
	};
	let months = (end.y - start.y) * 12 + (end.m - start.m);
	while (months > 0 && anchorOf(months) > naiveTo) months--;
	const rest = naiveTo - anchorOf(months);
	const days = floorDiv(rest, MS_PER_DAY);
	const clock = Math.floor((rest - days * MS_PER_DAY) / MS_PER_SECOND);
	return {
		years: floorDiv(months, 12),
		months: months % 12,
		days,
		hours: floorDiv(clock, 3600),
		minutes: floorDiv(clock % 3600, 60),
		seconds: clock % 60,
	};
}

function describeSpan(span: Span, zeroUnit: string): string {
	return describeParts(
		[
			[span.years, "year"],
			[span.months, "month"],
			[span.days, "day"],
			[span.hours, "hour"],
			[span.minutes, "minute"],
			[span.seconds, "second"],
		],
		zeroUnit,
	);
}

function describeElapsed(totalSeconds: number): string {
	const seconds = Math.abs(totalSeconds);
	return describeParts(
		[
			[floorDiv(seconds, 86_400), "day"],
			[floorDiv(seconds % 86_400, 3600), "hour"],
			[floorDiv(seconds % 3600, 60), "minute"],
			[Math.floor(seconds % 60), "second"],
		],
		"second",
	);
}

function read(operation: string, name: string, value: string | undefined, clock: Clock): Temporal {
	return parseTemporal(need(operation, name, value), clock.zone, clock.now);
}

export function dateDiff(request: CalculatorRequest, clock: Clock): Fields {
	const from = read("date_diff", "from", request.from, clock);
	const to = read("date_diff", "to", request.to, clock);
	if (from.kind === "date" && to.kind === "date") {
		const totalDays = to.days - from.days;
		const [early, late] = totalDays >= 0 ? [from.days, to.days] : [to.days, from.days];
		const span = calendarSpan(early * MS_PER_DAY, late * MS_PER_DAY);
		return {
			from: formatDate(from.days),
			to: formatDate(to.days),
			direction: totalDays >= 0 ? "forward" : "backward",
			totalDays,
			calendar: describeSpan(span, "day"),
		};
	}
	const fromMs = instantOf(from, clock.zone);
	const toMs = instantOf(to, clock.zone);
	const elapsedSeconds = (toMs - fromMs) / MS_PER_SECOND;
	const [early, late] = toMs >= fromMs ? [fromMs, toMs] : [toMs, fromMs];
	const span = calendarSpan(toNaive(clock.zone, early), toNaive(clock.zone, late));
	return {
		from: formatIso(clock.zone, fromMs),
		to: formatIso(clock.zone, toMs),
		...zoneFields(clock),
		direction: toMs >= fromMs ? "forward" : "backward",
		elapsedSeconds,
		elapsed: describeElapsed(elapsedSeconds),
		calendar: describeSpan(span, "second"),
	};
}

export function dateAdd(request: CalculatorRequest, clock: Clock): Fields {
	const start = read("date_add", "date", request.date, clock);
	const units = {
		years: wholeNumber("years", request.years ?? 0),
		months: wholeNumber("months", request.months ?? 0),
		weeks: wholeNumber("weeks", request.weeks ?? 0),
		days: wholeNumber("days", request.days ?? 0),
		hours: wholeNumber("hours", request.hours ?? 0),
		minutes: wholeNumber("minutes", request.minutes ?? 0),
		seconds: wholeNumber("seconds", request.seconds ?? 0),
	};
	const given = [
		request.years,
		request.months,
		request.weeks,
		request.days,
		request.hours,
		request.minutes,
		request.seconds,
	];
	if (given.every((value) => value === undefined)) {
		throw new CalculatorError("date_add needs at least one of years, months, weeks, days, hours, minutes, seconds.");
	}
	const elapsedMs = ((units.hours * 60 + units.minutes) * 60 + units.seconds) * MS_PER_SECOND;
	const calendarDays = units.weeks * 7 + units.days;
	if (start.kind === "date" && elapsedMs === 0) {
		const c = addMonths(civilFromDays(start.days), units.years * 12 + units.months);
		const result = checkYear(daysFromCivil(c.y, c.m, c.d) + calendarDays);
		return { date: formatDate(start.days), result: formatDate(result), weekday: weekdayName(result) };
	}
	// Calendar units move the wall clock; hours, minutes and seconds then move the timeline.
	const startMs = instantOf(start, clock.zone);
	const naive = toNaive(clock.zone, startMs);
	const day = floorDiv(naive, MS_PER_DAY);
	const c = addMonths(civilFromDays(day), units.years * 12 + units.months);
	const movedDay = checkYear(daysFromCivil(c.y, c.m, c.d) + calendarDays);
	const resultMs = fromNaive(clock.zone, movedDay * MS_PER_DAY + (naive - day * MS_PER_DAY)) + elapsedMs;
	const resultDay = checkYear(floorDiv(toNaive(clock.zone, resultMs), MS_PER_DAY));
	return {
		date: formatIso(clock.zone, startMs),
		result: formatIso(clock.zone, resultMs),
		weekday: weekdayName(resultDay),
		...zoneFields(clock),
	};
}

export function convertTimeZone(request: CalculatorRequest, clock: Clock, sourceGiven: boolean): Fields {
	const text = need("convert_timezone", "datetime", request.datetime);
	const target = parseZone(need("convert_timezone", "toTimeZone", request.toTimeZone));
	const value = parseTemporal(text, clock.zone, clock.now);
	if (value.kind === "date") {
		throw new CalculatorError(`convert_timezone needs a time of day, but "${text}" is only a date.`);
	}
	if (value.zoneless && !sourceGiven) {
		throw new CalculatorError(
			`"${text}" has no UTC offset. Give the source time zone in "timeZone" (for example Asia/Seoul), or add an offset such as +09:00 to the time.`,
		);
	}
	const targetDay = floorDiv(toNaive(target, value.epochMs), MS_PER_DAY);
	return {
		result: formatIso(target, value.epochMs),
		toTimeZone: target.name,
		utcOffset: formatOffset(offsetAt(target, value.epochMs)),
		weekday: weekdayName(targetDay),
		epochSeconds: Math.floor(value.epochMs / MS_PER_SECOND),
		...(sourceGiven || !value.zoneless ? { source: formatIso(clock.zone, value.epochMs) } : {}),
	};
}

export function dateInfo(request: CalculatorRequest, clock: Clock): Fields {
	const days = dayOf(read("date_info", "date", request.date, clock), clock.zone);
	const c = civilFromDays(days);
	return {
		date: formatDate(days),
		weekday: weekdayName(days),
		dayOfYear: dayOfYear(c),
		isoWeek: isoWeekDate(days),
		quarter: Math.floor((c.m - 1) / 3) + 1,
		daysInMonth: daysInMonth(c.y, c.m),
		leapYear: isLeapYear(c.y),
		daysLeftInYear: (isLeapYear(c.y) ? 366 : 365) - dayOfYear(c),
	};
}

/** Weekdays (Monday to Friday) in all days before `days`, counted from an arbitrary Monday-aligned origin. */
function weekdaysBefore(days: number): number {
	const k = days + 3;
	return floorDiv(k, 7) * 5 + Math.min(mod(k, 7), 5);
}

function holidaySet(request: CalculatorRequest, clock: Clock): Set<number> {
	const holidays = request.holidays ?? [];
	if (holidays.length > MAX_HOLIDAYS) throw new CalculatorError(`At most ${MAX_HOLIDAYS} holidays are supported.`);
	const set = new Set<number>();
	for (const holiday of holidays) {
		try {
			set.add(dayOf(parseTemporal(holiday, clock.zone, clock.now), clock.zone));
		} catch (error) {
			if (error instanceof CalculatorError)
				throw new CalculatorError(`Invalid holiday "${holiday}": ${error.message}`);
			throw error;
		}
	}
	return set;
}

export function businessDaysBetween(request: CalculatorRequest, clock: Clock): Fields {
	const from = dayOf(read("business_days_between", "from", request.from, clock), clock.zone);
	const to = dayOf(read("business_days_between", "to", request.to, clock), clock.zone);
	const holidays = holidaySet(request, clock);
	const [low, high] = from <= to ? [from, to] : [to, from];
	const end = high + (request.inclusive === true ? 1 : 0);
	const skipped = [...holidays].filter((day) => day >= low && day < end && isoWeekday(day) <= 5).length;
	const forward = weekdaysBefore(end) - weekdaysBefore(low) - skipped;
	return {
		from: formatDate(from),
		to: formatDate(to),
		businessDays: from <= to ? forward : -forward,
		calendarDays: to - from,
		counted: request.inclusive === true ? "both dates" : "start date up to, not including, the end date",
		holidaysSkipped: skipped,
	};
}

export function addBusinessDays(request: CalculatorRequest, clock: Clock): Fields {
	const start = dayOf(read("add_business_days", "date", request.date, clock), clock.zone);
	const count = need("add_business_days", "days", request.days);
	if (!Number.isInteger(count) || Math.abs(count) > MAX_BUSINESS_STEP) {
		throw new CalculatorError(`days must be a whole number between -${MAX_BUSINESS_STEP} and ${MAX_BUSINESS_STEP}.`);
	}
	const holidays = holidaySet(request, clock);
	const step = count < 0 ? -1 : 1;
	let day = start;
	for (let remaining = Math.abs(count); remaining > 0; ) {
		day = checkYear(day + step);
		if (isoWeekday(day) <= 5 && !holidays.has(day)) remaining--;
	}
	return { date: formatDate(start), result: formatDate(day), weekday: weekdayName(day) };
}
