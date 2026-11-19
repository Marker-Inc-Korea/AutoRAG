import { civilFromDays, daysFromCivil, floorDiv, MS_PER_DAY, mod, pad } from "./civil.ts";
import { CalculatorError } from "./errors.ts";

/** A UTC offset in force at every instant (`fixed`, seconds), or an IANA zone whose offset changes over time. */
export interface Zone {
	readonly name: string;
	readonly fixed?: number;
}

/** Zones without daylight saving whose abbreviation names exactly one place. */
const UNAMBIGUOUS_ABBREVIATIONS: Readonly<Record<string, string>> = { KST: "Asia/Seoul", JST: "Asia/Tokyo" };

/**
 * US abbreviations whose offset never changes with the date: PST is UTC-8 in July too, while the region it is
 * usually quoted for (America/Los_Angeles) is on UTC-7 then. The tool still refuses them, but it names the offset so the
 * caller passes that, instead of swapping in a region that follows daylight saving.
 */
const US_ABBREVIATION_OFFSETS: Readonly<Record<string, string>> = {
	PST: "-08:00",
	PDT: "-07:00",
	MST: "-07:00",
	MDT: "-06:00",
	EST: "-05:00",
	EDT: "-04:00",
};

function abbreviationMessage(text: string, upper: string): string {
	if (Object.hasOwn(US_ABBREVIATION_OFFSETS, upper)) {
		const offset = US_ABBREVIATION_OFFSETS[upper] as string;
		return `The time zone abbreviation "${text}" is not accepted, but it names the fixed offset ${offset} on every date. Pass "${offset}" to keep that meaning; pass an IANA region such as America/Los_Angeles only when the user means local time there, where the offset changes with daylight saving.`;
	}
	return `The time zone abbreviation "${text}" is used for several time zones with different offsets. Ask the user which one they mean, then pass an IANA name such as Asia/Kolkata or a fixed offset such as +05:30.`;
}
const formatters = new Map<string, Intl.DateTimeFormat>();

function ianaZone(name: string): Zone {
	try {
		const formatter = new Intl.DateTimeFormat("en-US", {
			timeZone: name,
			hourCycle: "h23",
			year: "numeric",
			month: "numeric",
			day: "numeric",
			hour: "numeric",
			minute: "numeric",
			second: "numeric",
		});
		const canonical = formatter.resolvedOptions().timeZone;
		formatters.set(canonical, formatter);
		return { name: canonical };
	} catch {
		throw new CalculatorError(
			`Unknown time zone "${name}". Use an IANA name such as Asia/Seoul or a fixed offset such as +09:00.`,
		);
	}
}

export function formatOffset(seconds: number): string {
	const magnitude = Math.abs(seconds);
	const base = `${seconds < 0 ? "-" : "+"}${pad(floorDiv(magnitude, 3600))}:${pad(floorDiv(mod(magnitude, 3600), 60))}`;
	return magnitude % 60 === 0 ? base : `${base}:${pad(magnitude % 60)}`;
}

export function fixedZone(offsetSeconds: number): Zone {
	return { name: offsetSeconds === 0 ? "UTC" : formatOffset(offsetSeconds), fixed: offsetSeconds };
}

export function parseZone(input: string): Zone {
	const text = input.trim();
	const upper = text.toUpperCase();
	if (upper === "UTC" || upper === "GMT" || upper === "UT" || upper === "Z") return fixedZone(0);
	const offset = /^(?:UTC|GMT)?([+-])(\d{1,2})(?::?(\d{2}))?$/i.exec(text);
	if (offset !== null) {
		const hours = Number(offset[2]);
		const minutes = Number(offset[3] ?? 0);
		if (hours > 14 || minutes > 59) throw new CalculatorError(`"${text}" is not a valid UTC offset.`);
		return fixedZone((offset[1] === "-" ? -1 : 1) * (hours * 3600 + minutes * 60));
	}
	if (Object.hasOwn(UNAMBIGUOUS_ABBREVIATIONS, upper)) return ianaZone(UNAMBIGUOUS_ABBREVIATIONS[upper] as string);
	if (/^[A-Z]{3,5}$/.test(text)) throw new CalculatorError(abbreviationMessage(text, upper));
	if (text === "") throw new CalculatorError("The time zone is empty.");
	return ianaZone(text);
}

/** Seconds east of UTC in force at `epochMs`. */
export function offsetAt(zone: Zone, epochMs: number): number {
	if (zone.fixed !== undefined) return zone.fixed;
	let formatter = formatters.get(zone.name);
	if (formatter === undefined) {
		ianaZone(zone.name);
		formatter = formatters.get(zone.name) as Intl.DateTimeFormat;
	}
	const seconds = Math.floor(epochMs / 1000);
	const fields: Record<string, number> = {};
	for (const part of formatter.formatToParts(new Date(seconds * 1000))) {
		if (part.type !== "literal") fields[part.type] = Number(part.value);
	}
	const naiveSeconds =
		daysFromCivil(fields.year as number, fields.month as number, fields.day as number) * 86_400 +
		(fields.hour as number) * 3600 +
		(fields.minute as number) * 60 +
		(fields.second as number);
	return naiveSeconds - seconds;
}

/** Wall-clock time in the zone, as milliseconds counted from 1970-01-01T00:00 with no zone attached. */
export function toNaive(zone: Zone, epochMs: number): number {
	return epochMs + offsetAt(zone, epochMs) * 1000;
}

/**
 * The instant a wall-clock time denotes. A time that occurs twice (clocks turn back) is its first
 * occurrence; a time that never occurs (clocks jump ahead) moves forward by the length of the gap.
 */
export function fromNaive(zone: Zone, naiveMs: number): number {
	if (zone.fixed !== undefined) return naiveMs - zone.fixed * 1000;
	const before = offsetAt(zone, naiveMs - MS_PER_DAY);
	const after = offsetAt(zone, naiveMs + MS_PER_DAY);
	const instants = [...new Set([before, after])]
		.map((offset) => ({ offset, instant: naiveMs - offset * 1000 }))
		.filter(({ offset, instant }) => offsetAt(zone, instant) === offset)
		.map(({ instant }) => instant);
	return instants.length > 0 ? Math.min(...instants) : naiveMs - before * 1000;
}

/** `YYYY-MM-DDTHH:MM:SS[.mmm]±HH:MM` for an instant in the zone. */
export function formatIso(zone: Zone, epochMs: number): string {
	const naive = toNaive(zone, epochMs);
	const days = floorDiv(naive, MS_PER_DAY);
	const c = civilFromDays(days);
	const ofDay = naive - days * MS_PER_DAY;
	const ms = ofDay % 1000;
	const seconds = floorDiv(ofDay, 1000);
	const clock = `${pad(floorDiv(seconds, 3600))}:${pad(floorDiv(seconds % 3600, 60))}:${pad(seconds % 60)}`;
	return `${pad(c.y, 4)}-${pad(c.m)}-${pad(c.d)}T${clock}${ms === 0 ? "" : `.${pad(ms, 3)}`}${formatOffset(offsetAt(zone, epochMs))}`;
}
