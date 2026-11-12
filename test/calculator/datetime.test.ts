import { describe, expect, it } from "vitest";
import { CalculatorError } from "../../src/calculator/errors.ts";
import { type CalculatorRequest, calculate } from "../../src/calculator/index.ts";

/** 2026-11-08 22:10:45 in Seoul, a Sunday. */
const NOW = new Date("2026-11-08T13:10:45Z");

const run = (request: CalculatorRequest) => calculate(request, { now: NOW, hostTimeZone: "Asia/Seoul" }).fields;

describe("calculator: now", () => {
	it("reports the clock in the requested zone, with the zone's own offset", () => {
		expect(run({ operation: "now", timeZone: "Asia/Seoul" })).toMatchObject({
			iso: "2026-11-08T22:10:45+09:00",
			date: "2026-11-08",
			weekday: "Sunday",
			timeZone: "Asia/Seoul",
			utcOffset: "+09:00",
			epochSeconds: 1794143445,
			isoWeek: "2026-W45-7",
		});
		expect(run({ operation: "now", timeZone: "America/Los_Angeles" })).toMatchObject({
			iso: "2026-11-08T05:10:45-08:00",
			utcOffset: "-08:00",
			epochSeconds: 1794143445,
		});
		expect(run({ operation: "now", timeZone: "Asia/Kolkata" })).toMatchObject({ iso: "2026-11-08T18:40:45+05:30" });
	});

	it("defaults to the host zone and says so", () => {
		expect(run({ operation: "now" })).toMatchObject({ timeZone: "Asia/Seoul", timeZoneSource: "host default" });
	});

	it("accepts KST, UTC and fixed offsets", () => {
		expect(run({ operation: "now", timeZone: "KST" }).iso).toBe("2026-11-08T22:10:45+09:00");
		expect(run({ operation: "now", timeZone: "UTC" }).iso).toBe("2026-11-08T13:10:45+00:00");
		expect(run({ operation: "now", timeZone: "+05:30" }).iso).toBe("2026-11-08T18:40:45+05:30");
		expect(run({ operation: "now", timeZone: "UTC-8" }).iso).toBe("2026-11-08T05:10:45-08:00");
	});

	it("refuses ambiguous abbreviations instead of guessing", () => {
		expect(() => run({ operation: "now", timeZone: "PST" })).toThrow(/IANA/);
		expect(() => run({ operation: "now", timeZone: "Mars/Base" })).toThrow(/unknown time zone/i);
	});

	it("tells the caller which fixed offset a US abbreviation names, so it never has to swap in a region", () => {
		for (const [abbreviation, offset] of [
			["PST", "-08:00"],
			["PDT", "-07:00"],
			["MST", "-07:00"],
			["MDT", "-06:00"],
			["EST", "-05:00"],
			["EDT", "-04:00"],
		] as const) {
			expect(() => run({ operation: "now", timeZone: abbreviation })).toThrow(
				new RegExp(`"${abbreviation}".*names the fixed offset ${offset}.*America/`),
			);
		}
	});

	it("names no offset for an abbreviation with several meanings, and says to ask", () => {
		for (const abbreviation of ["CST", "IST", "BST"]) {
			expect(() => run({ operation: "now", timeZone: abbreviation })).toThrow(/several time zones.*ask/i);
			expect(() => run({ operation: "now", timeZone: abbreviation })).not.toThrow(/fixed offset [+-]\d/);
		}
	});

	it("takes the offset an abbreviation names, so PST stays UTC-8 in July when Los Angeles is on UTC-7", () => {
		const july = (timeZone: string) =>
			calculate(
				{ operation: "convert_timezone", datetime: "2026-07-01T15:00:00", timeZone, toTimeZone: "Asia/Seoul" },
				{ now: NOW, hostTimeZone: "Asia/Seoul" },
			).fields.result;
		expect(july("-08:00")).toBe("2026-07-02T08:00:00+09:00");
		expect(july("America/Los_Angeles")).toBe("2026-07-02T07:00:00+09:00");
	});
});

describe("calculator: date_diff with dates", () => {
	it.each([
		["2026-01-01", "2026-11-08", 311, "10 months, 7 days"],
		["2024-02-29", "2025-02-28", 365, "1 year"],
		["2026-01-31", "2026-03-01", 29, "1 month, 1 day"],
		["2026-11-08", "2026-11-08", 0, "0 days"],
		["2000-01-01", "2026-11-08", 9808, "26 years, 10 months, 7 days"],
	])("%s → %s is %i days (%s)", (from, to, totalDays, calendar) => {
		expect(run({ operation: "date_diff", from, to })).toMatchObject({ totalDays, calendar });
	});

	it("reports a backward span with the same magnitudes and a negative total", () => {
		expect(run({ operation: "date_diff", from: "2026-11-08", to: "2026-01-01" })).toMatchObject({
			direction: "backward",
			totalDays: -311,
			calendar: "10 months, 7 days",
		});
	});

	it("resolves today and now against the clock in the zone", () => {
		expect(run({ operation: "date_diff", from: "today", to: "2026-12-25" })).toMatchObject({
			from: "2026-11-08",
			totalDays: 47,
		});
		expect(
			run({ operation: "date_diff", from: "2026-11-08", to: "today", timeZone: "Pacific/Auckland" }),
		).toMatchObject({
			to: "2026-11-09",
			totalDays: 1,
		});
	});

	it("reads the date formats people write", () => {
		const iso = run({ operation: "date_diff", from: "2026-01-01", to: "2026-11-08" }).totalDays;
		for (const from of ["2026/01/01", "2026.01.01", "20260101", "2026년 1월 1일"]) {
			expect(run({ operation: "date_diff", from, to: "2026-11-08" }).totalDays).toBe(iso);
		}
	});
});

describe("calculator: date_diff with times", () => {
	it("measures exact elapsed time and a wall-clock breakdown", () => {
		expect(
			run({
				operation: "date_diff",
				from: "2026-11-08T09:00:00",
				to: "2026-11-09T17:30:15",
				timeZone: "Asia/Seoul",
			}),
		).toMatchObject({
			elapsedSeconds: 117015,
			elapsed: "1 day, 8 hours, 30 minutes, 15 seconds",
			calendar: "1 day, 8 hours, 30 minutes, 15 seconds",
		});
	});

	it("keeps a daylight-saving day 23 hours long while the calendar still says one day", () => {
		expect(
			run({
				operation: "date_diff",
				from: "2026-03-08T00:00:00",
				to: "2026-03-09T00:00:00",
				timeZone: "America/New_York",
			}),
		).toMatchObject({ elapsedSeconds: 82800, elapsed: "23 hours", calendar: "1 day" });
	});

	it("honors explicit offsets and epoch seconds in either endpoint", () => {
		expect(
			run({
				operation: "date_diff",
				from: "2026-11-08T22:10:45+09:00",
				to: "2026-11-08T05:10:45-08:00",
			}).elapsedSeconds,
		).toBe(0);
		expect(run({ operation: "date_diff", from: "1794143445", to: "1794147045", timeZone: "UTC" }).elapsed).toBe(
			"1 hour",
		);
	});
});

describe("calculator: date_add", () => {
	it("adds days to a date", () => {
		expect(run({ operation: "date_add", date: "2026-11-08", days: 100 })).toMatchObject({
			result: "2027-02-16",
			weekday: "Tuesday",
		});
		expect(run({ operation: "date_add", date: "2026-11-08", weeks: -2 }).result).toBe("2026-10-25");
	});

	it("clamps to the end of a shorter month, like a calendar does", () => {
		expect(run({ operation: "date_add", date: "2026-01-31", months: 1 })).toMatchObject({
			result: "2026-02-28",
			weekday: "Saturday",
		});
		expect(run({ operation: "date_add", date: "2024-02-29", years: 1 }).result).toBe("2025-02-28");
		expect(run({ operation: "date_add", date: "2026-03-31", months: -1 }).result).toBe("2026-02-28");
	});

	it("adds clock time to a datetime", () => {
		expect(
			run({ operation: "date_add", date: "2026-11-08T22:00:00", timeZone: "Asia/Seoul", hours: 5, minutes: 30 }),
		).toMatchObject({ result: "2026-11-09T03:30:00+09:00" });
	});

	it("moves wall-clock days across a daylight-saving change but exact hours by elapsed time", () => {
		const base = { operation: "date_add", date: "2026-03-07T12:00:00", timeZone: "America/New_York" } as const;
		expect(run({ ...base, days: 1 }).result).toBe("2026-03-08T12:00:00-04:00");
		expect(run({ ...base, hours: 24 }).result).toBe("2026-03-08T13:00:00-04:00");
	});

	it("resolves a skipped local time forward and a repeated one to its first occurrence", () => {
		expect(
			run({ operation: "date_add", date: "2026-03-08T02:30:00", timeZone: "America/New_York", days: 0 }).result,
		).toBe("2026-03-08T03:30:00-04:00");
		expect(
			run({ operation: "date_add", date: "2026-11-01T01:30:00", timeZone: "America/New_York", days: 0 }).result,
		).toBe("2026-11-01T01:30:00-04:00");
	});

	it("rejects a date with a time part when no unit is given and fractional units", () => {
		expect(() => run({ operation: "date_add", date: "2026-11-08" })).toThrow(/at least one/i);
		expect(() => run({ operation: "date_add", date: "2026-11-08", days: 1.5 })).toThrow(/whole number/i);
	});
});

describe("calculator: convert_timezone", () => {
	it("converts an instant that carries its own offset", () => {
		expect(
			run({ operation: "convert_timezone", datetime: "2026-11-08T22:10:45+09:00", toTimeZone: "America/New_York" }),
		).toMatchObject({
			result: "2026-11-08T08:10:45-05:00",
			epochSeconds: 1794143445,
			toTimeZone: "America/New_York",
		});
	});

	it("converts a local time given with its source zone", () => {
		expect(
			run({
				operation: "convert_timezone",
				datetime: "2026-07-01T09:00:00",
				timeZone: "Europe/London",
				toTimeZone: "Asia/Seoul",
			}).result,
		).toBe("2026-07-01T17:00:00+09:00");
	});

	it("will not guess the source zone of a bare local time", () => {
		expect(() =>
			run({ operation: "convert_timezone", datetime: "2026-07-01T09:00:00", toTimeZone: "Asia/Seoul" }),
		).toThrow(/source time zone/i);
		expect(() => run({ operation: "convert_timezone", datetime: "2026-07-01T09:00:00Z" })).toThrow(/toTimeZone/);
	});
});

describe("calculator: date_info", () => {
	it("describes a date", () => {
		expect(run({ operation: "date_info", date: "2026-11-08" })).toMatchObject({
			weekday: "Sunday",
			dayOfYear: 312,
			isoWeek: "2026-W45-7",
			quarter: 4,
			daysInMonth: 30,
			leapYear: false,
			daysLeftInYear: 53,
		});
		expect(run({ operation: "date_info", date: "2024-02-29" })).toMatchObject({
			weekday: "Thursday",
			dayOfYear: 60,
			isoWeek: "2024-W09-4",
			leapYear: true,
			daysInMonth: 29,
		});
		expect(run({ operation: "date_info", date: "2026-12-31" }).isoWeek).toBe("2026-W53-4");
	});
});

describe("calculator: business days", () => {
	it("counts weekdays from the earlier date up to, not including, the later date", () => {
		expect(run({ operation: "business_days_between", from: "2026-01-01", to: "2026-11-08" })).toMatchObject({
			businessDays: 222,
			calendarDays: 311,
		});
		expect(run({ operation: "business_days_between", from: "2024-02-29", to: "2025-02-28" }).businessDays).toBe(261);
	});

	it("counts both dates on request, and a backward span is the negated forward count", () => {
		const span = { operation: "business_days_between", from: "2026-11-06", to: "2026-11-09" } as const;
		expect(run(span).businessDays).toBe(1);
		expect(run({ ...span, inclusive: true }).businessDays).toBe(2);
		expect(run({ operation: "business_days_between", from: "2026-11-09", to: "2026-11-06" }).businessDays).toBe(-1);
		expect(
			run({ operation: "business_days_between", from: "2026-11-09", to: "2026-11-06", inclusive: true })
				.businessDays,
		).toBe(-2);
	});

	it("skips listed holidays once, and ignores holidays on weekends or outside the span", () => {
		const base = { operation: "business_days_between", from: "2026-01-01", to: "2026-11-08" } as const;
		expect(run({ ...base, holidays: ["2026-01-01", "2026-01-01"] }).businessDays).toBe(221);
		expect(run({ ...base, holidays: ["2026-01-03", "2025-12-25", "2026-12-25"] }).businessDays).toBe(222);
	});

	it("adds and subtracts business days", () => {
		expect(run({ operation: "add_business_days", date: "2026-11-06", days: 1 })).toMatchObject({
			result: "2026-11-09",
			weekday: "Monday",
		});
		expect(run({ operation: "add_business_days", date: "2026-11-06", days: 5 }).result).toBe("2026-11-13");
		expect(run({ operation: "add_business_days", date: "2026-11-09", days: -1 }).result).toBe("2026-11-06");
		expect(
			run({ operation: "add_business_days", date: "2026-11-06", days: 1, holidays: ["2026-11-09"] }).result,
		).toBe("2026-11-10");
		expect(run({ operation: "add_business_days", date: "2026-11-07", days: 1 }).result).toBe("2026-11-09");
	});
});

describe("calculator: input validation", () => {
	it.each([
		[{ operation: "date_info", date: "2026-02-30" }, /not a real date/i],
		[{ operation: "date_info", date: "2026-13-01" }, /not a real date/i],
		[{ operation: "date_info", date: "yesterday" }, /could not read/i],
		[{ operation: "date_info", date: "2026-11-08T25:00:00" }, /not a real time/i],
		[{ operation: "date_info" }, /requires "date"/],
		[{ operation: "date_diff", from: "2026-01-01" }, /requires "to"/],
		[{ operation: "add_business_days", date: "2026-11-06", days: 2.5 }, /whole number/i],
		[{ operation: "add_business_days", date: "2026-11-06" }, /requires "days"/],
		[{ operation: "business_days_between", from: "2026-01-01", to: "2026-01-02", holidays: ["soon"] }, /holiday/i],
		[{ operation: "evaluate" }, /requires "expression"/],
		[{ operation: "nonsense" }, /unknown operation/i],
	] as [CalculatorRequest, RegExp][])("%j fails clearly", (request, message) => {
		expect(() => run(request)).toThrow(CalculatorError);
		expect(() => run(request)).toThrow(message);
	});
});

describe("calculator: evaluate and rendering", () => {
	it("evaluates expressions through the same entry point", () => {
		expect(run({ operation: "evaluate", expression: "0.1 + 0.2" })).toMatchObject({ value: "0.3", exact: true });
	});

	it("renders one `name: value` line per field for the model", () => {
		const { text } = calculate({ operation: "date_diff", from: "2026-01-01", to: "2026-11-08" }, { now: NOW });
		expect(text).toContain("totalDays: 311");
		expect(text).toContain("calendar: 10 months, 7 days");
		expect(text.split("\n").every((line: string) => /^[A-Za-z]+: .+/.test(line))).toBe(true);
	});

	it("marks an evaluated result that had to be rounded", () => {
		const { text } = calculate({ operation: "evaluate", expression: "1/3" }, { now: NOW });
		expect(text).toContain("value: 0.3333333333333333333333333333333333333333");
		expect(text).toMatch(/exact: false/);
	});
});
