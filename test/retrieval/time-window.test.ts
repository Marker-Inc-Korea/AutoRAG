import { describe, expect, it } from "vitest";
import {
	describeTimeWindow,
	expandedWindowTopK,
	inTimeWindow,
	parseTimestampMs,
	parseTimeWindow,
	resultTimeMs,
	TimeWindowError,
	windowFromOptions,
} from "../../src/retrieval/time-window.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

function result(metadata: Record<string, unknown>): RetrievalResult {
	return { id: "r", content: "c", source: "/s", score: 1, metadata };
}

describe("parseTimeWindow", () => {
	it("returns undefined when neither bound is given", () => {
		expect(parseTimeWindow({})).toBeUndefined();
		expect(parseTimeWindow({ since: "  ", until: undefined })).toBeUndefined();
	});

	it("reads a date-only bound as local midnight, so --since/--until 2026-09-01..2026-10-01 is September", () => {
		const window = parseTimeWindow({ since: "2026-09-01", until: "2026-10-01" });
		expect(window?.since).toBe(new Date(2026, 8, 1).getTime());
		expect(window?.until).toBe(new Date(2026, 9, 1).getTime());
	});

	it("honours an explicit offset and Z", () => {
		expect(parseTimeWindow({ since: "2026-09-01T00:00:00+09:00" })?.since).toBe(Date.UTC(2026, 7, 31, 15));
		expect(parseTimeWindow({ since: "2026-09-01T00:00:00Z" })?.since).toBe(Date.UTC(2026, 8, 1));
		expect(parseTimeWindow({ since: "2026-09-01T00:00:00+0900" })?.since).toBe(Date.UTC(2026, 7, 31, 15));
	});

	it("reads a datetime with no offset as local time", () => {
		expect(parseTimeWindow({ since: "2026-09-01T09:30" })?.since).toBe(new Date(2026, 8, 1, 9, 30).getTime());
	});

	it("accepts a single open-ended bound", () => {
		expect(parseTimeWindow({ since: "2026-09-01" })?.until).toBeUndefined();
		expect(parseTimeWindow({ until: "2026-09-01" })?.since).toBeUndefined();
	});

	it.each([
		["tomorrow-ish", "since"],
		["2026-13-01", "since"],
		["2026-02-31", "until"],
		["2026-09-01T25:00:00Z", "since"],
		["09/01/2026", "until"],
	])("rejects %s verbatim naming the field", (value, field) => {
		const input = field === "since" ? { since: value } : { until: value };
		expect(() => parseTimeWindow(input)).toThrow(TimeWindowError);
		expect(() => parseTimeWindow(input)).toThrow(`Invalid ${field} "${value}"`);
	});

	it("rejects an empty window", () => {
		expect(() => parseTimeWindow({ since: "2026-10-01", until: "2026-09-01" })).toThrow(
			/since .* must be before until/,
		);
		expect(() => parseTimeWindow({ since: "2026-10-01", until: "2026-10-01" })).toThrow(TimeWindowError);
	});
});

describe("resultTimeMs", () => {
	it("prefers publishedAt over the source mtime", () => {
		expect(resultTimeMs(result({ publishedAt: 100, sourceMtimeMs: 900 }))).toBe(100);
	});

	it("falls back to the source mtime for local files", () => {
		expect(resultTimeMs(result({ sourceMtimeMs: 900 }))).toBe(900);
	});

	it("is undefined when neither is a finite number", () => {
		expect(resultTimeMs(result({}))).toBeUndefined();
		expect(resultTimeMs(result({ publishedAt: "2026-09-01", sourceMtimeMs: Number.NaN }))).toBeUndefined();
	});
});

describe("inTimeWindow", () => {
	const window = { since: 100, until: 200 };
	it("is half-open: since inclusive, until exclusive", () => {
		expect(inTimeWindow(100, window)).toBe(true);
		expect(inTimeWindow(199, window)).toBe(true);
		expect(inTimeWindow(200, window)).toBe(false);
		expect(inTimeWindow(99, window)).toBe(false);
	});

	it("treats a missing bound as open", () => {
		expect(inTimeWindow(5, { until: 10 })).toBe(true);
		expect(inTimeWindow(50, { since: 10 })).toBe(true);
	});
});

describe("windowFromOptions", () => {
	it("is undefined with no bound and carries only the set bounds otherwise", () => {
		expect(windowFromOptions({ topK: 3 })).toBeUndefined();
		expect(windowFromOptions({ since: 1 })).toEqual({ since: 1 });
		expect(windowFromOptions({ since: 1, until: 2 })).toEqual({ since: 1, until: 2 });
	});
});

describe("expandedWindowTopK", () => {
	it("asks for more than topK so a window can be post-filtered, bounded by the ceiling", () => {
		expect(expandedWindowTopK(10)).toBe(50);
		expect(expandedWindowTopK(2)).toBe(22);
		expect(expandedWindowTopK(80)).toBe(300);
		// Never less than the caller asked for.
		expect(expandedWindowTopK(1000)).toBe(1000);
	});

	it("uses a documented base when the caller gave no topK", () => {
		expect(expandedWindowTopK(undefined)).toBe(250);
	});
});

describe("parseTimestampMs", () => {
	it("parses ISO and RFC 2822 strings and passes epoch ms through", () => {
		expect(parseTimestampMs("2026-09-01T00:00:00Z")).toBe(Date.UTC(2026, 8, 1));
		expect(parseTimestampMs("Tue, 01 Sep 2026 00:00:00 +0000")).toBe(Date.UTC(2026, 8, 1));
		expect(parseTimestampMs(1_710_000_000_123)).toBe(1_710_000_000_123);
	});

	it("reads a digits-only string as epoch milliseconds, or seconds when too small to be milliseconds", () => {
		expect(parseTimestampMs("1710000000000")).toBe(1_710_000_000_000);
		expect(parseTimestampMs("1710000000")).toBe(1_710_000_000_000);
		expect(parseTimestampMs(1_710_000_000)).toBe(1_710_000_000_000);
		expect(parseTimestampMs("1710000000.5")).toBe(1_710_000_000_500);
	});

	it("is undefined for anything it cannot place", () => {
		expect(parseTimestampMs("not a date")).toBeUndefined();
		expect(parseTimestampMs(undefined)).toBeUndefined();
		expect(parseTimestampMs("")).toBeUndefined();
	});
});

describe("describeTimeWindow", () => {
	it("renders both bounds as ISO instants, open ones as unbounded", () => {
		expect(describeTimeWindow({ since: Date.UTC(2026, 8, 1), until: Date.UTC(2026, 9, 1) })).toBe(
			"[2026-09-01T00:00:00.000Z, 2026-10-01T00:00:00.000Z)",
		);
		expect(describeTimeWindow({ since: Date.UTC(2026, 8, 1) })).toBe("[2026-09-01T00:00:00.000Z, unbounded)");
	});
});
