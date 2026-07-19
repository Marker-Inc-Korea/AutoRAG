import { describe, expect, it } from "vitest";
import { type CronParseResult, isDue, parseCronExpr } from "../../src/datasource/polling.ts";
import type { PollingMetadata } from "../../src/datasource/types.ts";

const poll = (overrides: Partial<PollingMetadata> = {}): PollingMetadata => ({
	mode: "poll",
	intervalMs: 60_000,
	lastIndexedAt: 1_000,
	...overrides,
});

describe("isDue", () => {
	describe("poll mode arithmetic", () => {
		it("is due when never indexed", () => {
			const meta = poll({ lastIndexedAt: undefined });
			expect(isDue(meta, 5_000)).toBe(true);
		});

		it("is due when the interval has elapsed", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 1_000 });
			expect(isDue(meta, 61_000)).toBe(true);
			expect(isDue(meta, 1_000 + 60_000)).toBe(true);
		});

		it("is not due before the interval elapses", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 1_000 });
			expect(isDue(meta, 30_000)).toBe(false);
			expect(isDue(meta, 1_000 + 59_999)).toBe(false);
		});

		it("is not due exactly one millisecond before the interval", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 0 });
			expect(isDue(meta, 59_999)).toBe(false);
		});
	});

	describe("non-poll mode", () => {
		it("is never due when mode is not 'poll'", () => {
			const meta: PollingMetadata = { mode: "none", intervalMs: 60_000 };
			expect(isDue(meta, 10_000_000)).toBe(false);
		});
	});

	describe("misconfigured interval", () => {
		it("is never due when intervalMs is undefined", () => {
			const meta = poll({ intervalMs: undefined });
			expect(isDue(meta, 10_000_000)).toBe(false);
		});

		it("is never due when intervalMs is non-positive", () => {
			expect(isDue(poll({ intervalMs: 0 }), 10_000)).toBe(false);
			expect(isDue(poll({ intervalMs: -5 }), 10_000)).toBe(false);
		});
	});

	describe("explicit lastIndexedAt override", () => {
		it("uses the explicit lastIndexedAt when provided", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 5_000 });
			// Explicit override of 1_000 takes precedence over metadata's 5_000.
			expect(isDue(meta, 61_000, 1_000)).toBe(true);
			expect(isDue(meta, 30_000, 1_000)).toBe(false);
		});

		it("falls back to metadata.lastIndexedAt when override is undefined", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 1_000 });
			expect(isDue(meta, 61_000)).toBe(true);
		});
	});

	describe("non-finite timestamps", () => {
		it("is not due when `now` is non-finite", () => {
			const meta = poll({ intervalMs: 60_000, lastIndexedAt: 1_000 });
			expect(isDue(meta, Number.NaN, 1_000)).toBe(false);
			expect(isDue(meta, Number.POSITIVE_INFINITY, 1_000)).toBe(false);
		});
	});
});

describe("parseCronExpr", () => {
	const okCases: readonly string[] = [
		"*/5 * * * *",
		"0 9 * * 1-5",
		"0 0 1 * *",
		"30 4 * * 0",
		"30 4 * * 7", // 7 == Sunday
		"0,15,30,45 * * * *",
		"10-20/5 * * * *",
		"*/15 0-23 * 1-12 0-7",
	];

	for (const expr of okCases) {
		it(`accepts "${expr}"`, () => {
			const result: CronParseResult = parseCronExpr(expr);
			expect(result.ok).toBe(true);
			if (result.ok) expect(result.expr).toBe(expr.trim());
		});
	}

	it("accepts surrounding whitespace by trimming", () => {
		const result = parseCronExpr("   0 9 * * *   ");
		expect(result.ok).toBe(true);
		if (result.ok) expect(result.expr).toBe("0 9 * * *");
	});

	describe("invalid syntax", () => {
		it("rejects an empty string", () => {
			const result = parseCronExpr("");
			expect(result.ok).toBe(false);
			if (!result.ok) expect(result.error).toMatch(/empty/);
		});

		it("rejects a string with the wrong number of fields", () => {
			const result = parseCronExpr("0 9 * *");
			expect(result.ok).toBe(false);
			if (!result.ok) expect(result.error).toMatch(/5 fields/);
		});

		it("rejects too many fields", () => {
			const result = parseCronExpr("0 9 * * * 2025");
			expect(result.ok).toBe(false);
		});

		it("rejects an out-of-bounds minute", () => {
			const result = parseCronExpr("60 * * * *");
			expect(result.ok).toBe(false);
			if (!result.ok) expect(result.error).toMatch(/minute/);
		});

		it("rejects an out-of-bounds hour", () => {
			const result = parseCronExpr("* 24 * * *");
			expect(result.ok).toBe(false);
		});

		it("rejects an out-of-bounds day-of-month (0 not allowed)", () => {
			const result = parseCronExpr("* * 0 * *");
			expect(result.ok).toBe(false);
		});

		it("rejects an out-of-bounds month", () => {
			const result = parseCronExpr("* * * 13 *");
			expect(result.ok).toBe(false);
		});

		it("rejects an out-of-bounds day-of-week (8 not allowed)", () => {
			const result = parseCronExpr("* * * * 8");
			expect(result.ok).toBe(false);
		});

		it("rejects a non-numeric value", () => {
			const result = parseCronExpr("abc * * * *");
			expect(result.ok).toBe(false);
		});

		it("rejects a range with low > high", () => {
			const result = parseCronExpr("* 5-1 * * *");
			expect(result.ok).toBe(false);
			if (!result.ok) expect(result.error).toMatch(/low > high/);
		});

		it("rejects a non-positive step", () => {
			const result = parseCronExpr("*/0 * * * *");
			expect(result.ok).toBe(false);
			if (!result.ok) expect(result.error).toMatch(/step/);
		});

		it("rejects extended L/W/? syntax (unsupported)", () => {
			expect(parseCronExpr("0 9 15W * ?").ok).toBe(false);
			expect(parseCronExpr("0 9 * * 5L").ok).toBe(false);
		});
	});
});
