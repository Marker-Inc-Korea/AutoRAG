import { describe, expect, it } from "vitest";
import {
	cutByRecord,
	fairShares,
	OVERFETCH_ABSOLUTE_CAP,
	RecordCut,
	recordCutFromOptions,
} from "../../src/retrieval/record-cut.ts";

interface Hit {
	readonly record: string | undefined;
	readonly n: number;
}

/** `count` ranked hits of one record, numbered from `start`. */
function hits(record: string | undefined, count: number, start = 0): Hit[] {
	return Array.from({ length: count }, (_, index) => ({ record, n: start + index }));
}

const recordOf = (hit: Hit): string | undefined => hit.record;

describe("cutByRecord", () => {
	it("returns everything when the pool already fits the limit", () => {
		const pool = [...hits("a", 3), ...hits("b", 2, 3)];
		expect(cutByRecord(pool, 10, { perRecord: 2, reserve: 5, recordOf })).toEqual(pool);
	});

	it("keeps everything for a limit below 1", () => {
		const pool = hits("a", 5);
		expect(cutByRecord(pool, 0, { perRecord: 1, reserve: 0, recordOf })).toHaveLength(5);
	});

	it("lets a long record fill only its per-record share of the head, then reserves slots for unseen records", () => {
		// 20 passages of a long document outrank a keyword-only hit from "b" and "c".
		const pool = [...hits("long", 20), ...hits("b", 1, 20), ...hits("c", 1, 21)];
		const kept = cutByRecord(pool, 8, { perRecord: 3, reserve: 4, recordOf });

		expect(kept).toHaveLength(8);
		const byRecord = (record: string) => kept.filter((hit) => hit.record === record).length;
		expect(byRecord("b")).toBe(1);
		expect(byRecord("c")).toBe(1);
		// head = 8 - 4 = 4: long takes 3 (its cap) and b the 4th; the reserve pass adds c, and the
		// fill pass tops up with long's next-best passages in rank order.
		expect(kept.map((hit) => hit.n)).toEqual([0, 1, 2, 3, 4, 5, 20, 21]);
		expect(byRecord("long")).toBe(6);
	});

	it("gives the reserve to the best hit of each record not yet represented, in rank order", () => {
		const pool = [...hits("a", 10), ...hits("b", 1, 10), ...hits("c", 2, 11), ...hits("d", 1, 13)];
		const kept = cutByRecord(pool, 6, { perRecord: 2, reserve: 3, recordOf });

		// head (3): a0, a1 (a is capped at 2), then b. Reserve: c's best, d's best. Fill: a2.
		expect(kept.map((hit) => hit.n)).toEqual([0, 1, 2, 10, 11, 13]);
		expect(new Set(kept.map((hit) => hit.record))).toEqual(new Set(["a", "b", "c", "d"]));
	});

	it("fills the rest in rank order when the pool has fewer records than reserved slots", () => {
		const pool = hits("only", 12);
		const kept = cutByRecord(pool, 5, { perRecord: 2, reserve: 3, recordOf });
		expect(kept.map((hit) => hit.n)).toEqual([0, 1, 2, 3, 4]);
	});

	it("keeps the survivors in their original order", () => {
		const pool = [...hits("a", 5), ...hits("b", 1, 5), ...hits("a", 1, 6), ...hits("c", 1, 7)];
		const kept = cutByRecord(pool, 4, { perRecord: 2, reserve: 2, recordOf });
		const ranks = kept.map((hit) => hit.n);
		expect(ranks).toEqual([...ranks].sort((x, y) => x - y));
	});

	it("treats a hit with no record as a record of its own", () => {
		const pool = [...hits(undefined, 6)];
		const kept = cutByRecord(pool, 4, { perRecord: 1, reserve: 0, recordOf });
		expect(kept).toHaveLength(4);
		expect(kept.map((hit) => hit.n)).toEqual([0, 1, 2, 3]);
	});

	it("clamps a per-record count below 1 and a reserve outside 0..limit", () => {
		const pool = [...hits("a", 6), ...hits("b", 1, 6)];
		expect(cutByRecord(pool, 3, { perRecord: 0, reserve: -4, recordOf })).toHaveLength(3);
		expect(cutByRecord(pool, 3, { perRecord: 2, reserve: 99, recordOf }).map((hit) => hit.record)).toContain("b");
	});
});

describe("RecordCut", () => {
	it("asks the backend for three times the limit by default, never past the cap", () => {
		const cut = new RecordCut();
		expect(cut.fusedLimit(50, OVERFETCH_ABSOLUTE_CAP)).toBe(150);
		expect(cut.fusedLimit(120, OVERFETCH_ABSOLUTE_CAP)).toBe(300);
	});

	it("never fetches less than the caller's own limit, even past the cap", () => {
		expect(new RecordCut().fusedLimit(500, OVERFETCH_ABSOLUTE_CAP)).toBe(500);
	});

	it("allows 8 passages per record at a limit of 50, and at least the floor for small limits", () => {
		const cut = new RecordCut();
		expect(cut.perRecord(50)).toBe(8);
		expect(cut.perRecord(10)).toBe(4);
	});

	it("reserves half the limit for records not shown yet by default", () => {
		// 30 passages of one long record, then one keyword-only passage of each of 30 others.
		const pool = [...hits("long", 30), ...Array.from({ length: 30 }, (_, i) => ({ record: `r${i}`, n: 30 + i }))];
		const kept = new RecordCut().cut(pool, 20, recordOf);

		expect(kept).toHaveLength(20);
		// per-record = max(4, 20 / 6) = 4; head = 10 (long x4 + r0..r5); reserve = 10 (r6..r15).
		expect(kept.filter((hit) => hit.record === "long")).toHaveLength(4);
		expect(new Set(kept.map((hit) => hit.record)).size).toBe(17);
	});

	it("without the cut, a plain top-N is monopolised by the long record (control)", () => {
		const pool = [...hits("long", 30), ...Array.from({ length: 30 }, (_, i) => ({ record: `r${i}`, n: 30 + i }))];
		expect(new Set(pool.slice(0, 20).map((hit) => hit.record))).toEqual(new Set(["long"]));
	});
});

describe("recordCutFromOptions", () => {
	it("is off when no options are given or enabled is false", () => {
		expect(recordCutFromOptions(undefined)).toBeUndefined();
		expect(recordCutFromOptions({ enabled: false })).toBeUndefined();
	});

	it("uses the PipesHub defaults when enabled with no tuning", () => {
		const cut = recordCutFromOptions({ enabled: true });
		expect(cut).toMatchObject({ depth: 3, newRecordShare: 0.5, perRecordFloor: 4, perRecordDivisor: 6 });
	});

	it("applies tuning and rejects out-of-range values", () => {
		const cut = recordCutFromOptions({ enabled: true, depth: 5, newRecordShare: 0.25, perRecordFloor: 2 });
		expect(cut).toMatchObject({ depth: 5, newRecordShare: 0.25, perRecordFloor: 2 });
		expect(() => recordCutFromOptions({ enabled: true, depth: 11 })).toThrow(/depth/);
		expect(() => recordCutFromOptions({ enabled: true, depth: 0 })).toThrow(/depth/);
		expect(() => recordCutFromOptions({ enabled: true, newRecordShare: 1.5 })).toThrow(/newRecordShare/);
		expect(() => recordCutFromOptions({ enabled: true, perRecordDivisor: 0 })).toThrow(/perRecordDivisor/);
	});
});

describe("fairShares", () => {
	it("gives a document that needs less than an equal split exactly what it needs and splits the rest", () => {
		expect(fairShares([100, 5_000, 5_000], 3_100)).toEqual([100, 1_500, 1_500]);
	});

	it("splits equally when every document is larger than its split", () => {
		expect(fairShares([9_000, 9_000, 9_000], 3_000)).toEqual([1_000, 1_000, 1_000]);
	});

	it("never lets the first document take the whole budget (the empty-tail bug)", () => {
		const shares = fairShares([1_000_000, 4_000, 4_000], 12_000);
		expect(shares[1]).toBe(4_000);
		expect(shares[2]).toBe(4_000);
		expect(shares[0]).toBe(4_000);
	});

	it("keeps input order and never sums past the total", () => {
		const shares = fairShares([7, 3, 900, 0], 100);
		expect(shares).toEqual([7, 3, 90, 0]);
		expect(shares.reduce((a, b) => a + b, 0)).toBeLessThanOrEqual(100);
		expect(shares[3]).toBe(0);
	});

	it("returns zeros for an empty or non-positive budget", () => {
		expect(fairShares([5, 5], 0)).toEqual([0, 0]);
		expect(fairShares([5, 5], -3)).toEqual([0, 0]);
		expect(fairShares([], 10)).toEqual([]);
	});
});
