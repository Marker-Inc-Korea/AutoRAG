import { describe, expect, it } from "vitest";
import {
	EMPTY_SELECTION,
	ROW_HEIGHT,
	applyRowClick,
	moveSelection,
	scrollTopForIndex,
	selectionCount,
} from "../src/renderer/src/state/selection";

const order = ["a", "b", "c", "d", "e"];
const plain = { meta: false, shift: false };

describe("applyRowClick", () => {
	it("selects a single row", () => {
		const next = applyRowClick(EMPTY_SELECTION, "c", order, plain);
		expect(next).toEqual({ keys: ["c"], anchor: "c", focus: "c" });
	});

	it("replaces a multi-selection on a plain click", () => {
		const multi = { keys: ["a", "b"], anchor: "a", focus: "b" };
		expect(applyRowClick(multi, "d", order, plain).keys).toEqual(["d"]);
	});

	it("adds and removes with the command key", () => {
		const one = applyRowClick(EMPTY_SELECTION, "a", order, plain);
		const two = applyRowClick(one, "c", order, { meta: true, shift: false });
		expect(two.keys).toEqual(["a", "c"]);
		expect(two.focus).toBe("c");
		const back = applyRowClick(two, "a", order, { meta: true, shift: false });
		expect(back.keys).toEqual(["c"]);
		expect(back.focus).toBe("c");
	});

	it("selects a range with shift, in both directions", () => {
		const anchored = applyRowClick(EMPTY_SELECTION, "b", order, plain);
		expect(applyRowClick(anchored, "d", order, { meta: false, shift: true }).keys).toEqual(["b", "c", "d"]);
		const fromEnd = applyRowClick(EMPTY_SELECTION, "d", order, plain);
		expect(applyRowClick(fromEnd, "b", order, { meta: false, shift: true }).keys).toEqual(["b", "c", "d"]);
	});

	it("keeps the anchor across repeated shift clicks", () => {
		const anchored = applyRowClick(EMPTY_SELECTION, "b", order, plain);
		const wide = applyRowClick(anchored, "e", order, { meta: false, shift: true });
		expect(applyRowClick(wide, "c", order, { meta: false, shift: true }).keys).toEqual(["b", "c"]);
	});

	it("falls back to a single row when shift has no anchor", () => {
		expect(applyRowClick(EMPTY_SELECTION, "c", order, { meta: false, shift: true }).keys).toEqual(["c"]);
	});
});

describe("moveSelection", () => {
	it("moves down and up by one row", () => {
		const at = applyRowClick(EMPTY_SELECTION, "b", order, plain);
		expect(moveSelection(order, at, 1).focus).toBe("c");
		expect(moveSelection(order, at, -1).focus).toBe("a");
	});

	it("clamps at both ends", () => {
		const first = applyRowClick(EMPTY_SELECTION, "a", order, plain);
		expect(moveSelection(order, first, -1).focus).toBe("a");
		const last = applyRowClick(EMPTY_SELECTION, "e", order, plain);
		expect(moveSelection(order, last, 1).focus).toBe("e");
	});

	it("starts at the first row when nothing is selected", () => {
		expect(moveSelection(order, EMPTY_SELECTION, 1).focus).toBe("a");
	});

	it("collapses a multi-selection to one row", () => {
		const multi = { keys: ["a", "b", "c"], anchor: "a", focus: "c" };
		expect(moveSelection(order, multi, 1).keys).toEqual(["d"]);
	});

	it("keeps the selection on an empty list", () => {
		expect(moveSelection([], EMPTY_SELECTION, 1)).toEqual(EMPTY_SELECTION);
	});
});

describe("scrollTopForIndex", () => {
	it("keeps the target row 80px below the top edge", () => {
		expect(ROW_HEIGHT).toBe(32);
		expect(scrollTopForIndex(10)).toBe(240);
	});

	it("never scrolls above the top", () => {
		expect(scrollTopForIndex(0)).toBe(0);
		expect(scrollTopForIndex(2)).toBe(0);
	});
});

describe("selectionCount", () => {
	it("counts the selected keys", () => {
		expect(selectionCount(EMPTY_SELECTION)).toBe(0);
		expect(selectionCount({ keys: ["a", "b"], anchor: "a", focus: "b" })).toBe(2);
	});
});
