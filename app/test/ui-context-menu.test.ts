import { describe, expect, it } from "vitest";
import { buildContextMenu, estimateMenuHeight, menuPlacement } from "../src/renderer/src/state/context-menu";
import type { FinderMenuItem } from "../src/renderer/src/state/context-menu";

const file = {
	target: { name: "a.pdf", kind: "file" as const },
	selectionCount: 1,
	indexIncluded: true,
	clipboardCount: 0,
};

function items(entries: ReturnType<typeof buildContextMenu>): FinderMenuItem[] {
	return entries.filter((e): e is FinderMenuItem => e.kind === "item");
}

describe("buildContextMenu", () => {
	it("opens with Quick Look and its space keycap for a file", () => {
		const first = items(buildContextMenu(file))[0];
		expect(first?.label).toBe("Quick Look");
		expect(first?.keycap).toBe("space");
		expect(first?.action).toBe("quickLook");
	});

	it("opens with 열기 and the return keycap for a folder", () => {
		const menu = buildContextMenu({ ...file, target: { name: "Finance", kind: "folder" } });
		const first = items(menu)[0];
		expect(first?.label).toBe("열기");
		expect(first?.keycap).toBe("↩");
		expect(first?.action).toBe("open");
	});

	it("offers the index toggle for files only", () => {
		expect(items(buildContextMenu(file)).map((i) => i.label)).toContain("인덱싱에서 제외");
		expect(items(buildContextMenu({ ...file, indexIncluded: false })).map((i) => i.label)).toContain("인덱싱에 포함");
		const folder = items(buildContextMenu({ ...file, target: { name: "Finance", kind: "folder" } }));
		expect(folder.map((i) => i.action)).not.toContain("toggleIndex");
	});

	it("lists the standard file operations in Finder order", () => {
		const ops = items(buildContextMenu(file)).map((i) => i.label);
		expect(ops.slice(2)).toEqual([
			"복사",
			"붙여넣기",
			"잘라내기",
			"복제",
			"이름 변경",
			"경로 복사",
			"휴지통으로 이동",
		]);
	});

	it("enables 붙여넣기 only when the clipboard has items", () => {
		const disabled = items(buildContextMenu(file)).find((i) => i.action === "paste");
		expect(disabled?.disabled).toBe(true);
		const enabled = items(buildContextMenu({ ...file, clipboardCount: 2 })).find((i) => i.action === "paste");
		expect(enabled?.disabled).toBe(false);
	});

	it("disables 이름 변경 for a multi-selection", () => {
		const single = items(buildContextMenu(file)).find((i) => i.action === "rename");
		expect(single?.disabled).toBe(false);
		const many = items(buildContextMenu({ ...file, selectionCount: 3 })).find((i) => i.action === "rename");
		expect(many?.disabled).toBe(true);
	});

	it("marks the trash item destructive and counts a multi-selection", () => {
		const one = items(buildContextMenu(file)).at(-1);
		expect(one?.label).toBe("휴지통으로 이동");
		expect(one?.keycap).toBe("⌫");
		expect(one?.tone).toBe("danger");
		const many = items(buildContextMenu({ ...file, selectionCount: 4 })).at(-1);
		expect(many?.label).toBe("4개 항목 휴지통으로 이동");
	});

	it("separates the groups exactly like the reference", () => {
		expect(buildContextMenu(file).map((e) => e.kind)).toEqual([
			"item",
			"separator",
			"item",
			"separator",
			"item",
			"item",
			"item",
			"item",
			"item",
			"item",
			"separator",
			"item",
		]);
	});
});

describe("menuPlacement", () => {
	const size = { width: 220, height: 300 };
	const viewport = { width: 1440, height: 900 };

	it("anchors at the cursor when the menu fits", () => {
		expect(menuPlacement({ x: 400, y: 200 }, size, viewport)).toEqual({ left: 400, top: 200 });
	});

	it("flips above the cursor when the bottom edge would clip", () => {
		expect(menuPlacement({ x: 400, y: 800 }, size, viewport)).toEqual({ left: 400, top: 500 });
	});

	it("pulls back from the right edge", () => {
		expect(menuPlacement({ x: 1400, y: 200 }, size, viewport)).toEqual({ left: 1212, top: 200 });
	});

	it("clamps the left edge to the window margin", () => {
		expect(menuPlacement({ x: 2, y: 10 }, size, viewport)).toEqual({ left: 8, top: 10 });
	});

	it("clamps a menu taller than the viewport to the top margin", () => {
		expect(menuPlacement({ x: 400, y: 10 }, { width: 220, height: 1000 }, viewport)).toEqual({ left: 400, top: 8 });
	});
});

describe("estimateMenuHeight", () => {
	it("measures items, separators, and the menu padding", () => {
		expect(estimateMenuHeight(buildContextMenu(file))).toBe(9 * 28 + 3 * 9 + 2 * 5);
	});
});
