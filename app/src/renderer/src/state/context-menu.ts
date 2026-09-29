/**
 * Row context menu contents.
 *
 * The first two groups and the trash item are the reference's (handoff README
 * §2 "Context menu"). The middle group is the standard Finder file-operation
 * set the product requires on top of the design: copy / paste / cut /
 * duplicate / rename / copy path. "Show in Enclosing Folder" is a product
 * addition too, and only the virtual Recents view offers it — elsewhere the
 * row already lives in the folder it would open.
 */

export type FinderMenuAction =
	| "quickLook"
	| "open"
	| "showInEnclosingFolder"
	| "toggleIndex"
	| "retryIndex"
	| "copy"
	| "paste"
	| "cut"
	| "duplicate"
	| "rename"
	| "copyPath"
	| "trash";

export type MenuTone = "default" | "danger" | "warning";

export interface FinderMenuItem {
	readonly kind: "item";
	readonly action: FinderMenuAction;
	readonly label: string;
	readonly keycap: string | null;
	readonly tone: MenuTone;
	readonly disabled: boolean;
}

export interface FinderMenuSeparator {
	readonly kind: "separator";
}

export type FinderMenuEntry = FinderMenuItem | FinderMenuSeparator;

export interface ContextMenuInput {
	readonly target: { readonly name: string; readonly kind: "file" | "folder" };
	readonly selectionCount: number;
	readonly indexIncluded: boolean;
	readonly indexFailed?: boolean;
	/** Items held by the in-app file clipboard; 0 disables 붙여넣기. */
	readonly clipboardCount: number;
	/** True while the listing is the virtual Recents location. */
	readonly inRecents?: boolean;
}

function item(
	action: FinderMenuAction,
	label: string,
	options: { keycap?: string; tone?: MenuTone; disabled?: boolean } = {},
): FinderMenuItem {
	return {
		kind: "item",
		action,
		label,
		keycap: options.keycap ?? null,
		tone: options.tone ?? "default",
		disabled: options.disabled ?? false,
	};
}

const SEPARATOR: FinderMenuSeparator = { kind: "separator" };

export function buildContextMenu(input: ContextMenuInput): FinderMenuEntry[] {
	const isFolder = input.target.kind === "folder";
	const multi = input.selectionCount > 1;
	const entries: FinderMenuEntry[] = [
		isFolder ? item("open", "열기", { keycap: "↩" }) : item("quickLook", "Quick Look", { keycap: "space" }),
	];

	if (input.inRecents === true) {
		entries.push(SEPARATOR, item("showInEnclosingFolder", "Show in Enclosing Folder"));
	}

	if (!isFolder) {
		entries.push(SEPARATOR);
		entries.push(
			input.indexFailed === true
				? item("retryIndex", "인덱싱 다시 시도", { tone: "warning" })
				: item("toggleIndex", input.indexIncluded ? "인덱싱에서 제외" : "인덱싱에 포함"),
		);
	}

	entries.push(
		SEPARATOR,
		item("copy", "복사"),
		item("paste", "붙여넣기", { disabled: input.clipboardCount === 0 }),
		item("cut", "잘라내기"),
		item("duplicate", "복제"),
		item("rename", "이름 변경", { disabled: multi }),
		item("copyPath", "경로 복사"),
		SEPARATOR,
		item("trash", multi ? `${input.selectionCount}개 항목 휴지통으로 이동` : "휴지통으로 이동", {
			keycap: "⌫",
			tone: "danger",
		}),
	);

	return entries;
}

/* ------------------------------------------------------------- positioning */

export interface MenuPoint {
	readonly x: number;
	readonly y: number;
}

export interface MenuSize {
	readonly width: number;
	readonly height: number;
}

export interface MenuPlacement {
	readonly left: number;
	readonly top: number;
}

/** Keeps the fixed-position menu inside the window, like the reference's
 * "opens upward when the row is among the last 4" rule for the access menu. */
/** Mirrors --w-context-menu; the placement math needs the number. */
export const CONTEXT_MENU_WIDTH = 220;
const MENU_MARGIN = 8;
const MENU_ITEM_HEIGHT = 28;
/** 1px rule plus its 4px margins. */
const MENU_SEPARATOR_HEIGHT = 9;
/** --space-5 padding, top and bottom. */
const MENU_PADDING = 5;

export function estimateMenuHeight(entries: readonly FinderMenuEntry[]): number {
	let height = MENU_PADDING * 2;
	for (const entry of entries) {
		height += entry.kind === "separator" ? MENU_SEPARATOR_HEIGHT : MENU_ITEM_HEIGHT;
	}
	return height;
}

export function menuPlacement(cursor: MenuPoint, size: MenuSize, viewport: MenuSize): MenuPlacement {
	const left = Math.max(MENU_MARGIN, Math.min(cursor.x, viewport.width - size.width - MENU_MARGIN));
	const overflows = cursor.y + size.height > viewport.height - MENU_MARGIN;
	const top = overflows ? Math.max(MENU_MARGIN, cursor.y - size.height) : cursor.y;
	return { left, top };
}
