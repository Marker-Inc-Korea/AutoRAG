/**
 * Row selection for the Finder list: single, command-toggle, shift-range, and
 * keyboard movement.
 *
 * Reference: handoff README §2 "Selection". Keys are full item paths so a
 * selection survives a search (where rows come from many locations).
 */

/** File row height; the list is a fixed-height grid. */
export const ROW_HEIGHT = 32;
/** Pixels of context kept above the focused row when scrolling it into view. */
export const SCROLL_LEAD = 80;

export interface SelectionState {
	readonly keys: readonly string[];
	/** Range anchor for shift-click. */
	readonly anchor: string | null;
	/** The row the keyboard acts on. */
	readonly focus: string | null;
}

export const EMPTY_SELECTION: SelectionState = { keys: [], anchor: null, focus: null };

export interface ClickModifiers {
	readonly meta: boolean;
	readonly shift: boolean;
}

function single(key: string): SelectionState {
	return { keys: [key], anchor: key, focus: key };
}

export function applyRowClick(
	state: SelectionState,
	key: string,
	orderedKeys: readonly string[],
	modifiers: ClickModifiers,
): SelectionState {
	if (modifiers.meta) {
		const keys = state.keys.includes(key) ? state.keys.filter((k) => k !== key) : [...state.keys, key];
		return { keys, anchor: key, focus: keys[keys.length - 1] ?? null };
	}
	if (modifiers.shift) {
		const anchor = state.anchor ?? state.focus ?? key;
		const from = orderedKeys.indexOf(anchor);
		const to = orderedKeys.indexOf(key);
		if (from < 0 || to < 0) {
			return single(key);
		}
		return {
			keys: orderedKeys.slice(Math.min(from, to), Math.max(from, to) + 1),
			anchor,
			focus: key,
		};
	}
	return single(key);
}

export function moveSelection(
	orderedKeys: readonly string[],
	state: SelectionState,
	delta: 1 | -1,
): SelectionState {
	if (orderedKeys.length === 0) {
		return state;
	}
	const current = state.focus === null ? -1 : orderedKeys.indexOf(state.focus);
	const target = current < 0 ? 0 : current + delta;
	const next = orderedKeys[Math.min(orderedKeys.length - 1, Math.max(0, target))];
	return next === undefined ? state : single(next);
}

export function scrollTopForIndex(index: number): number {
	return Math.max(0, index * ROW_HEIGHT - SCROLL_LEAD);
}

export function isSelected(state: SelectionState, key: string): boolean {
	return state.keys.includes(key);
}

export function selectionCount(state: SelectionState): number {
	return state.keys.length;
}
