/**
 * Finder tab model: one path + its own back/forward history and selection.
 *
 * Reference: handoff README §2 "Tab strip" and the state sketch's `Tab`.
 */

import { EMPTY_SELECTION, type SelectionState } from "./selection";
import { basename } from "./paths";

export interface FinderTab {
	readonly id: number;
	readonly path: string;
	readonly back: readonly string[];
	readonly fwd: readonly string[];
	readonly selection: SelectionState;
	/** Reveal flash dot, held for --time-flash. */
	readonly flash: boolean;
}

export interface TabsState {
	readonly tabs: readonly FinderTab[];
	readonly activeId: number;
	readonly nextId: number;
}

function blankTab(id: number, path: string): FinderTab {
	return { id, path, back: [], fwd: [], selection: EMPTY_SELECTION, flash: false };
}

export function createTabsState(path: string): TabsState {
	return { tabs: [blankTab(1, path)], activeId: 1, nextId: 2 };
}

export function activeTab(state: TabsState): FinderTab {
	const tab = state.tabs.find((t) => t.id === state.activeId) ?? state.tabs[0];
	if (tab === undefined) {
		throw new Error("TabsState must always hold at least one tab");
	}
	return tab;
}

export function patchActiveTab(state: TabsState, patch: (tab: FinderTab) => FinderTab): TabsState {
	return { ...state, tabs: state.tabs.map((tab) => (tab.id === state.activeId ? patch(tab) : tab)) };
}

export function openTab(state: TabsState): TabsState {
	return openTabAt(state, activeTab(state).path);
}

/** A new tab on an explicit path, e.g. the folder that encloses a Recents row. */
export function openTabAt(state: TabsState, path: string): TabsState {
	const id = state.nextId;
	return {
		tabs: [...state.tabs, blankTab(id, path)],
		activeId: id,
		nextId: id + 1,
	};
}

export function canCloseTab(state: TabsState): boolean {
	return state.tabs.length > 1;
}

export function closeTab(state: TabsState, id: number): TabsState {
	if (!canCloseTab(state)) {
		return state;
	}
	const index = state.tabs.findIndex((tab) => tab.id === id);
	if (index < 0) {
		return state;
	}
	const tabs = state.tabs.filter((tab) => tab.id !== id);
	const fallback = tabs[Math.max(0, index - 1)];
	return {
		...state,
		tabs,
		activeId: state.activeId === id && fallback !== undefined ? fallback.id : state.activeId,
	};
}

export function selectTab(state: TabsState, id: number): TabsState {
	return state.tabs.some((tab) => tab.id === id) ? { ...state, activeId: id } : state;
}

export function navigateTab(state: TabsState, path: string): TabsState {
	return patchActiveTab(state, (tab) =>
		tab.path === path
			? { ...tab, selection: EMPTY_SELECTION }
			: { ...tab, path, back: [...tab.back, tab.path], fwd: [], selection: EMPTY_SELECTION },
	);
}

export function goBack(state: TabsState): TabsState {
	const tab = activeTab(state);
	const previous = tab.back[tab.back.length - 1];
	if (previous === undefined) {
		return state;
	}
	return patchActiveTab(state, (t) => ({
		...t,
		path: previous,
		back: t.back.slice(0, -1),
		fwd: [t.path, ...t.fwd],
		selection: EMPTY_SELECTION,
	}));
}

export function goForward(state: TabsState): TabsState {
	const tab = activeTab(state);
	const next = tab.fwd[0];
	if (next === undefined) {
		return state;
	}
	return patchActiveTab(state, (t) => ({
		...t,
		path: next,
		back: [...t.back, t.path],
		fwd: t.fwd.slice(1),
		selection: EMPTY_SELECTION,
	}));
}

/** The Contacts view keeps its own title while searching. */
export const CONTACTS_PATH = "Contacts";

export function tabTitle(tab: { readonly path: string }, view: { readonly isActive: boolean; readonly query: string }): string {
	const query = view.query.trim();
	if (view.isActive && query !== "" && tab.path !== CONTACTS_PATH) {
		return `Search "${query}"`;
	}
	return basename(tab.path) || tab.path;
}
