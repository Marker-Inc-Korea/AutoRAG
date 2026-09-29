import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { KeyboardEvent as ReactKeyboardEvent, MouseEvent as ReactMouseEvent, RefObject } from "react";
import type { FinderEntry } from "../data/entries";
import type { NavItem } from "../data/places";
import { type FinderLocation, type FinderSource, pickInitialPath } from "../data/source";
import { validateName } from "../state/collision";
import {
	buildContextMenu,
	type FinderMenuAction,
	type FinderMenuEntry,
} from "../state/context-menu";
import { emptyStateText, indexToast, searchSummaryText, statusBarText, trashToast } from "../state/format";
import { type FocusZone, resolveKeyAction } from "../state/keymap";
import { activeNavPath, basename, breadcrumbTrail, type Crumb } from "../state/paths";
import {
	applyRowClick,
	EMPTY_SELECTION,
	moveSelection,
	type SelectionState,
	scrollTopForIndex,
} from "../state/selection";
import { type SortKey, type SortState, cycleSort, sortEntries } from "../state/sort";
import { visibleEntries } from "../state/visibility";
import {
	activeTab,
	canCloseTab,
	closeTab,
	CONTACTS_PATH,
	createTabsState,
	goBack,
	goForward,
	navigateTab,
	openTab,
	patchActiveTab,
	selectTab,
	tabTitle,
	type TabsState,
} from "../state/tabs";
import type { TabView } from "../components/TabStrip";

/** --time-toast */
const TOAST_MS = 2400;
/** --time-flash */
const FLASH_MS = 1800;

interface ContextMenuState {
	readonly cursor: { readonly x: number; readonly y: number };
	readonly entry: FinderEntry;
}

interface RenameState {
	readonly path: string;
	readonly draft: string;
}

export interface FinderController {
	readonly tabs: readonly TabView[];
	readonly canCloseTabs: boolean;
	readonly path: string;
	readonly crumbs: readonly Crumb[];
	readonly canGoBack: boolean;
	readonly canGoForward: boolean;
	readonly query: string;
	readonly searchPlaceholder: string;
	readonly searchFocused: boolean;
	readonly searching: boolean;
	readonly searchSummary: string;
	readonly emptyText: string;
	readonly rows: readonly FinderEntry[];
	readonly sort: SortState | null;
	readonly zone: FocusZone;
	readonly selection: SelectionState;
	readonly statusText: string;
	readonly toast: string | null;
	readonly flashPath: string | null;
	readonly renamePath: string | null;
	readonly renameDraft: string;
	readonly indexOverrides: Readonly<Record<string, boolean>>;
	readonly activeNavLabel: string | null;
	readonly pendingRequests: number;
	readonly contextMenu: { readonly cursor: { readonly x: number; readonly y: number } } | null;
	readonly contextMenuEntries: readonly FinderMenuEntry[];
	readonly listRef: RefObject<HTMLDivElement | null>;
	readonly searchRef: RefObject<HTMLInputElement | null>;
	navigate(path: string): void;
	navigateNav(item: NavItem): void;
	selectTabId(id: number): void;
	closeTabId(id: number): void;
	newTab(): void;
	back(): void;
	forward(): void;
	setQuery(value: string): void;
	setSearchFocused(value: boolean): void;
	onSearchKeyDown(event: ReactKeyboardEvent<HTMLInputElement>): void;
	sortBy(key: SortKey): void;
	focusFinder(): void;
	clickRow(entry: FinderEntry, event: ReactMouseEvent<HTMLDivElement>): void;
	openEntry(entry: FinderEntry): void;
	openContextMenu(entry: FinderEntry, event: ReactMouseEvent<HTMLDivElement>): void;
	closeContextMenu(): void;
	runMenuAction(action: FinderMenuAction): void;
	toggleIndex(entry: FinderEntry): void;
	setRenameDraft(value: string): void;
	commitRename(): void;
	cancelRename(): void;
}

export function useFinderController(
	source: FinderSource,
	options: { readonly showHiddenFiles?: boolean } = {},
): FinderController {
	const showHiddenFiles = options.showHiddenFiles ?? false;
	const [tabsState, setTabsState] = useState<TabsState>(() => createTabsState(source.initialPath));
	const [query, setQueryValue] = useState("");
	const [sort, setSort] = useState<SortState | null>(null);
	const [zone, setZone] = useState<FocusZone>("finder");
	const [listing, setListing] = useState<readonly FinderEntry[]>([]);
	const [hits, setHits] = useState<readonly FinderEntry[]>([]);
	const [locations, setLocations] = useState<readonly FinderLocation[]>([]);
	const [indexOverrides, setIndexOverrides] = useState<Readonly<Record<string, boolean>>>({});
	const [toast, setToast] = useState<string | null>(null);
	const [contextMenu, setContextMenu] = useState<ContextMenuState | null>(null);
	const [rename, setRename] = useState<RenameState | null>(null);
	const [clipboardCount, setClipboardCount] = useState(0);
	const [flashPath, setFlashPath] = useState<string | null>(null);
	const [searchFocused, setSearchFocused] = useState(false);
	const [revision, setRevision] = useState(0);

	const listRef = useRef<HTMLDivElement | null>(null);
	const searchRef = useRef<HTMLInputElement | null>(null);
	const scrollTarget = useRef<string | null>(null);

	const tab = activeTab(tabsState);
	const path = tab.path;
	const selection = tab.selection;
	const searching = query.trim() !== "";

	const refresh = useCallback(() => setRevision((value) => value + 1), []);

	const showToast = useCallback((message: string) => setToast(message), []);

	/** Failures surface verbatim — the operator sees the real message. */
	const reportError = useCallback(
		(error: unknown) => setToast(error instanceof Error ? error.message : String(error)),
		[],
	);

	useEffect(() => {
		let live = true;
		source
			.locations()
			.then((next) => {
				if (live) {
					setLocations(next);
				}
			})
			.catch(reportError);
		return () => {
			live = false;
		};
	}, [source, reportError]);

	/** The bridge source starts with no path; the first listing needs one. */
	useEffect(() => {
		if (path !== "" || locations.length === 0) {
			return;
		}
		const start = pickInitialPath(locations);
		if (start !== null) {
			setTabsState((state) => navigateTab(state, start));
		}
	}, [path, locations]);

	useEffect(() => {
		if (path === "") {
			return;
		}
		let live = true;
		source
			.list(path)
			.then((entries) => {
				if (live) {
					setListing(entries);
				}
			})
			.catch((error: unknown) => {
				if (live) {
					setListing([]);
					reportError(error);
				}
			});
		return () => {
			live = false;
		};
	}, [source, path, revision, reportError]);

	useEffect(() => {
		if (!searching) {
			setHits([]);
			return;
		}
		let live = true;
		source
			.search(query)
			.then((entries) => {
				if (live) {
					setHits(entries);
				}
			})
			.catch((error: unknown) => {
				if (live) {
					setHits([]);
					reportError(error);
				}
			});
		return () => {
			live = false;
		};
	}, [source, query, searching, revision, reportError]);

	useEffect(() => {
		if (toast === null) {
			return;
		}
		const timer = setTimeout(() => setToast(null), TOAST_MS);
		return () => clearTimeout(timer);
	}, [toast]);

	useEffect(() => {
		if (flashPath === null) {
			return;
		}
		const timer = setTimeout(() => setFlashPath(null), FLASH_MS);
		return () => clearTimeout(timer);
	}, [flashPath]);

	const rows = useMemo(
		() => sortEntries(visibleEntries(searching ? hits : listing, showHiddenFiles), sort),
		[searching, hits, listing, sort, showHiddenFiles],
	);
	const orderedKeys = useMemo(() => rows.map((entry) => entry.path), [rows]);

	/** Keyboard movement keeps the focused row in view (row height 32, lead 80). */
	useEffect(() => {
		const target = scrollTarget.current;
		if (target === null) {
			return;
		}
		scrollTarget.current = null;
		const index = orderedKeys.indexOf(target);
		if (index >= 0 && listRef.current !== null) {
			listRef.current.scrollTop = scrollTopForIndex(index);
		}
	}, [orderedKeys]);

	const patchSelection = useCallback((next: SelectionState) => {
		setTabsState((state) => patchActiveTab(state, (current) => ({ ...current, selection: next })));
	}, []);

	const selectedEntries = useMemo(
		() => rows.filter((entry) => selection.keys.includes(entry.path)),
		[rows, selection.keys],
	);
	const focusedEntry = useMemo(
		() => rows.find((entry) => entry.path === selection.focus) ?? null,
		[rows, selection.focus],
	);

	const navigate = useCallback((next: string) => {
		setTabsState((state) => navigateTab(state, next));
		setQueryValue("");
	}, []);

	const navigateNav = useCallback(
		(item: NavItem) => {
			const match = locations.find((location) => location.name === item.label);
			navigate(match?.path ?? item.label);
		},
		[locations, navigate],
	);

	const revealEntry = useCallback((entry: FinderEntry) => {
		setTabsState((state) =>
			patchActiveTab(navigateTab(state, entry.location), (current) => ({
				...current,
				selection: { keys: [entry.path], anchor: entry.path, focus: entry.path },
			})),
		);
		setQueryValue("");
		setFlashPath(entry.path);
		scrollTarget.current = entry.path;
	}, []);

	const openEntry = useCallback(
		(entry: FinderEntry) => {
			if (entry.kind === "folder") {
				navigate(entry.path);
				return;
			}
			if (searching || entry.location !== path) {
				revealEntry(entry);
				return;
			}
			source.quickLook(entry.path).catch(reportError);
		},
		[navigate, path, revealEntry, searching, source, reportError],
	);

	const trashPaths = useCallback(
		(paths: readonly string[]) => {
			if (paths.length === 0) {
				return;
			}
			source
				.trash(paths)
				.then(() => {
					patchSelection(EMPTY_SELECTION);
					showToast(trashToast(paths));
					refresh();
				})
				.catch(reportError);
		},
		[source, patchSelection, showToast, refresh, reportError],
	);

	const toggleIndex = useCallback(
		(entry: FinderEntry) => {
			const included = indexOverrides[entry.path] ?? true;
			setIndexOverrides((current) => ({ ...current, [entry.path]: !included }));
			showToast(indexToast(entry.name, !included));
		},
		[indexOverrides, showToast],
	);

	const quickLookSelection = useCallback(() => {
		const entry = focusedEntry;
		if (entry === null || entry.kind === "folder") {
			return;
		}
		source.quickLook(entry.path).catch(reportError);
	}, [focusedEntry, source, reportError]);

	const menuTargets = useCallback(
		(entry: FinderEntry): readonly string[] =>
			selection.keys.includes(entry.path) && selection.keys.length > 1 ? selection.keys : [entry.path],
		[selection.keys],
	);

	const runMenuAction = useCallback(
		(action: FinderMenuAction) => {
			const menu = contextMenu;
			if (menu === null) {
				return;
			}
			const entry = menu.entry;
			const targets = menuTargets(entry);
			setContextMenu(null);
			switch (action) {
				case "quickLook":
					source.quickLook(entry.path).catch(reportError);
					return;
				case "open":
					navigate(entry.path);
					return;
				case "toggleIndex":
					toggleIndex(entry);
					return;
				case "retryIndex":
					return;
				case "copy":
					source
						.clipboardSet({ op: "copy", paths: targets })
						.then(() => setClipboardCount(targets.length))
						.catch(reportError);
					return;
				case "cut":
					source
						.clipboardSet({ op: "cut", paths: targets })
						.then(() => setClipboardCount(targets.length))
						.catch(reportError);
					return;
				case "paste":
					source
						.clipboardGet()
						.then(async (clipboard) => {
							if (clipboard === null || clipboard.paths.length === 0) {
								return;
							}
							if (clipboard.op === "copy") {
								await source.copyInto(clipboard.paths, path);
							} else {
								await source.moveInto(clipboard.paths, path);
								setClipboardCount(0);
							}
							refresh();
						})
						.catch(reportError);
					return;
				case "duplicate":
					source
						.duplicate(targets)
						.then(refresh)
						.catch(reportError);
					return;
				case "rename":
					setRename({ path: entry.path, draft: entry.name });
					return;
				case "copyPath":
					source.copyPaths(targets).catch(reportError);
					return;
				case "trash":
					trashPaths(targets);
					return;
			}
		},
		[contextMenu, menuTargets, navigate, path, refresh, reportError, source, toggleIndex, trashPaths],
	);

	const commitRename = useCallback(() => {
		const editing = rename;
		if (editing === null) {
			return;
		}
		const current = basename(editing.path);
		const siblings = rows.map((entry) => entry.name);
		const check = validateName(editing.draft, siblings, current);
		setRename(null);
		if (!check.ok) {
			showToast(check.message);
			return;
		}
		const next = editing.draft.trim();
		if (next === current) {
			return;
		}
		source
			.rename(editing.path, next)
			.then(refresh)
			.catch(reportError);
	}, [rename, rows, showToast, source, refresh, reportError]);

	const openContextMenu = useCallback(
		(entry: FinderEntry, event: ReactMouseEvent<HTMLDivElement>) => {
			event.preventDefault();
			setZone("finder");
			if (!selection.keys.includes(entry.path)) {
				patchSelection({ keys: [entry.path], anchor: entry.path, focus: entry.path });
			}
			source
				.clipboardGet()
				.then((clipboard) => setClipboardCount(clipboard?.paths.length ?? 0))
				.catch(reportError);
			setContextMenu({ cursor: { x: event.clientX, y: event.clientY }, entry });
		},
		[selection.keys, patchSelection, source, reportError],
	);

	const dismiss = useCallback(() => {
		if (contextMenu !== null) {
			setContextMenu(null);
			return;
		}
		if (rename !== null) {
			setRename(null);
			return;
		}
		if (query !== "") {
			setQueryValue("");
		}
	}, [contextMenu, rename, query]);

	useEffect(() => {
		const onKeyDown = (event: globalThis.KeyboardEvent): void => {
			const target = event.target;
			const typing =
				target instanceof HTMLElement &&
				(target.tagName === "INPUT" || target.tagName === "TEXTAREA" || target.isContentEditable);
			const action = resolveKeyAction(
				{
					key: event.key,
					meta: event.metaKey || event.ctrlKey,
					shift: event.shiftKey,
					alt: event.altKey,
				},
				{ zone, typing },
			);
			if (action === null) {
				return;
			}
			switch (action.type) {
				case "dismiss":
					dismiss();
					return;
				case "focusSearch":
					event.preventDefault();
					searchRef.current?.focus();
					return;
				case "newTab":
					event.preventDefault();
					setTabsState((state) => openTab(state));
					return;
				case "closeTab":
					event.preventDefault();
					setTabsState((state) => closeTab(state, state.activeId));
					return;
				case "quickLook":
					event.preventDefault();
					quickLookSelection();
					return;
				case "moveSelection": {
					event.preventDefault();
					const next = moveSelection(orderedKeys, selection, action.delta);
					scrollTarget.current = next.focus;
					patchSelection(next);
					return;
				}
				case "moveEvidence":
					return;
				case "open":
					event.preventDefault();
					if (focusedEntry !== null) {
						openEntry(focusedEntry);
					}
					return;
				case "trash":
					event.preventDefault();
					trashPaths(selection.keys);
					return;
			}
		};
		window.addEventListener("keydown", onKeyDown);
		return () => window.removeEventListener("keydown", onKeyDown);
	}, [
		zone,
		dismiss,
		orderedKeys,
		selection,
		patchSelection,
		quickLookSelection,
		focusedEntry,
		openEntry,
		trashPaths,
	]);

	const tabViews = useMemo<readonly TabView[]>(
		() =>
			tabsState.tabs.map((current) => {
				const active = current.id === tabsState.activeId;
				return {
					id: current.id,
					title: tabTitle(current, { isActive: active, query }),
					active,
					flash: current.flash,
				};
			}),
		[tabsState, query],
	);

	const contextMenuEntries = useMemo<readonly FinderMenuEntry[]>(() => {
		const entry = contextMenu?.entry;
		if (entry === undefined) {
			return [];
		}
		return buildContextMenu({
			target: { name: entry.name, kind: entry.kind },
			selectionCount: menuTargets(entry).length,
			indexIncluded: indexOverrides[entry.path] ?? true,
			clipboardCount,
		});
	}, [contextMenu, menuTargets, indexOverrides, clipboardCount]);

	const navTargets = useMemo(
		() => (locations.length > 0 ? locations : [{ name: basename(path), path }]),
		[locations, path],
	);
	const activeNav = activeNavPath(path, navTargets);
	const activeNavLabel =
		navTargets.find((target) => target.path === activeNav)?.name ?? (path === "" ? null : basename(path));

	return {
		tabs: tabViews,
		canCloseTabs: canCloseTab(tabsState),
		path,
		crumbs: breadcrumbTrail(path),
		canGoBack: tab.back.length > 0,
		canGoForward: tab.fwd.length > 0,
		query,
		searchPlaceholder: path === CONTACTS_PATH ? "Search contacts" : "Instant search",
		searchFocused,
		searching,
		searchSummary: searchSummaryText(rows.length),
		emptyText: emptyStateText(query),
		rows,
		sort,
		zone,
		selection,
		statusText: statusBarText(rows.length, selectedEntries.length),
		toast,
		flashPath,
		renamePath: rename?.path ?? null,
		renameDraft: rename?.draft ?? "",
		indexOverrides,
		activeNavLabel,
		pendingRequests: source.pendingRequests,
		contextMenu: contextMenu === null ? null : { cursor: contextMenu.cursor },
		contextMenuEntries,
		listRef,
		searchRef,
		navigate,
		navigateNav,
		selectTabId: (id) => setTabsState((state) => selectTab(state, id)),
		closeTabId: (id) => setTabsState((state) => closeTab(state, id)),
		newTab: () => setTabsState((state) => openTab(state)),
		back: () => setTabsState((state) => goBack(state)),
		forward: () => setTabsState((state) => goForward(state)),
		setQuery: setQueryValue,
		setSearchFocused,
		onSearchKeyDown: (event) => {
			if (event.key === "Enter") {
				const first = rows[0];
				if (first !== undefined) {
					openEntry(first);
				}
			}
			if (event.key === "Escape") {
				setQueryValue("");
				event.currentTarget.blur();
			}
		},
		sortBy: (key) => setSort((current) => cycleSort(current, key)),
		focusFinder: () => setZone("finder"),
		clickRow: (entry, event) => {
			setZone("finder");
			patchSelection(
				applyRowClick(selection, entry.path, orderedKeys, {
					meta: event.metaKey || event.ctrlKey,
					shift: event.shiftKey,
				}),
			);
		},
		openEntry,
		openContextMenu,
		closeContextMenu: () => setContextMenu(null),
		runMenuAction,
		toggleIndex,
		setRenameDraft: (value) => setRename((current) => (current === null ? null : { ...current, draft: value })),
		commitRename,
		cancelRename: () => setRename(null),
	};
}
