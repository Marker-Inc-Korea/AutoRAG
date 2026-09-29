/**
 * The Finder's data port.
 *
 * One narrow interface with two adapters: the real Electron fs bridge
 * (`window.autorag.fs`) and an in-memory fixture source used whenever that
 * bridge is absent — vitest, a plain browser, or a renderer opened before the
 * preload lands. Components never touch `window` directly.
 */

import type { FsBridge, FsClipboard } from "../../../shared/fs-contract";
import { copyName } from "../state/collision";
import { basename, dirname, joinPath } from "../state/paths";
import type { VersionFamiliesResult, VersionFamilyData, VersionRelation } from "../state/version-family";
import { type FinderEntry, entryFromFixture, entryFromFs, entryFromSearchHit } from "./entries";
import { FIXTURE_FAMILIES, FIXTURE_INITIAL_PATH, FIXTURE_PENDING_REQUESTS, FIXTURE_ROOTS, FIXTURE_TREE } from "./fixtures";
import type { FixtureItem } from "./entries";

export interface FinderLocation {
	readonly name: string;
	readonly path: string;
	readonly section: "favorites" | "cloud";
	readonly available: boolean;
}

export interface SearchHits {
	readonly entries: readonly FinderEntry[];
}

export interface FinderSource {
	/** True when the native OS surfaces (Quick Look, Reveal) are real. */
	readonly native: boolean;
	readonly initialPath: string;
	/** Pending agent requests behind the sidebar bell; 0 until that lane lands. */
	readonly pendingRequests: number;
	list(path: string): Promise<readonly FinderEntry[]>;
	search(query: string): Promise<readonly FinderEntry[]>;
	locations(): Promise<readonly FinderLocation[]>;
	quickLook(path: string): Promise<void>;
	/** Open the file with the OS default application (double-click behavior). */
	open(path: string): Promise<void>;
	reveal(path: string): Promise<void>;
	trash(paths: readonly string[]): Promise<void>;
	rename(path: string, name: string): Promise<void>;
	duplicate(paths: readonly string[]): Promise<void>;
	copyInto(paths: readonly string[], destDir: string): Promise<void>;
	moveInto(paths: readonly string[], destDir: string): Promise<void>;
	copyPaths(paths: readonly string[]): Promise<void>;
	clipboardSet(clipboard: FsClipboard): Promise<void>;
	clipboardGet(): Promise<FsClipboard | null>;
	/** Version families from the duplicate detector, with an explicit error when dupey is missing. */
	versionFamilies(): Promise<VersionFamiliesResult>;
}

/* -------------------------------------------------------------- real bridge */

export function createBridgeSource(fs: FsBridge, initialPath: string): FinderSource {
	return {
		native: true,
		initialPath,
		pendingRequests: 0,
		async list(path) {
			const listing = await fs.listDir(path);
			return listing.entries.map((entry) => entryFromFs(entry, path));
		},
		async search(query) {
			const hits = await fs.search(query);
			return hits.map(entryFromSearchHit);
		},
		async locations() {
			const locations = await fs.locations();
			return locations.map((location) => ({
				name: location.name,
				path: location.path,
				section: location.section,
				available: location.available,
			}));
		},
		quickLook: (path) => fs.quickLook(path),
		open: (path) => fs.open(path),
		reveal: (path) => fs.reveal(path),
		trash: async (paths) => {
			await fs.trash(paths);
		},
		rename: async (path, name) => {
			await fs.rename(path, name);
		},
		duplicate: async (paths) => {
			await fs.duplicate(paths);
		},
		copyInto: async (paths, destDir) => {
			await fs.copy(paths, destDir);
		},
		moveInto: async (paths, destDir) => {
			await fs.move(paths, destDir);
		},
		copyPaths: (paths) => fs.copyPathsToClipboard(paths),
		clipboardSet: (clipboard) => fs.clipboardSet(clipboard),
		clipboardGet: () => fs.clipboardGet(),
		versionFamilies: async () => {
			const result = await fs.versionFamilies();
			return {
				families: result.families.map((family) => ({
					head: family.head,
					members: family.members,
					entriesByPath: Object.fromEntries(
						family.entries.map((entry) => [entry.path, entryFromFs(entry)]),
					),
				})),
				error: result.error,
			};
		},
	};
}

/* ----------------------------------------------------------- fixture source */

interface FixtureState {
	tree: Map<string, FixtureItem[]>;
	clipboard: FsClipboard | null;
}

function seedFixtureState(): FixtureState {
	const tree = new Map<string, FixtureItem[]>();
	for (const [path, items] of Object.entries(FIXTURE_TREE)) {
		tree.set(path, [...items]);
	}
	return { tree, clipboard: null };
}

function takenNames(state: FixtureState, dir: string): string[] {
	return (state.tree.get(dir) ?? []).map((item) => item.name);
}

function removeItem(state: FixtureState, path: string): FixtureItem | null {
	const dir = dirname(path);
	const name = basename(path);
	const items = state.tree.get(dir);
	if (items === undefined) {
		return null;
	}
	const index = items.findIndex((item) => item.name === name);
	if (index < 0) {
		return null;
	}
	const [removed] = items.splice(index, 1);
	return removed ?? null;
}

function insertItem(state: FixtureState, dir: string, item: FixtureItem): void {
	const items = state.tree.get(dir);
	if (items === undefined) {
		state.tree.set(dir, [item]);
		return;
	}
	items.push(item);
}

function findItem(state: FixtureState, path: string): FixtureItem | null {
	const items = state.tree.get(dirname(path)) ?? [];
	return items.find((item) => item.name === basename(path)) ?? null;
}

export function createFixtureSource(): FinderSource {
	const state = seedFixtureState();
	const resolve = (): Promise<void> => Promise.resolve();

	return {
		native: false,
		initialPath: FIXTURE_INITIAL_PATH,
		pendingRequests: FIXTURE_PENDING_REQUESTS,
		list: (path) =>
			Promise.resolve((state.tree.get(path) ?? []).map((item) => entryFromFixture(item, path))),
		search: (query) => {
			const needle = query.trim().toLowerCase();
			if (needle === "") {
				return Promise.resolve([]);
			}
			const hits: FinderEntry[] = [];
			for (const [path, items] of state.tree) {
				if (path === "Recents") {
					continue;
				}
				for (const item of items) {
					if (item.name.toLowerCase().includes(needle)) {
						hits.push(entryFromFixture(item, path));
					}
				}
			}
			return Promise.resolve(hits);
		},
		locations: () =>
			Promise.resolve(
				FIXTURE_ROOTS.map((name) => ({
					name,
					path: name,
					section: name === "Recents" || name === "Desktop" || name === "Downloads" || name === "Documents"
						? ("favorites" as const)
						: ("cloud" as const),
					available: state.tree.has(name),
				})),
			),
		quickLook: resolve,
		open: resolve,
		reveal: resolve,
		trash: (paths) => {
			for (const path of paths) {
				removeItem(state, path);
			}
			return resolve();
		},
		rename: (path, name) => {
			const item = findItem(state, path);
			if (item !== null) {
				removeItem(state, path);
				insertItem(state, dirname(path), { ...item, name });
			}
			return resolve();
		},
		duplicate: (paths) => {
			for (const path of paths) {
				const item = findItem(state, path);
				if (item === null) {
					continue;
				}
				const dir = dirname(path);
				insertItem(state, dir, { ...item, name: copyName(item.name, [...takenNames(state, dir), item.name]) });
			}
			return resolve();
		},
		copyInto: (paths, destDir) => {
			for (const path of paths) {
				const item = findItem(state, path);
				if (item === null) {
					continue;
				}
				insertItem(state, destDir, { ...item, name: copyName(item.name, takenNames(state, destDir)) });
			}
			return resolve();
		},
		moveInto: (paths, destDir) => {
			for (const path of paths) {
				const item = findItem(state, path);
				if (item === null || dirname(path) === destDir) {
					continue;
				}
				removeItem(state, path);
				insertItem(state, destDir, { ...item, name: copyName(item.name, takenNames(state, destDir)) });
			}
			return resolve();
		},
		copyPaths: resolve,
		clipboardSet: (clipboard) => {
			state.clipboard = clipboard;
			return resolve();
		},
		clipboardGet: () => Promise.resolve(state.clipboard),
		versionFamilies: () =>
			Promise.resolve({
				families: FIXTURE_FAMILIES.flatMap((family) => {
					const headItem = findItem(state, family.head);
					if (headItem === null) {
						return [];
					}
					const entriesByPath: Record<string, FinderEntry> = {
						[family.head]: entryFromFixture(headItem, dirname(family.head)),
					};
					const members = family.members.flatMap(([where, name, relation]) => {
						const item = (state.tree.get(where) ?? []).find((candidate) => candidate.name === name);
						if (item === undefined) {
							return [];
						}
						const path = joinPath(where, name);
						entriesByPath[path] = entryFromFixture(item, where);
						return [{ path, relation: relation as VersionRelation }];
					});
					return [{ head: family.head, members, entriesByPath }];
				}),
				error: null,
			}),
	};
}

/* ------------------------------------------------------------- port picking */

/** Favorites order used to choose the tab the real app opens on. */
const PREFERRED_START = ["Documents", "Desktop", "Downloads"];

export function pickInitialPath(locations: readonly FinderLocation[]): string | null {
	for (const name of PREFERRED_START) {
		const match = locations.find((location) => location.name === name && location.available);
		if (match !== undefined) {
			return match.path;
		}
	}
	return locations.find((location) => location.available)?.path ?? null;
}
