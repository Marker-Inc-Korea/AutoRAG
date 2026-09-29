/**
 * Column sorting for the Finder list — the 3-click cycle and the comparator.
 *
 * Reference: handoff README §2 "Column header" — first click ascending (Date
 * and Size start descending), second flips, third clears. Folders always sort
 * before files and Name compares with the Korean locale.
 */

export type SortKey = "name" | "date" | "size" | "kind" | "where";
export type SortDirection = 1 | -1;

export interface SortState {
	readonly key: SortKey;
	readonly dir: SortDirection;
	/** True once the direction has been flipped, i.e. the next click clears. */
	readonly flipped: boolean;
}

/** Columns whose first click sorts descending. */
const DESCENDING_FIRST: readonly SortKey[] = ["date", "size"];

export function cycleSort(current: SortState | null, key: SortKey): SortState | null {
	if (current !== null && current.key === key) {
		return current.flipped ? null : { key, dir: current.dir === 1 ? -1 : 1, flipped: true };
	}
	return { key, dir: DESCENDING_FIRST.includes(key) ? -1 : 1, flipped: false };
}

export interface SortableEntry {
	readonly name: string;
	readonly kind: "folder" | "file";
	/** Sortable modification stamp; see state/format.parseModifiedLabel. */
	readonly modifiedValue: number;
	/** Sortable size in bytes; folders are 0. */
	readonly sizeValue: number;
	readonly kindLabel: string;
	/** Parent location, shown in the "Where" column while searching. */
	readonly location: string;
}

const collator = new Intl.Collator("ko");

function byName(a: SortableEntry, b: SortableEntry): number {
	return collator.compare(a.name, b.name);
}

const COMPARATORS: Record<SortKey, (a: SortableEntry, b: SortableEntry) => number> = {
	name: byName,
	date: (a, b) => a.modifiedValue - b.modifiedValue,
	size: (a, b) => a.sizeValue - b.sizeValue,
	kind: (a, b) => collator.compare(a.kindLabel, b.kindLabel) || byName(a, b),
	where: (a, b) => collator.compare(a.location, b.location) || byName(a, b),
};

export function sortEntries<T extends SortableEntry>(entries: readonly T[], sort: SortState | null): T[] {
	if (sort === null) {
		return [...entries];
	}
	const compare = COMPARATORS[sort.key];
	return [...entries].sort((a, b) =>
		a.kind !== b.kind ? (a.kind === "folder" ? -1 : 1) : compare(a, b) * sort.dir,
	);
}
