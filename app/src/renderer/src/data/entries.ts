/**
 * The row model the Finder renders, plus adapters for its two sources: the
 * Electron fs bridge (OS-absolute paths, ISO stamps, byte sizes) and the
 * prototype fixtures (display-domain labels such as "14 msgs").
 */

import type { FsEntry, FsSearchResult } from "../../../shared/fs-contract";
import { formatModified, formatSize, parseModifiedLabel, parseSizeLabel } from "../state/format";
import { type FileKind, kindFromExtension, kindMeta } from "../state/kinds";
import { basename, dirname, joinPath } from "../state/paths";
import type { SortableEntry } from "../state/sort";

export interface FinderEntry extends SortableEntry {
	readonly name: string;
	/** Full path; the selection, clipboard, and fs calls all key on this. */
	readonly path: string;
	readonly kind: "folder" | "file";
	readonly fileKind: FileKind;
	/** OS tile icon (Finder thumbnail) as a data URL; null renders the letter tile. */
	readonly iconDataUrl: string | null;
	readonly dateLabel: string;
	readonly sizeLabel: string;
}

export function entryFromFs(entry: FsEntry, location?: string): FinderEntry {
	const fileKind: FileKind = entry.kind === "folder" ? "folder" : kindFromExtension(entry.ext);
	return {
		name: entry.name,
		path: entry.path,
		kind: entry.kind,
		fileKind,
		iconDataUrl: entry.iconDataUrl,
		dateLabel: formatModified(entry.modifiedAt),
		sizeLabel: formatSize(entry.kind === "folder" ? null : entry.size),
		// The OS-reported kind wins; the extension map is the fallback.
		kindLabel: entry.osKind ?? kindMeta(fileKind).label,
		location: location ?? dirname(entry.path),
		modifiedValue: Date.parse(entry.modifiedAt) || 0,
		sizeValue: entry.size ?? 0,
	};
}

export function entryFromSearchHit(hit: FsSearchResult): FinderEntry {
	return { ...entryFromFs(hit.entry), location: dirname(hit.entry.path) };
}

/** A prototype fixture row: the reference's `F()` / `D()` shape. */
export interface FixtureItem {
	readonly name: string;
	readonly fileKind: FileKind;
	readonly dateLabel: string;
	readonly sizeLabel: string;
}

export function entryFromFixture(item: FixtureItem, location: string): FinderEntry {
	return {
		name: item.name,
		path: joinPath(location, item.name),
		kind: item.fileKind === "folder" ? "folder" : "file",
		fileKind: item.fileKind,
		iconDataUrl: null,
		dateLabel: item.dateLabel,
		sizeLabel: item.sizeLabel,
		kindLabel: kindMeta(item.fileKind).label,
		location,
		modifiedValue: parseModifiedLabel(item.dateLabel),
		sizeValue: parseSizeLabel(item.sizeLabel),
	};
}

export function fixtureFromEntry(entry: FinderEntry): FixtureItem {
	return {
		name: entry.name,
		fileKind: entry.fileKind,
		dateLabel: entry.dateLabel,
		sizeLabel: entry.sizeLabel,
	};
}

export function renameEntry(entry: FinderEntry, name: string): FinderEntry {
	return { ...entry, name, path: joinPath(entry.location, name) };
}

export function displayName(path: string): string {
	return basename(path);
}
