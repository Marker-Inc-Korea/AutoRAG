/**
 * Finder contract shared by the Electron main process and the renderer.
 *
 * The main process implements every channel; the preload exposes typed
 * wrappers on `window.autorag.fs`. All paths crossing this boundary are
 * absolute OS paths. Datasource virtual identifiers (`/kakao/...`) never
 * appear here — they belong to the search bridge, not the filesystem.
 */

/** IPC channel names. Keep in 1:1 sync with the main-process handlers. */
export const FS_CHANNELS = {
	listDir: "fs:listDir",
	stat: "fs:stat",
	search: "fs:search",
	locations: "fs:locations",
	copy: "fs:copy",
	move: "fs:move",
	duplicate: "fs:duplicate",
	rename: "fs:rename",
	trash: "fs:trash",
	reveal: "fs:reveal",
	quickLook: "fs:quickLook",
	open: "fs:open",
	clipboardSet: "fs:clipboardSet",
	clipboardGet: "fs:clipboardGet",
	copyPathsToClipboard: "fs:copyPathsToClipboard",
	versionFamilies: "fs:versionFamilies",
} as const;

export type FsChannel = (typeof FS_CHANNELS)[keyof typeof FS_CHANNELS];

export interface FsEntry {
	/** File or folder name (last path segment). */
	readonly name: string;
	/** Absolute OS path. */
	readonly path: string;
	readonly kind: "folder" | "file";
	/** Lowercase extension without the dot; "" for folders and extensionless files. */
	readonly ext: string;
	/** Bytes; null for folders. */
	readonly size: number | null;
	/** ISO 8601 modification time. */
	readonly modifiedAt: string;
	readonly isSymlink: boolean;
	/**
	 * The kind the OS itself reports — Finder's Kind / Explorer's Type, e.g.
	 * "MPEG-4 movie". Null for folders and whenever the OS answers nothing; the
	 * renderer then falls back to the extension map.
	 */
	readonly osKind: string | null;
}

export interface DirListing {
	readonly path: string;
	readonly entries: readonly FsEntry[];
}

/** A browsable top-level location shown in the sidebar. */
export interface FsLocation {
	/** Display name, e.g. "Desktop", "Documents". */
	readonly name: string;
	readonly path: string;
	/** Sidebar section for grouping. */
	readonly section: "favorites" | "cloud";
	/** True when the path exists and is readable right now. */
	readonly available: boolean;
}

/** In-app file clipboard (copy/cut + paste between folders). */
export interface FsClipboard {
	readonly op: "copy" | "cut";
	readonly paths: readonly string[];
}

export interface FsOpError {
	readonly path: string;
	readonly message: string;
}

/** Batch operation result: one entry per requested path. */
export interface FsBatchResult {
	readonly ok: readonly string[];
	readonly failed: readonly FsOpError[];
}

/** Substring filename search over every available location (except Recents). */
export interface FsSearchResult {
	readonly entry: FsEntry;
	/** The location name this hit belongs to, e.g. "Documents". */
	readonly location: string;
}

export interface FsBridge {
	listDir(path: string): Promise<DirListing>;
	stat(path: string): Promise<FsEntry>;
	search(query: string): Promise<readonly FsSearchResult[]>;
	locations(): Promise<readonly FsLocation[]>;
	/** Copy paths into destDir. Name collisions get a " copy" suffix (Finder-style). */
	copy(paths: readonly string[], destDir: string): Promise<FsBatchResult>;
	/** Move paths into destDir (cut + paste). */
	move(paths: readonly string[], destDir: string): Promise<FsBatchResult>;
	/** Duplicate in place (Finder "Duplicate": "<name> copy"). */
	duplicate(paths: readonly string[]): Promise<FsBatchResult>;
	/** Rename a single entry. Rejects empty names and name collisions. */
	rename(path: string, newName: string): Promise<FsEntry>;
	/** Move to the macOS Trash (shell.trashItem). */
	trash(paths: readonly string[]): Promise<FsBatchResult>;
	/** Reveal in the real Finder (shell.showItemInFolder). */
	reveal(path: string): Promise<void>;
	/**
	 * Open the NATIVE macOS Quick Look for the path, exactly like pressing
	 * Space in Finder. Implementation: spawn `qlmanage -p <path>` detached.
	 * Folders and missing paths are rejected by the caller, not here.
	 */
	quickLook(path: string): Promise<void>;
	/**
	 * Open the path with the OS default application for its type, exactly
	 * like double-clicking the file in Finder / Explorer. Implementation:
	 * Electron `shell.openPath`; a non-empty result message rejects.
	 */
	open(path: string): Promise<void>;
	clipboardSet(clipboard: FsClipboard): Promise<void>;
	clipboardGet(): Promise<FsClipboard | null>;
	/** Write the absolute paths as text to the OS clipboard ("Copy Path"). */
	copyPathsToClipboard(paths: readonly string[]): Promise<void>;
	/**
	 * Version families from the dupey duplicate detector: one entry per family
	 * (head path, members with relations, entries for rendering), plus an
	 * explicit error when dupey is missing or a scan failed.
	 */
	versionFamilies(): Promise<FsVersionFamiliesResult>;
}

export type FsVersionRelation = "exact" | "near" | "contains";

export interface FsVersionMember {
	readonly path: string;
	readonly relation: FsVersionRelation;
}

export interface FsVersionFamily {
	readonly head: string;
	readonly members: readonly FsVersionMember[];
	readonly entries: readonly FsEntry[];
}

export interface FsVersionFamilyError {
	readonly code: "dupey-missing" | "scan-failed";
	readonly message: string;
	/** Present for dupey-missing: the command that installs the required CLI. */
	readonly installCommand: string | null;
}

/**
 * Version families never degrade silently: the renderer renders `error` when
 * the dupey CLI is missing or a scan fails.
 */
export interface FsVersionFamiliesResult {
	readonly families: readonly FsVersionFamily[];
	readonly error: FsVersionFamilyError | null;
}
