import { randomUUID } from "node:crypto";
import type { BigIntStats, Dirent } from "node:fs";
import {
	copyFileSync,
	lstatSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmdirSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { opendir, stat } from "node:fs/promises";
import { basename, dirname, extname, join, relative, sep } from "node:path";
import { renameWithRetry } from "../../../mirror/atomic-write.ts";

/**
 * Vault walking and staging.
 *
 * MinSync indexes whatever markdown sits under `<workspace>/files`, so the vault is mirrored there:
 * only notes, only what changed, and never the vault itself (it is read, not written). A staging
 * record remembers what was copied, which is how an unchanged note is skipped, a removed note is
 * deleted from the index, and a staged path is mapped back to the note it came from.
 */

const NOTE_EXTENSIONS: ReadonlySet<string> = new Set([".md", ".markdown"]);
const SKIPPED_DIRECTORIES: ReadonlySet<string> = new Set(["node_modules"]);
const MAX_WALK_DEPTH = 64;

export interface VaultNote {
	/** Vault-relative path, NFC-normalized, `/`-separated. */
	readonly relativePath: string;
	readonly sourcePath: string;
	readonly size: number;
	/** Exact nanosecond mtime as a decimal string (it exceeds a double's integer range). */
	readonly mtimeNs: string;
}

export interface VaultListing {
	readonly notes: ReadonlyMap<string, VaultNote>;
	/** Folders (vault-relative) that could not be read; their staged notes are kept as they were. */
	readonly unreadableFolders: readonly string[];
	/** Notes dropped because another note has the same NFC-normalized path. */
	readonly collisions: readonly string[];
}

export class VaultUnavailableError extends Error {
	readonly code: string;

	constructor(code: string) {
		super(`the Obsidian vault is not readable (${code})`);
		this.name = "VaultUnavailableError";
		this.code = code;
	}
}

function errnoCode(error: unknown): string {
	const code = error instanceof Error ? (error as NodeJS.ErrnoException).code : undefined;
	return typeof code === "string" ? code : "UNKNOWN";
}

/**
 * List the vault's notes. Dot-prefixed folders and files (`.obsidian`, `.trash`, `.git`),
 * `node_modules`, and symlinks are not part of the vault's knowledge and are skipped.
 * Throws {@link VaultUnavailableError} when the vault root itself cannot be read.
 */
export async function listVaultNotes(vaultPath: string): Promise<VaultListing> {
	try {
		if (!(await stat(vaultPath)).isDirectory()) throw new VaultUnavailableError("ENOTDIR");
		await (await opendir(vaultPath)).close();
	} catch (error) {
		if (error instanceof VaultUnavailableError) throw error;
		throw new VaultUnavailableError(errnoCode(error));
	}
	const notes = new Map<string, VaultNote>();
	const unreadableFolders: string[] = [];
	const collisions: string[] = [];

	const walk = async (directory: string, depth: number): Promise<void> => {
		let entries: Dirent[];
		try {
			entries = [];
			for await (const entry of await opendir(directory)) entries.push(entry);
		} catch {
			unreadableFolders.push(toRelative(vaultPath, directory));
			return;
		}
		entries.sort((a, b) => a.name.localeCompare(b.name));
		for (const entry of entries) {
			if (entry.name.startsWith(".") || entry.isSymbolicLink()) continue;
			const path = join(directory, entry.name);
			if (entry.isDirectory()) {
				if (SKIPPED_DIRECTORIES.has(entry.name)) continue;
				if (depth + 1 > MAX_WALK_DEPTH) {
					unreadableFolders.push(toRelative(vaultPath, path));
					continue;
				}
				await walk(path, depth + 1);
				continue;
			}
			if (!entry.isFile() || !NOTE_EXTENSIONS.has(extname(entry.name).toLowerCase())) continue;
			let fileStat: BigIntStats;
			try {
				fileStat = await stat(path, { bigint: true });
			} catch (error) {
				// A note deleted mid-walk is simply gone; anything else keeps its previous copy.
				if (errnoCode(error) !== "ENOENT") unreadableFolders.push(toRelative(vaultPath, directory));
				continue;
			}
			const relativePath = toRelative(vaultPath, path);
			if (notes.has(relativePath)) {
				collisions.push(relativePath);
				continue;
			}
			notes.set(relativePath, {
				relativePath,
				sourcePath: path,
				size: Number(fileStat.size),
				mtimeNs: fileStat.mtimeNs.toString(),
			});
		}
	};
	await walk(vaultPath, 0);
	return { notes, unreadableFolders, collisions };
}

function toRelative(vaultPath: string, path: string): string {
	return relative(vaultPath, path).split(sep).join("/").normalize("NFC");
}

interface StagingEntry {
	readonly sourcePath: string;
	readonly size: number;
	readonly mtimeNs: string;
}

export interface StagingState {
	readonly version: 1;
	readonly entries: Readonly<Record<string, StagingEntry>>;
}

export function loadStagingState(statePath: string): StagingState {
	try {
		const parsed: unknown = JSON.parse(readFileSync(statePath, "utf8"));
		if (
			typeof parsed === "object" &&
			parsed !== null &&
			(parsed as { version?: unknown }).version === 1 &&
			typeof (parsed as { entries?: unknown }).entries === "object" &&
			(parsed as { entries?: unknown }).entries !== null
		) {
			return parsed as StagingState;
		}
	} catch {
		// Missing or unreadable: the next stage copies everything.
	}
	return { version: 1, entries: {} };
}

function saveStagingState(statePath: string, state: StagingState): void {
	mkdirSync(dirname(statePath), { recursive: true });
	const temporary = `${statePath}.${randomUUID()}.tmp`;
	try {
		writeFileSync(temporary, `${JSON.stringify(state)}\n`);
		renameWithRetry(temporary, statePath);
	} finally {
		rmSync(temporary, { force: true });
	}
}

export interface StageResult {
	readonly staged: number;
	readonly copied: number;
	readonly removed: number;
	/** Notes that could not be copied this time; an earlier staged copy, if any, was kept. */
	readonly failed: readonly string[];
}

function isUnder(relativePath: string, folder: string): boolean {
	return folder === "" || relativePath === folder || relativePath.startsWith(`${folder}/`);
}

/**
 * Bring `<filesRoot>` in line with the vault: copy new and changed notes, delete notes that no
 * longer exist, and leave notes under an unreadable folder exactly as they were.
 */
export function stageNotes(
	filesRoot: string,
	statePath: string,
	listing: VaultListing,
	options: { readonly force?: boolean } = {},
): StageResult {
	const previous = loadStagingState(statePath);
	const next: Record<string, StagingEntry> = {};
	const failed: string[] = [];
	let copied = 0;
	mkdirSync(filesRoot, { recursive: true });

	for (const note of listing.notes.values()) {
		const target = join(filesRoot, ...note.relativePath.split("/"));
		const before = previous.entries[note.relativePath];
		const unchanged =
			options.force !== true &&
			before !== undefined &&
			before.size === note.size &&
			before.mtimeNs === note.mtimeNs &&
			lstatSync(target, { throwIfNoEntry: false })?.isFile() === true;
		if (!unchanged) {
			try {
				copyAtomically(note.sourcePath, target);
				copied += 1;
			} catch {
				failed.push(note.relativePath);
				if (before !== undefined) next[note.relativePath] = before;
				continue;
			}
		}
		next[note.relativePath] = { sourcePath: note.sourcePath, size: note.size, mtimeNs: note.mtimeNs };
	}
	for (const [relativePath, entry] of Object.entries(previous.entries)) {
		if (relativePath in next || listing.notes.has(relativePath)) continue;
		if (listing.unreadableFolders.some((folder) => isUnder(relativePath, folder))) next[relativePath] = entry;
	}

	const removed = removeUnexpected(filesRoot, filesRoot, new Set(Object.keys(next)));
	saveStagingState(statePath, { version: 1, entries: next });
	return { staged: Object.keys(next).length, copied, removed, failed };
}

function copyAtomically(source: string, destination: string): void {
	mkdirSync(dirname(destination), { recursive: true });
	const temporary = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.tmp`);
	try {
		copyFileSync(source, temporary);
		renameWithRetry(temporary, destination);
	} finally {
		rmSync(temporary, { force: true });
	}
}

/** Delete staged files that are not wanted and the folders they leave empty. Returns the files removed. */
function removeUnexpected(filesRoot: string, directory: string, wanted: ReadonlySet<string>): number {
	let removed = 0;
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) {
			removed += removeUnexpected(filesRoot, path, wanted);
			if (readdirSync(path).length === 0) rmdirSync(path);
		} else if (!wanted.has(toStagedRelative(filesRoot, path))) {
			rmSync(path, { force: true });
			removed += 1;
		}
	}
	return removed;
}

function toStagedRelative(filesRoot: string, path: string): string {
	return relative(filesRoot, path).split(sep).join("/").normalize("NFC");
}
