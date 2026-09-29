import { constants } from "node:fs";
import { access, lstat, stat } from "node:fs/promises";
import { basename, join } from "node:path";
import type { FsEntry } from "../shared/fs-contract";

/** True when the error carries the given Node errno code. */
export function hasErrorCode(error: unknown, code: string): boolean {
	return error instanceof Error && "code" in error && error.code === code;
}

/** Verbatim error text for operator-facing diagnostics. */
export function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : String(error);
}

export async function pathExists(path: string): Promise<boolean> {
	try {
		await access(path);
		return true;
	} catch (error) {
		if (hasErrorCode(error, "ENOENT") || hasErrorCode(error, "ENOTDIR")) return false;
		throw error;
	}
}

export async function isAvailable(path: string): Promise<boolean> {
	try {
		await access(path, constants.R_OK);
		return true;
	} catch (error) {
		if (
			hasErrorCode(error, "ENOENT") ||
			hasErrorCode(error, "ENOTDIR") ||
			hasErrorCode(error, "EACCES") ||
			hasErrorCode(error, "EPERM")
		) {
			return false;
		}
		throw error;
	}
}

/** Lowercase extension without the dot; "" for extensionless names and dotfiles. */
export function extensionOf(name: string): string {
	const dot = name.lastIndexOf(".");
	return dot <= 0 ? "" : name.slice(dot + 1).toLowerCase();
}

function splitBaseExt(name: string, isFolder: boolean): { readonly base: string; readonly ext: string } {
	if (isFolder) return { base: name, ext: "" };
	const dot = name.lastIndexOf(".");
	return dot <= 0 ? { base: name, ext: "" } : { base: name.slice(0, dot), ext: name.slice(dot) };
}

/**
 * Finder-style collision naming inside destDir: the bare name when free,
 * otherwise "name copy.ext", "name copy 2.ext", ... (ext stays last).
 */
export async function resolveCopyName(destDir: string, name: string, isFolder: boolean): Promise<string> {
	if (!(await pathExists(join(destDir, name)))) return name;
	const { base, ext } = splitBaseExt(name, isFolder);
	for (let attempt = 1; ; attempt += 1) {
		const candidate = attempt === 1 ? `${base} copy${ext}` : `${base} copy ${attempt}${ext}`;
		if (!(await pathExists(join(destDir, candidate)))) return candidate;
	}
}

/**
 * Build the contract entry for one on-disk path. Symlinks are reported with
 * the target's metadata when resolvable, else with the link's own metadata.
 */
export async function buildFsEntry(path: string): Promise<FsEntry> {
	const linkStat = await lstat(path);
	const isSymlink = linkStat.isSymbolicLink();
	let effective = linkStat;
	if (isSymlink) {
		try {
			effective = await stat(path);
		} catch (error) {
			// Broken symlink: fall back to the link's own metadata.
			if (!hasErrorCode(error, "ENOENT")) throw error;
		}
	}
	const isFolder = effective.isDirectory();
	const name = basename(path);
	return {
		name,
		path,
		kind: isFolder ? "folder" : "file",
		ext: isFolder ? "" : extensionOf(name),
		size: isFolder ? null : effective.size,
		modifiedAt: effective.mtime.toISOString(),
		isSymlink,
		// The OS metadata lookup lands in fs-service, which owns process spawning.
		osKind: null,
	};
}
