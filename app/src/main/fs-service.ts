import { spawn } from "node:child_process";
import type { Dirent } from "node:fs";
import { cp, lstat, readdir, rename as fsRename, rm } from "node:fs/promises";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import { scanWithDupey, type DupeyScanResult } from "@autorag/librarian";
import {
	RECENTS_PATH,
	type DirListing,
	type FsBatchResult,
	type FsBridge,
	type FsClipboard,
	type FsEntry,
	type FsLocation,
	type FsOpError,
	type FsSearchResult,
	type FsVersionFamiliesResult,
	type FsVersionFamily,
	type FsVersionFamilyError,
	type FsVersionMember,
	type FsVersionRelation,
} from "../shared/fs-contract";
import { createDupeyProbe, DUPEY_INSTALL_COMMAND, type DupeyProbe } from "./dupey";
import { buildFsEntry, errorMessage, hasErrorCode, isAvailable, pathExists, resolveCopyName } from "./fs-entry";
import { createRecentsStore, type RecentsStore } from "./recents-store";

/** Subset of Electron's shell used by the service (injected for testability). */
export interface FsShell {
	trashItem(path: string): Promise<void>;
	showItemInFolder(fullPath: string): void;
	/** Electron shell.openPath: resolves with "" on success or a verbatim error message. */
	openPath(path: string): Promise<string>;
}

/** Subset of Electron's clipboard used by the service (injected for testability). */
export interface FsOsClipboard {
	writeText(text: string): void;
}

export type QuickLookSpawner = (path: string) => void;

export interface FsServiceDeps {
	readonly shell: FsShell;
	readonly clipboard: FsOsClipboard;
	/** Defaults to os.homedir(); tests inject a fixture home. */
	readonly homeDir?: string;
	/** Defaults to spawning `/usr/bin/qlmanage -p <path>` detached. */
	readonly spawnQuickLook?: QuickLookSpawner;
	/** Defaults to the dupey CLI; tests inject fixture scans. */
	readonly scanDuplicates?: (dir: string) => Promise<DupeyScanResult>;
	/** Defaults to probing the dupey CLI; tests inject a fixed status. */
	readonly dupey?: DupeyProbe;
	/** Clock for the family cache TTL; tests inject a fixed one. */
	readonly now?: () => number;
	/** Recently opened/previewed files. Defaults to `<homeDir>/recents`; production injects the userData store. */
	readonly recents?: RecentsStore;
}

export type FsRenameErrorCode = "invalid-name" | "name-collision";

export class FsRenameError extends Error {
	readonly code: FsRenameErrorCode;

	constructor(code: FsRenameErrorCode, message: string) {
		super(message);
		this.name = "FsRenameError";
		this.code = code;
	}
}

/** Search stays bounded: shallow walk, hard result cap, symlinked dirs not followed. */
const SEARCH_MAX_DEPTH = 4;
const SEARCH_MAX_RESULTS = 200;

interface SearchContext {
	readonly needle: string;
	readonly location: string;
	readonly results: FsSearchResult[];
}

export function previewCommandForPlatform(
	platform: NodeJS.Platform,
	path: string,
): { readonly command: string; readonly args: readonly string[] } {
	if (platform === "darwin") return { command: "/usr/bin/qlmanage", args: ["-p", path] };
	if (platform === "win32") return { command: "explorer.exe", args: [path] };
	return { command: "xdg-open", args: [path] };
}

function defaultQuickLook(path: string): void {
	const preview = previewCommandForPlatform(process.platform, path);
	const child = spawn(preview.command, preview.args, { detached: true, stdio: "ignore" });
	child.unref();
}

function compareEntries(a: FsEntry, b: FsEntry): number {
	if (a.kind !== b.kind) return a.kind === "folder" ? -1 : 1;
	return a.name.localeCompare(b.name);
}

async function runBatch(paths: readonly string[], operation: (path: string) => Promise<void>): Promise<FsBatchResult> {
	const ok: string[] = [];
	const failed: FsOpError[] = [];
	for (const path of paths) {
		try {
			await operation(path);
			ok.push(path);
		} catch (error) {
			failed.push({ path, message: errorMessage(error) });
		}
	}
	return { ok, failed };
}

async function walkForSearch(dir: string, depth: number, context: SearchContext): Promise<void> {
	if (depth > SEARCH_MAX_DEPTH || context.results.length >= SEARCH_MAX_RESULTS) return;
	let dirents: Dirent[];
	try {
		dirents = await readdir(dir, { withFileTypes: true });
	} catch (error) {
		// Unreadable directory: skip the subtree, keep the verbatim reason visible.
		console.warn(`fs:search skipping ${dir}: ${errorMessage(error)}`);
		return;
	}
	for (const dirent of dirents) {
		if (context.results.length >= SEARCH_MAX_RESULTS) return;
		const fullPath = join(dir, dirent.name);
		if (dirent.name.toLowerCase().includes(context.needle)) {
			try {
				context.results.push({ entry: await buildFsEntry(fullPath), location: context.location });
			} catch (error) {
				console.warn(`fs:search skipping ${fullPath}: ${errorMessage(error)}`);
			}
		}
		if (dirent.isDirectory() && !dirent.isSymbolicLink()) {
			await walkForSearch(fullPath, depth + 1, context);
		}
	}
}

const VERSION_FAMILY_TTL_MS = 5 * 60 * 1000;

interface DupeyMemberInfo {
	readonly path?: string;
	readonly relation?: string;
	readonly exact_hash?: string;
	readonly joined_with?: string;
}

function clampRelation(value: string | undefined): FsVersionRelation {
	return value === "exact" || value === "contains" ? value : "near";
}

/** dupey family (pick-keeper head + per-member relations) to the app contract. */
export function mapDupeyFamily(
	family: DupeyScanResult["families"][number],
): { readonly head: string; readonly members: readonly FsVersionMember[] } | null {
	const head =
		(family.pick as { readonly ranked?: readonly { readonly path?: string }[] } | undefined)?.ranked?.[0]
			?.path ?? family.files[0];
	if (typeof head !== "string" || head.length === 0) return null;
	const memberInfos = new Map<string, DupeyMemberInfo>();
	for (const member of family.members as readonly DupeyMemberInfo[]) {
		if (typeof member?.path === "string") memberInfos.set(member.path, member);
	}
	const headHash = memberInfos.get(head)?.exact_hash;
	const members: FsVersionMember[] = [];
	for (const path of family.files) {
		if (path === head) continue;
		const info = memberInfos.get(path);
		const relation =
			info?.exact_hash !== undefined && headHash !== undefined && info.exact_hash === headHash
				? "exact"
				: clampRelation(info?.relation);
		members.push({ path, relation });
	}
	return { head, members };
}

export function createFsService(deps: FsServiceDeps): FsBridge {
	const home = deps.homeDir ?? homedir();
	const spawnQuickLook = deps.spawnQuickLook ?? defaultQuickLook;
	const scanDuplicates = deps.scanDuplicates ?? ((dir: string) => scanWithDupey(dir));
	const dupey = deps.dupey ?? createDupeyProbe();
	const now = deps.now ?? (() => Date.now());
	const recents = deps.recents ?? createRecentsStore({ directory: join(home, "recents") });
	// In-app clipboard lives in main-process memory only.
	let inAppClipboard: FsClipboard | null = null;
	const familyCache = new Map<string, { readonly at: number; readonly families: readonly FsVersionFamily[] }>();

	async function mutatingBatch(
		paths: readonly string[],
		operation: (path: string) => Promise<void>,
	): Promise<FsBatchResult> {
		const result = await runBatch(paths, operation);
		if (result.ok.length > 0) familyCache.clear();
		return result;
	}

	async function listDir(path: string): Promise<DirListing> {
		if (path === RECENTS_PATH) {
			return { path, entries: await recentEntries() };
		}
		const dirents = await readdir(path, { withFileTypes: true });
		const entries: FsEntry[] = [];
		for (const dirent of dirents) {
			const fullPath = join(path, dirent.name);
			try {
				entries.push(await buildFsEntry(fullPath));
			} catch (error) {
				// Unreadable entry: skip it, keep the verbatim reason visible.
				console.warn(`fs:listDir skipping ${fullPath}: ${errorMessage(error)}`);
			}
		}
		entries.sort(compareEntries);
		return { path, entries };
	}

	/**
	 * The virtual Recents listing: recorded paths that still exist, most recent
	 * first. A file that moved or was deleted simply drops out of the view; any
	 * other failure keeps its verbatim reason visible.
	 */
	async function recentEntries(): Promise<FsEntry[]> {
		const entries: FsEntry[] = [];
		for (const stored of await recents.list()) {
			try {
				entries.push(await buildFsEntry(stored));
			} catch (error) {
				if (!hasErrorCode(error, "ENOENT") && !hasErrorCode(error, "ENOTDIR")) {
					console.warn(`fs:listDir ${RECENTS_PATH} skipping ${stored}: ${errorMessage(error)}`);
				}
			}
		}
		return entries;
	}

	async function locations(): Promise<readonly FsLocation[]> {
		const result: FsLocation[] = [
			{ name: RECENTS_PATH, path: RECENTS_PATH, section: "favorites", available: true },
		];
		const fixed: readonly { readonly name: string; readonly path: string; readonly section: "favorites" | "cloud" }[] =
			[
				{ name: "Desktop", path: join(home, "Desktop"), section: "favorites" },
				{ name: "Downloads", path: join(home, "Downloads"), section: "favorites" },
				{ name: "Documents", path: join(home, "Documents"), section: "favorites" },
				{ name: "iCloud Drive", path: join(home, "Library", "Mobile Documents", "com~apple~CloudDocs"), section: "cloud" },
			];
		for (const spec of fixed) {
			result.push({ name: spec.name, path: spec.path, section: spec.section, available: await isAvailable(spec.path) });
		}
		// Third-party cloud drives appear only when their well-known paths exist.
		const cloudStorage = join(home, "Library", "CloudStorage");
		const googleDrive = await resolveCloudDrive(cloudStorage, "GoogleDrive-", join(home, "Google Drive"));
		if (googleDrive !== null) {
			result.push({ name: "Google Drive", path: googleDrive, section: "cloud", available: true });
		}
		const dropbox = await resolveCloudDrive(cloudStorage, "Dropbox", join(home, "Dropbox"));
		if (dropbox !== null) {
			result.push({ name: "Dropbox", path: dropbox, section: "cloud", available: true });
		}
		return result;
	}

	async function resolveCloudDrive(cloudStorage: string, prefix: string, legacyPath: string): Promise<string | null> {
		try {
			const names = await readdir(cloudStorage);
			const match = names.find((name) => name.startsWith(prefix));
			if (match !== undefined) return join(cloudStorage, match);
		} catch (error) {
			// No CloudStorage directory means no modern-style drive; fall through to the legacy path.
			if (!hasErrorCode(error, "ENOENT") && !hasErrorCode(error, "ENOTDIR")) throw error;
		}
		return (await pathExists(legacyPath)) ? legacyPath : null;
	}

	async function search(query: string): Promise<readonly FsSearchResult[]> {
		const needle = query.trim().toLowerCase();
		if (needle.length === 0) return [];
		const results: FsSearchResult[] = [];
		for (const location of await locations()) {
			if (!location.available || location.path === RECENTS_PATH) continue;
			await walkForSearch(location.path, 0, { needle, location: location.name, results });
			if (results.length >= SEARCH_MAX_RESULTS) break;
		}
		return results;
	}

	async function versionFamilies(): Promise<FsVersionFamiliesResult> {
		const status = await dupey.status();
		if (!status.available) {
			return {
				families: [],
				error: {
					code: "dupey-missing",
					message: `dupey CLI is required for version stacks: ${status.error ?? "not found on PATH"}`,
					installCommand: DUPEY_INSTALL_COMMAND,
				},
			};
		}
		const result: FsVersionFamily[] = [];
		const scanErrors: string[] = [];
		for (const location of await locations()) {
			if (!location.available || location.path === RECENTS_PATH) continue;
			const cached = familyCache.get(location.path);
			if (cached !== undefined && now() - cached.at < VERSION_FAMILY_TTL_MS) {
				result.push(...cached.families);
				continue;
			}
			let families: readonly FsVersionFamily[] = [];
			try {
				const scan = await scanDuplicates(location.path);
				const resolved = await Promise.all(
					scan.families.map(async (family): Promise<FsVersionFamily | null> => {
						const mapped = mapDupeyFamily(family);
						if (mapped === null || mapped.members.length === 0) return null;
						const entries: FsEntry[] = [];
						try {
							entries.push(await buildFsEntry(mapped.head));
						} catch {
							return null;
						}
						for (const member of mapped.members) {
							try {
								entries.push(await buildFsEntry(member.path));
							} catch {
								continue;
							}
						}
						const members = mapped.members.filter((member) =>
							entries.some((entry) => entry.path === member.path),
						);
						if (members.length === 0) return null;
						return { head: mapped.head, members, entries };
					}),
				);
				families = resolved.filter((family): family is FsVersionFamily => family !== null);
			} catch (error) {
				const message = `fs:versionFamilies failed for ${location.path}: ${errorMessage(error)}`;
				console.error(message);
				scanErrors.push(message);
			}
			familyCache.set(location.path, { at: now(), families });
			result.push(...families);
		}
		const error: FsVersionFamilyError | null =
			scanErrors.length === 0
				? null
				: { code: "scan-failed", message: scanErrors.join("\n"), installCommand: null };
		return { families: result, error };
	}

	async function copy(paths: readonly string[], destDir: string): Promise<FsBatchResult> {
		return mutatingBatch(paths, async (source) => {
			const sourceStat = await lstat(source);
			const targetName = await resolveCopyName(destDir, basename(source), sourceStat.isDirectory());
			await cp(source, join(destDir, targetName), { recursive: true });
		});
	}

	async function move(paths: readonly string[], destDir: string): Promise<FsBatchResult> {
		return mutatingBatch(paths, async (source) => {
			const sourceStat = await lstat(source);
			const target = join(destDir, await resolveCopyName(destDir, basename(source), sourceStat.isDirectory()));
			try {
				await fsRename(source, target);
			} catch (error) {
				// Cross-device move: copy the tree, then remove the source.
				if (!hasErrorCode(error, "EXDEV")) throw error;
				await cp(source, target, { recursive: true });
				await rm(source, { recursive: true, force: true });
			}
		});
	}

	async function duplicate(paths: readonly string[]): Promise<FsBatchResult> {
		return mutatingBatch(paths, async (source) => {
			const sourceStat = await lstat(source);
			const dir = dirname(source);
			const targetName = await resolveCopyName(dir, basename(source), sourceStat.isDirectory());
			await cp(source, join(dir, targetName), { recursive: true });
		});
	}

	async function rename(path: string, newName: string): Promise<FsEntry> {
		if (newName.trim().length === 0) {
			throw new FsRenameError("invalid-name", "Invalid name: the new name must not be empty");
		}
		if (newName.includes("/") || newName.includes("\\") || newName === "." || newName === "..") {
			throw new FsRenameError("invalid-name", `Invalid name "${newName}": path separators are not allowed`);
		}
		const dir = dirname(path);
		const currentName = basename(path);
		if (newName === currentName) {
			throw new FsRenameError("name-collision", `A file or folder named "${newName}" already exists`);
		}
		const target = join(dir, newName);
		if (newName.toLowerCase() !== currentName.toLowerCase() && (await pathExists(target))) {
			throw new FsRenameError("name-collision", `A file or folder named "${newName}" already exists`);
		}
		await fsRename(path, target);
		familyCache.clear();
		return buildFsEntry(target);
	}

	return {
		listDir,
		stat: buildFsEntry,
		search,
		locations,
		copy,
		move,
		duplicate,
		rename,
		trash: (paths) => mutatingBatch(paths, (path) => deps.shell.trashItem(path)),
		reveal: async (path) => {
			deps.shell.showItemInFolder(path);
		},
		quickLook: async (path) => {
			await recents.record(path);
			spawnQuickLook(path);
		},
		open: async (path) => {
			const message = await deps.shell.openPath(path);
			// Failures surface verbatim — the operator sees the real message.
			if (message !== "") {
				throw new Error(message);
			}
			await recents.record(path);
		},
		clipboardSet: async (clipboard) => {
			inAppClipboard = { op: clipboard.op, paths: [...clipboard.paths] };
		},
		clipboardGet: async () => inAppClipboard,
		copyPathsToClipboard: async (paths) => {
			deps.clipboard.writeText(paths.join("\n"));
		},
		versionFamilies,
	};
}
