import { randomUUID } from "node:crypto";
import {
	copyFileSync,
	existsSync,
	lstatSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { basename, dirname, isAbsolute, join, relative, sep } from "node:path";
import { normalizeVirtualPath } from "../filesystem/source-paths.ts";
import { renameWithRetry } from "../mirror/atomic-write.ts";
import { loadMirrorIndex } from "../mirror/index-store.ts";
import { parsedVisualOutputPath } from "../mirror/paths.ts";
import { isStandaloneImagePath } from "../parser/image-metadata.ts";
import {
	MINSYNC_VISUAL_SUFFIX,
	minSyncDocumentPath,
	minSyncImageDocumentPath,
	minSyncVisualDocumentPath,
	minSyncWorkspaceRoot,
} from "./paths.ts";

const NS_PER_MS = 1_000_000;

export type MinSyncMirrorKind = "text" | "visual" | "image";

export interface MinSyncWorkspaceEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly parsedOutputPath: string;
	readonly minSyncPath: string;
	/** The source file's last-modified time (epoch ms) as the mirror index recorded it; not when it was authored. */
	readonly sourceMtimeMs: number;
	/** Which parsed mirror of the source file this staged document came from. */
	readonly mirrorKind?: MinSyncMirrorKind;
}

export interface MinSyncWorkspaceSyncResult {
	readonly workspacePath: string;
	readonly entries: readonly MinSyncWorkspaceEntry[];
	/**
	 * Parsed-mirror ids that could not be staged: no canonical source-id form
	 * or a missing parsed output. Callers report these instead of dropping
	 * them silently.
	 */
	readonly excluded: readonly string[];
	/**
	 * Parsed-mirror ids that could not be materialized in the staging tree
	 * because the filesystem rejected the path length or name (e.g.
	 * `ENAMETOOLONG` or Windows `EINVAL`). Isolated per document so one long
	 * path never aborts the refresh for other files.
	 */
	readonly stagingFailed: readonly string[];
}

interface MinSyncStagingEntry {
	readonly outputPath: string;
	readonly updatedAt: string;
	readonly visualOutputPath?: string;
	readonly visualStamp?: string;
	readonly imageSourcePath?: string;
	readonly imageStamp?: string;
}

interface MinSyncStagingState {
	readonly version: 1;
	readonly entries: Readonly<Record<string, MinSyncStagingEntry>>;
}

function fileStamp(path: string): string | undefined {
	const stat = lstatSync(path, { throwIfNoEntry: false });
	if (!stat?.isFile()) return undefined;
	return `${stat.size}:${stat.mtimeMs}`;
}

export function syncMinSyncWorkspace(
	root: string,
	options: {
		readonly workspacePath?: string;
		readonly minSyncBinaryPath?: string;
		/**
		 * When true (e.g., `native-multimodal:google/embeddinggemma-2`), stage raw standalone image files
		 * alongside their parsed mirrors so MinSync embeds image pixels directly into the dense vector space.
		 */
		readonly stageRawImages?: boolean;
	} = {},
): MinSyncWorkspaceSyncResult {
	const workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(root);
	const filesRoot = join(workspacePath, "files");
	const index = loadMirrorIndex(root, options);
	const candidates = Object.values(index.entries).sort((a, b) => a.virtualPath.localeCompare(b.virtualPath));
	const excluded = candidates
		.filter((entry) => normalizeVirtualPath(entry.virtualPath) !== entry.virtualPath || !existsSync(entry.outputPath))
		.map((entry) => entry.virtualPath);
	const baseEntries: MinSyncWorkspaceEntry[] = candidates
		.filter((entry) => normalizeVirtualPath(entry.virtualPath) === entry.virtualPath && existsSync(entry.outputPath))
		.map((entry) => {
			const minSyncPath = minSyncDocumentPath(workspacePath, entry.virtualPath);
			return {
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parsedOutputPath: entry.outputPath,
				minSyncPath,
				sourceMtimeMs: Math.floor(entry.sourceMtimeNs / NS_PER_MS),
				mirrorKind: "text" as const,
			};
		});
	const previousState = loadStagingState(workspacePath);
	ensureManagedFilesRoot(filesRoot, previousState === undefined);

	const currentState: Record<string, MinSyncStagingEntry> = {};
	const desiredPaths = new Set<string>();
	const stagedEntries: MinSyncWorkspaceEntry[] = [];
	const stagingFailed: string[] = [];
	for (const entry of baseEntries) {
		const mirrorEntry = index.entries[entry.virtualPath];
		if (mirrorEntry === undefined) continue;
		const previousEntry = previousState?.entries[entry.virtualPath];
		let unchanged: boolean;
		try {
			unchanged =
				previousEntry?.outputPath === mirrorEntry.outputPath &&
				previousEntry.updatedAt === mirrorEntry.updatedAt &&
				isRegularFile(entry.minSyncPath);
		} catch (error) {
			if (!isPerDocumentStagingError(error)) throw error;
			stagingFailed.push(entry.virtualPath);
			continue;
		}
		if (!unchanged) {
			try {
				ensureManagedDirectory(filesRoot, dirname(entry.minSyncPath));
				copyFileAtomically(entry.parsedOutputPath, entry.minSyncPath);
			} catch (error) {
				if (!isPerDocumentStagingError(error)) throw error;
				stagingFailed.push(entry.virtualPath);
				continue;
			}
		}
		stagedEntries.push(entry);
		desiredPaths.add(entry.minSyncPath);

		const visualOutputPath = parsedVisualOutputPath(root, entry.virtualPath);
		const visualStamp = fileStamp(visualOutputPath);
		let visualStaged = false;
		if (visualStamp !== undefined) {
			const visualMinSyncPath = minSyncVisualDocumentPath(workspacePath, entry.virtualPath);
			try {
				const visualUnchanged =
					previousEntry?.visualOutputPath === visualOutputPath &&
					previousEntry.visualStamp === visualStamp &&
					isRegularFile(visualMinSyncPath);
				if (!visualUnchanged) {
					ensureManagedDirectory(filesRoot, dirname(visualMinSyncPath));
					copyFileAtomically(visualOutputPath, visualMinSyncPath);
				}
				stagedEntries.push({
					virtualPath: entry.virtualPath,
					sourcePath: entry.sourcePath,
					parsedOutputPath: visualOutputPath,
					minSyncPath: visualMinSyncPath,
					sourceMtimeMs: entry.sourceMtimeMs,
					mirrorKind: "visual",
				});
				desiredPaths.add(visualMinSyncPath);
				visualStaged = true;
			} catch (error) {
				if (!isPerDocumentStagingError(error)) throw error;
				stagingFailed.push(`${entry.virtualPath}${MINSYNC_VISUAL_SUFFIX}`);
			}
		}

		let imageStaged = false;
		const imageStamp =
			options.stageRawImages === true && isStandaloneImagePath(entry.virtualPath)
				? fileStamp(entry.sourcePath)
				: undefined;
		if (imageStamp !== undefined) {
			const imageMinSyncPath = minSyncImageDocumentPath(workspacePath, entry.virtualPath);
			try {
				const imageUnchanged =
					previousEntry?.imageSourcePath === entry.sourcePath &&
					previousEntry.imageStamp === imageStamp &&
					isRegularFile(imageMinSyncPath);
				if (!imageUnchanged) {
					ensureManagedDirectory(filesRoot, dirname(imageMinSyncPath));
					copyFileAtomically(entry.sourcePath, imageMinSyncPath);
				}
				stagedEntries.push({
					virtualPath: entry.virtualPath,
					sourcePath: entry.sourcePath,
					parsedOutputPath: entry.parsedOutputPath,
					minSyncPath: imageMinSyncPath,
					sourceMtimeMs: entry.sourceMtimeMs,
					mirrorKind: "image",
				});
				desiredPaths.add(imageMinSyncPath);
				imageStaged = true;
			} catch (error) {
				if (!isPerDocumentStagingError(error)) throw error;
				stagingFailed.push(entry.virtualPath);
			}
		}

		currentState[entry.virtualPath] = {
			outputPath: mirrorEntry.outputPath,
			updatedAt: mirrorEntry.updatedAt,
			...(visualStaged && visualStamp !== undefined ? { visualOutputPath, visualStamp } : {}),
			...(imageStaged && imageStamp !== undefined ? { imageSourcePath: entry.sourcePath, imageStamp } : {}),
		};
	}

	// `desiredPaths` is driven by the whole mirror index, not by one parse batch,
	// so a caller that stages once per batch never drops a not-yet-reached
	// document: only entries the mirror no longer holds leave the staged tree.
	removeUnexpectedStagedFiles(filesRoot, desiredPaths);
	saveStagingState(workspacePath, { version: 1, entries: currentState });

	return { workspacePath, entries: stagedEntries, excluded, stagingFailed };
}

function stagingStatePath(workspacePath: string): string {
	return join(dirname(workspacePath), `.${basename(workspacePath)}-autorag-staging.json`);
}

function loadStagingState(workspacePath: string): MinSyncStagingState | undefined {
	const path = stagingStatePath(workspacePath);
	if (!isRegularFile(path)) return undefined;
	try {
		const parsed: unknown = JSON.parse(readFileSync(path, "utf8"));
		return isMinSyncStagingState(parsed) ? parsed : undefined;
	} catch {
		return undefined;
	}
}

function saveStagingState(workspacePath: string, state: MinSyncStagingState): void {
	const path = stagingStatePath(workspacePath);
	const temporaryPath = `${path}.${randomUUID()}.tmp`;
	writeFileSync(temporaryPath, `${JSON.stringify(state, null, 2)}\n`);
	renameWithRetry(temporaryPath, path);
}

function isMinSyncStagingState(value: unknown): value is MinSyncStagingState {
	if (!isRecord(value) || value.version !== 1 || !isRecord(value.entries)) return false;
	return Object.entries(value.entries).every(
		([virtualPath, entry]) =>
			normalizeVirtualPath(virtualPath) === virtualPath &&
			isRecord(entry) &&
			typeof entry.outputPath === "string" &&
			typeof entry.updatedAt === "string",
	);
}

function ensureManagedFilesRoot(filesRoot: string, rebuild: boolean): void {
	const status = lstatSync(filesRoot, { throwIfNoEntry: false });
	if (status !== undefined && (rebuild || !status.isDirectory())) {
		rmSync(filesRoot, { recursive: status.isDirectory(), force: true });
	}
	mkdirSync(filesRoot, { recursive: true });
}

function ensureManagedDirectory(filesRoot: string, directory: string): void {
	const relativeDirectory = relative(filesRoot, directory);
	if (relativeDirectory === ".." || relativeDirectory.startsWith(`..${sep}`) || isAbsolute(relativeDirectory)) {
		throw new Error("MinSync staging path escaped its managed files root");
	}
	const segments = relativeDirectory.split(sep).filter(Boolean);
	let current = filesRoot;
	for (const segment of segments) {
		current = join(current, segment);
		const status = lstatSync(current, { throwIfNoEntry: false });
		if (status?.isDirectory()) continue;
		if (status !== undefined) {
			rmSync(current, { force: true });
		}
		mkdirSync(current);
	}
}

function copyFileAtomically(source: string, destination: string): void {
	const temporaryPath = join(dirname(destination), `.${randomUUID()}.tmp`);
	try {
		copyFileSync(source, temporaryPath);
		renameWithRetry(temporaryPath, destination);
	} catch (error) {
		try {
			rmSync(temporaryPath, { force: true });
		} catch {
			// Never let cleanup mask the copy/rename error (e.g. ENAMETOOLONG on lstat).
		}
		throw error;
	}
}

function isPerDocumentStagingError(error: unknown): boolean {
	if (!isRecord(error) || typeof error.code !== "string") return false;
	return error.code === "ENAMETOOLONG" || error.code === "EINVAL";
}

function isRegularFile(path: string): boolean {
	return lstatSync(path, { throwIfNoEntry: false })?.isFile() === true;
}

function removeUnexpectedStagedFiles(directory: string, desiredPaths: ReadonlySet<string>): void {
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) {
			removeUnexpectedStagedFiles(path, desiredPaths);
		} else if (!desiredPaths.has(path)) {
			rmSync(path, { force: true });
		}
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}

export function buildMinSyncPathMap(
	root: string,
	workspacePath: string,
	minSyncBinaryPath?: string,
): ReadonlyMap<string, MinSyncWorkspaceEntry> {
	const index = loadMirrorIndex(root, { minSyncBinaryPath });
	const byPath = new Map<string, MinSyncWorkspaceEntry>();
	for (const entry of Object.values(index.entries)) {
		const minSyncPath = minSyncDocumentPath(workspacePath, entry.virtualPath);
		const mapped: MinSyncWorkspaceEntry = {
			virtualPath: entry.virtualPath,
			sourcePath: entry.sourcePath,
			parsedOutputPath: entry.outputPath,
			minSyncPath,
			sourceMtimeMs: Math.floor(entry.sourceMtimeNs / NS_PER_MS),
			mirrorKind: "text",
		};
		byPath.set(entry.outputPath, mapped);
		byPath.set(minSyncPath, mapped);
		byPath.set(`files/${entry.virtualPath.replace(/^\/+/, "")}.md`, mapped);
		byPath.set(legacyParsedVirtualPath(root, entry.virtualPath), mapped);

		const visualOutputPath = parsedVisualOutputPath(root, entry.virtualPath);
		const visualMinSyncPath = minSyncVisualDocumentPath(workspacePath, entry.virtualPath);
		if (existsSync(visualOutputPath) || existsSync(visualMinSyncPath)) {
			const mappedVisual: MinSyncWorkspaceEntry = {
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parsedOutputPath: visualOutputPath,
				minSyncPath: visualMinSyncPath,
				sourceMtimeMs: mapped.sourceMtimeMs,
				mirrorKind: "visual",
			};
			byPath.set(visualOutputPath, mappedVisual);
			byPath.set(visualMinSyncPath, mappedVisual);
			byPath.set(`files/${entry.virtualPath.replace(/^\/+/, "")}${MINSYNC_VISUAL_SUFFIX}`, mappedVisual);
		}

		if (isStandaloneImagePath(entry.virtualPath)) {
			const imageMinSyncPath = minSyncImageDocumentPath(workspacePath, entry.virtualPath);
			const mappedImage: MinSyncWorkspaceEntry = {
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parsedOutputPath: entry.outputPath,
				minSyncPath: imageMinSyncPath,
				sourceMtimeMs: mapped.sourceMtimeMs,
				mirrorKind: "image",
			};
			byPath.set(imageMinSyncPath, mappedImage);
			byPath.set(`files/${entry.virtualPath.replace(/^\/+/, "")}`, mappedImage);
		}
	}
	return byPath;
}

function legacyParsedVirtualPath(root: string, virtualPath: string): string {
	return join(root, ".autorag", "parsed", "files", `${virtualPath.replace(/^\/+/, "")}.md`);
}
