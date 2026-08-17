import { randomUUID } from "node:crypto";
import {
	copyFileSync,
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { basename, dirname, join } from "node:path";
import { normalizeVirtualPath } from "../filesystem/source-paths.ts";
import { loadMirrorIndex } from "../mirror/index-store.ts";
import { minSyncDocumentPath, minSyncWorkspaceRoot } from "./paths.ts";

export interface MinSyncWorkspaceEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly parsedOutputPath: string;
	readonly minSyncPath: string;
}

export interface MinSyncWorkspaceSyncResult {
	readonly workspacePath: string;
	readonly entries: readonly MinSyncWorkspaceEntry[];
}

interface MinSyncStagingEntry {
	readonly outputPath: string;
	readonly updatedAt: string;
}

interface MinSyncStagingState {
	readonly version: 1;
	readonly entries: Readonly<Record<string, MinSyncStagingEntry>>;
}

export function syncMinSyncWorkspace(
	root: string,
	options: { readonly workspacePath?: string } = {},
): MinSyncWorkspaceSyncResult {
	const workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(root);
	const filesRoot = join(workspacePath, "files");
	const index = loadMirrorIndex(root);
	const entries = Object.values(index.entries)
		.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath))
		.filter((entry) => existsSync(entry.outputPath))
		.map((entry) => {
			const minSyncPath = minSyncDocumentPath(workspacePath, entry.virtualPath);
			return {
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parsedOutputPath: entry.outputPath,
				minSyncPath,
			};
		});
	const previousState = loadStagingState(workspacePath);
	if (previousState === undefined) {
		rmSync(filesRoot, { recursive: true, force: true });
	}
	mkdirSync(filesRoot, { recursive: true });

	const currentState: Record<string, MinSyncStagingEntry> = {};
	const desiredPaths = new Set<string>();
	for (const entry of entries) {
		const mirrorEntry = index.entries[entry.virtualPath];
		if (mirrorEntry === undefined) continue;
		const previousEntry = previousState?.entries[entry.virtualPath];
		const unchanged =
			previousEntry?.outputPath === mirrorEntry.outputPath &&
			previousEntry.updatedAt === mirrorEntry.updatedAt &&
			existsSync(entry.minSyncPath);
		if (!unchanged) {
			mkdirSync(dirname(entry.minSyncPath), { recursive: true });
			copyFileSync(entry.parsedOutputPath, entry.minSyncPath);
		}
		currentState[entry.virtualPath] = {
			outputPath: mirrorEntry.outputPath,
			updatedAt: mirrorEntry.updatedAt,
		};
		desiredPaths.add(entry.minSyncPath);
	}

	removeUnexpectedStagedFiles(filesRoot, desiredPaths);
	saveStagingState(workspacePath, { version: 1, entries: currentState });

	return { workspacePath, entries };
}

function stagingStatePath(workspacePath: string): string {
	return join(dirname(workspacePath), `.${basename(workspacePath)}-autorag-staging.json`);
}

function loadStagingState(workspacePath: string): MinSyncStagingState | undefined {
	const path = stagingStatePath(workspacePath);
	if (!existsSync(path)) return undefined;
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
	renameSync(temporaryPath, path);
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

export function buildMinSyncPathMap(root: string, workspacePath: string): ReadonlyMap<string, MinSyncWorkspaceEntry> {
	const index = loadMirrorIndex(root);
	const byPath = new Map<string, MinSyncWorkspaceEntry>();
	for (const entry of Object.values(index.entries)) {
		const minSyncPath = minSyncDocumentPath(workspacePath, entry.virtualPath);
		const mapped = {
			virtualPath: entry.virtualPath,
			sourcePath: entry.sourcePath,
			parsedOutputPath: entry.outputPath,
			minSyncPath,
		};
		byPath.set(entry.outputPath, mapped);
		byPath.set(minSyncPath, mapped);
		byPath.set(`files/${entry.virtualPath.replace(/^\/+/, "")}.md`, mapped);
		byPath.set(legacyParsedVirtualPath(root, entry.virtualPath), mapped);
	}
	return byPath;
}

function legacyParsedVirtualPath(root: string, virtualPath: string): string {
	return join(root, ".autorag", "parsed", "files", `${virtualPath.replace(/^\/+/, "")}.md`);
}
