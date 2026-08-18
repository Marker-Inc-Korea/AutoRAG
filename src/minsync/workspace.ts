import { randomUUID } from "node:crypto";
import {
	copyFileSync,
	existsSync,
	lstatSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	renameSync,
	rmSync,
	writeFileSync,
} from "node:fs";
import { basename, dirname, isAbsolute, join, relative, sep } from "node:path";
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
		.filter((entry) => normalizeVirtualPath(entry.virtualPath) === entry.virtualPath && existsSync(entry.outputPath))
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
	ensureManagedFilesRoot(filesRoot, previousState === undefined);

	const currentState: Record<string, MinSyncStagingEntry> = {};
	const desiredPaths = new Set<string>();
	for (const entry of entries) {
		const mirrorEntry = index.entries[entry.virtualPath];
		if (mirrorEntry === undefined) continue;
		const previousEntry = previousState?.entries[entry.virtualPath];
		const unchanged =
			previousEntry?.outputPath === mirrorEntry.outputPath &&
			previousEntry.updatedAt === mirrorEntry.updatedAt &&
			isRegularFile(entry.minSyncPath);
		if (!unchanged) {
			ensureManagedDirectory(filesRoot, dirname(entry.minSyncPath));
			copyFileAtomically(entry.parsedOutputPath, entry.minSyncPath);
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
	const temporaryPath = join(dirname(destination), `.${basename(destination)}.${randomUUID()}.tmp`);
	try {
		copyFileSync(source, temporaryPath);
		renameSync(temporaryPath, destination);
	} finally {
		rmSync(temporaryPath, { force: true });
	}
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
