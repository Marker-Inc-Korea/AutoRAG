import { createHash, randomUUID } from "node:crypto";
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
import { loadMirrorIndex, type ParsedMirrorIndex } from "../mirror/index-store.ts";
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
	readonly fingerprint: string;
	readonly changed: boolean;
}

interface MinSyncStagingEntry {
	readonly outputPath: string;
	readonly updatedAt: string;
	readonly contentSha256?: string;
}

interface MinSyncStagingState {
	readonly version: 1;
	readonly fingerprint?: string;
	readonly entries: Readonly<Record<string, MinSyncStagingEntry>>;
}

export function syncMinSyncWorkspace(
	root: string,
	options: { readonly workspacePath?: string; readonly configurationFingerprint?: string } = {},
): MinSyncWorkspaceSyncResult {
	const workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(root);
	const filesRoot = join(workspacePath, "files");
	const index = loadMirrorIndex(root);
	const fingerprint = minSyncMirrorFingerprint(index, options.configurationFingerprint);
	const previousState = loadStagingState(workspacePath);
	let changed = previousState?.fingerprint !== fingerprint;
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
	ensureManagedFilesRoot(filesRoot, previousState === undefined);

	const currentState: Record<string, MinSyncStagingEntry> = {};
	const desiredPaths = new Set<string>();
	for (const entry of entries) {
		const mirrorEntry = index.entries[entry.virtualPath];
		if (mirrorEntry === undefined) continue;
		const previousEntry = previousState?.entries[entry.virtualPath];
		const unchanged =
			previousEntry?.outputPath === mirrorEntry.outputPath &&
			(mirrorEntry.contentSha256 === undefined
				? previousEntry.updatedAt === mirrorEntry.updatedAt
				: previousEntry.contentSha256 === mirrorEntry.contentSha256) &&
			isRegularFile(entry.minSyncPath);
		if (!unchanged) {
			ensureManagedDirectory(filesRoot, dirname(entry.minSyncPath));
			copyFileAtomically(entry.parsedOutputPath, entry.minSyncPath);
			changed = true;
		}
		currentState[entry.virtualPath] = {
			outputPath: mirrorEntry.outputPath,
			updatedAt: mirrorEntry.updatedAt,
			...(mirrorEntry.contentSha256 === undefined ? {} : { contentSha256: mirrorEntry.contentSha256 }),
		};
		desiredPaths.add(entry.minSyncPath);
	}

	changed = removeUnexpectedStagedFiles(filesRoot, desiredPaths) || changed;
	if (changed || previousState === undefined) {
		saveStagingState(workspacePath, { version: 1, fingerprint, entries: currentState });
	}

	return { workspacePath, entries, fingerprint, changed };
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

export function minSyncMirrorFingerprint(index: ParsedMirrorIndex, configurationFingerprint?: string): string {
	const entries = Object.values(index.entries)
		.map(
			(entry) =>
				`${entry.virtualPath}\u0000${entry.sourcePath}\u0000${entry.outputPath}\u0000${entry.parserName}\u0000${entry.contentSha256 ?? "-"}`,
		)
		.sort();
	const material = [`config:${configurationFingerprint ?? "-"}`, `entries:${entries.length}`, ...entries].join("\n");
	return createHash("sha256").update(material, "utf8").digest("hex");
}

function saveStagingState(workspacePath: string, state: MinSyncStagingState): void {
	const path = stagingStatePath(workspacePath);
	const temporaryPath = `${path}.${randomUUID()}.tmp`;
	writeFileSync(temporaryPath, `${JSON.stringify(state, null, 2)}\n`);
	renameSync(temporaryPath, path);
}

function isMinSyncStagingState(value: unknown): value is MinSyncStagingState {
	if (
		!isRecord(value) ||
		value.version !== 1 ||
		(value.fingerprint !== undefined && typeof value.fingerprint !== "string") ||
		!isRecord(value.entries)
	) {
		return false;
	}
	return Object.entries(value.entries).every(
		([virtualPath, entry]) =>
			normalizeVirtualPath(virtualPath) === virtualPath &&
			isRecord(entry) &&
			typeof entry.outputPath === "string" &&
			typeof entry.updatedAt === "string" &&
			(entry.contentSha256 === undefined || typeof entry.contentSha256 === "string"),
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

function removeUnexpectedStagedFiles(directory: string, desiredPaths: ReadonlySet<string>): boolean {
	let changed = false;
	for (const entry of readdirSync(directory, { withFileTypes: true })) {
		const path = join(directory, entry.name);
		if (entry.isDirectory()) {
			changed = removeUnexpectedStagedFiles(path, desiredPaths) || changed;
		} else if (!desiredPaths.has(path)) {
			rmSync(path, { force: true });
			changed = true;
		}
	}
	return changed;
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
