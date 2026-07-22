import { copyFileSync, existsSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
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

export function syncMinSyncWorkspace(
	root: string,
	options: { readonly workspacePath?: string } = {},
): MinSyncWorkspaceSyncResult {
	const workspacePath = options.workspacePath ?? minSyncWorkspaceRoot(root);
	const filesRoot = join(workspacePath, "files");
	rmSync(filesRoot, { recursive: true, force: true });
	mkdirSync(filesRoot, { recursive: true });

	const index = loadMirrorIndex(root);
	const entries = Object.values(index.entries)
		.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath))
		.filter((entry) => existsSync(entry.outputPath))
		.map((entry) => {
			const minSyncPath = minSyncDocumentPath(workspacePath, entry.virtualPath);
			mkdirSync(dirname(minSyncPath), { recursive: true });
			copyFileSync(entry.outputPath, minSyncPath);
			return {
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parsedOutputPath: entry.outputPath,
				minSyncPath,
			};
		});

	return { workspacePath, entries };
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
