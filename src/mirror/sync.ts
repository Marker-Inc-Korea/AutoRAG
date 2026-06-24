import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { opendir, readFile, stat } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import { createDefaultParserRegistry } from "../parser/defaults.ts";
import { ParseError } from "../parser/errors.ts";
import type { ParserRegistry } from "../parser/registry.ts";
import type { ParseOutput } from "../parser/types.ts";
import { loadMirrorIndex, type ParsedMirrorEntry, type ParsedMirrorIndex, saveMirrorIndex } from "./index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "./paths.ts";

export interface ParsedMirrorSyncOptions {
	readonly root: string;
	readonly searchPaths: readonly string[];
	readonly registry?: ParserRegistry;
	readonly force?: boolean;
}

export interface ParsedMirrorSyncResult {
	readonly scanned: number;
	readonly written: number;
	readonly deleted: number;
	readonly skipped: number;
	readonly indexPath: string;
}

interface CurrentEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly sizeBytes: number;
	readonly mtimeNs: number;
}

export async function syncParsedMirrors(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorSyncResult> {
	const registry = options.registry ?? createDefaultParserRegistry();
	const current = await listCurrentFiles(options.searchPaths);
	const previous = loadMirrorIndex(options.root);
	const nextEntries: Record<string, ParsedMirrorEntry> = {};
	const handledPrevious = new Set<string>();
	let written = 0;
	let skipped = 0;
	let deleted = 0;

	for (const entry of current) {
		const parser = registry.getForVirtualPath(entry.virtualPath);
		if (!parser) {
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			continue;
		}

		const previousEntry = previous.entries[entry.virtualPath];
		const outputPath = parsedOutputPath(options.root, entry.virtualPath);
		const unchanged =
			!options.force &&
			previousEntry?.sourceMtimeNs === entry.mtimeNs &&
			previousEntry.sourceSizeBytes === entry.sizeBytes &&
			previousEntry.parserName === parser.name &&
			existsSync(outputPath);

		if (!unchanged) {
			const bytes = await readFile(entry.sourcePath);
			let parsed: ParseOutput;
			try {
				parsed = await parser.parse({ virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, bytes });
			} catch (error) {
				if (!(error instanceof ParseError)) throw error;
				deleted += removePrevious(options.root, previous, entry.virtualPath);
				handledPrevious.add(entry.virtualPath);
				skipped += 1;
				continue;
			}
			writeAtomic(outputPath, parsed.markdown);
			written += 1;
		}

		nextEntries[entry.virtualPath] = {
			virtualPath: entry.virtualPath,
			sourcePath: entry.sourcePath,
			outputPath,
			parserName: parser.name,
			sourceMtimeNs: entry.mtimeNs,
			sourceSizeBytes: entry.sizeBytes,
			updatedAt: unchanged ? (previousEntry?.updatedAt ?? new Date().toISOString()) : new Date().toISOString(),
		};
	}

	for (const [virtualPath, entry] of Object.entries(previous.entries)) {
		if (handledPrevious.has(virtualPath)) continue;
		if (nextEntries[virtualPath]) continue;
		removeFile(parsedOutputPath(options.root, entry.virtualPath));
		deleted += 1;
	}

	const index: ParsedMirrorIndex = { version: 1, entries: nextEntries };
	saveMirrorIndex(options.root, index);
	return {
		scanned: current.length,
		written,
		deleted,
		skipped,
		indexPath: parsedMirrorIndexPath(options.root),
	};
}

async function listCurrentFiles(searchPaths: readonly string[]): Promise<CurrentEntry[]> {
	const entries: CurrentEntry[] = [];
	for (const sourceRoot of planSourceRoots(searchPaths)) {
		await collectFiles(sourceRoot, sourceRoot.rootPath, entries);
	}
	entries.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath));
	return entries;
}

async function collectFiles(sourceRoot: SourceRoot, directory: string, entries: CurrentEntry[]): Promise<void> {
	const dir = await opendir(directory);
	for await (const entry of dir) {
		if (entry.name === ".autorag" || entry.name === ".git" || entry.name === "node_modules") continue;
		const sourcePath = resolve(directory, entry.name);
		if (entry.isDirectory()) {
			await collectFiles(sourceRoot, sourcePath, entries);
			continue;
		}
		if (!entry.isFile()) continue;
		const fileStat = await stat(sourcePath, { bigint: true });
		entries.push({
			virtualPath: sourceIdentifier(sourceRoot, sourcePath),
			sourcePath,
			sizeBytes: Number(fileStat.size),
			mtimeNs: Number(fileStat.mtimeNs),
		});
	}
}

function removePrevious(root: string, index: ParsedMirrorIndex, virtualPath: string): number {
	const previous = index.entries[virtualPath];
	if (!previous) return 0;
	removeFile(parsedOutputPath(root, previous.virtualPath));
	return 1;
}

function writeAtomic(path: string, content: string): void {
	mkdirSync(dirname(path), { recursive: true });
	const tmp = `${path}.${randomUUID()}.tmp`;
	writeFileSync(tmp, content);
	renameSync(tmp, path);
}

function removeFile(path: string): void {
	if (existsSync(path)) rmSync(path, { force: true });
}
