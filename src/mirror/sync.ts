import { randomUUID } from "node:crypto";
import { type Dir, existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import { opendir, readFile, stat } from "node:fs/promises";
import { dirname, extname, resolve } from "node:path";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import { createDefaultParserRegistry, type DefaultParserRegistryOptions } from "../parser/defaults.ts";
import { ParseError } from "../parser/errors.ts";
import type { ParserRegistry } from "../parser/registry.ts";
import { normalizeMarkdown } from "../parser/text.ts";
import type { ParseOutput } from "../parser/types.ts";
import { loadMirrorIndex, type ParsedMirrorEntry, type ParsedMirrorIndex, saveMirrorIndex } from "./index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "./paths.ts";

/** Refuse to fully load/parse sources larger than this. Keeps Node heap stable on home folders. */
export const DEFAULT_MAX_SOURCE_BYTES = 50 * 1024 * 1024;

/** Persist partial mirror progress this often so an OOM mid-run still restarts incrementally. */
const MIRROR_CHECKPOINT_EVERY = 25;

const SKIP_DIR_NAMES = new Set([
	".autorag",
	".git",
	".jikji",
	".cache",
	".venv",
	"__pycache__",
	"node_modules",
	"dist",
	"build",
	"target",
	".Trash",
	".Spotlight-V100",
	".DocumentRevisions-V100",
	".fseventsd",
	"CacheStorage",
	"Service Worker",
	"Code Cache",
	"GPUCache",
	"GrShaderCache",
	"ShaderCache",
]);

export interface ParsedMirrorSyncOptions {
	readonly root: string;
	readonly searchPaths: readonly string[];
	readonly registry?: ParserRegistry;
	readonly parserOptions?: DefaultParserRegistryOptions;
	readonly force?: boolean;
	/** Override max source size for parse attempts (bytes). */
	readonly maxSourceBytes?: number;
}

export type ParsedMirrorDiagnosticCode =
	| "unsupported-file"
	| "parser-skipped"
	| "parser-failed"
	| "deleted-mirror"
	| "stale-index";

/** Path-opaque refresh diagnostic. `source` is an opaque virtual path, never a real fs path. */
export interface ParsedMirrorDiagnostic {
	readonly code: ParsedMirrorDiagnosticCode;
	readonly severity: "info" | "warning";
	readonly message: string;
	readonly source: string;
}

export interface ParsedMirrorSyncResult {
	readonly scanned: number;
	readonly written: number;
	readonly deleted: number;
	readonly skipped: number;
	readonly indexPath: string;
	readonly diagnostics: readonly ParsedMirrorDiagnostic[];
}

interface CurrentEntry {
	readonly virtualPath: string;
	readonly sourcePath: string;
	readonly sizeBytes: number;
	readonly mtimeNs: number;
}

export async function syncParsedMirrors(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorSyncResult> {
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	const maxSourceBytes = options.maxSourceBytes ?? DEFAULT_MAX_SOURCE_BYTES;
	const supportedExtensions = supportedExtensionSet(registry);
	const current = await listCurrentFiles(options.searchPaths, supportedExtensions);
	const previous = loadMirrorIndex(options.root);
	const nextEntries: Record<string, ParsedMirrorEntry> = {};
	const handledPrevious = new Set<string>();
	let written = 0;
	let skipped = 0;
	let deleted = 0;
	let sinceCheckpoint = 0;
	const diagnostics: ParsedMirrorDiagnostic[] = [];

	const checkpoint = (): void => {
		// Progressive index: processed files + still-valid previous entries for unprocessed ones so
		// a crash mid-run does not force a full re-parse of already written mirrors.
		const merged: Record<string, ParsedMirrorEntry> = { ...previous.entries, ...nextEntries };
		// Drop previous entries for supported paths we already decided to remove/skip in this pass.
		for (const virtualPath of handledPrevious) {
			if (!(virtualPath in nextEntries)) delete merged[virtualPath];
		}
		saveMirrorIndex(options.root, { version: 1, entries: merged });
		sinceCheckpoint = 0;
	};

	for (const entry of current) {
		const parser = registry.getForVirtualPath(entry.virtualPath);
		if (!parser) {
			// Collect only filters to known extensions, so this is rare (double-check safety).
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			continue;
		}

		if (entry.sizeBytes > maxSourceBytes) {
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			diagnostics.push({
				code: "parser-skipped",
				severity: "warning",
				message: "Source exceeds the configured max parse size and was skipped during indexing.",
				source: entry.virtualPath,
			});
			sinceCheckpoint += 1;
			if (sinceCheckpoint >= MIRROR_CHECKPOINT_EVERY) checkpoint();
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
			let parsed: ParseOutput;
			try {
				// Prefer sourcePath streaming into parsers; still pass bytes for parsers that need them,
				// but only after the size gate above.
				const bytes = await readFile(entry.sourcePath);
				parsed = await parser.parse({ virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, bytes });
			} catch (error) {
				if (!(error instanceof ParseError)) throw error;
				deleted += removePrevious(options.root, previous, entry.virtualPath);
				handledPrevious.add(entry.virtualPath);
				skipped += 1;
				diagnostics.push({
					code: "parser-failed",
					severity: "warning",
					message: "The registered parser failed on this file; it was skipped during indexing.",
					source: entry.virtualPath,
				});
				sinceCheckpoint += 1;
				if (sinceCheckpoint >= MIRROR_CHECKPOINT_EVERY) checkpoint();
				continue;
			}
			writeAtomic(outputPath, normalizeMarkdown(parsed.markdown));
			written += 1;
			sinceCheckpoint += 1;
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
		handledPrevious.add(entry.virtualPath);

		if (sinceCheckpoint >= MIRROR_CHECKPOINT_EVERY) checkpoint();
	}

	for (const [virtualPath, entry] of Object.entries(previous.entries)) {
		if (handledPrevious.has(virtualPath)) continue;
		if (nextEntries[virtualPath]) continue;
		// Only delete previous index entries for files no longer present among current supported files.
		// Entries outside this run's supported collect set that disappeared should still be cleaned.
		removeFile(parsedOutputPath(options.root, entry.virtualPath));
		deleted += 1;
		diagnostics.push({
			code: "deleted-mirror",
			severity: "info",
			message: "A previously indexed document is gone; its parsed mirror was removed.",
			source: virtualPath,
		});
	}

	const index: ParsedMirrorIndex = { version: 1, entries: nextEntries };
	saveMirrorIndex(options.root, index);
	return {
		scanned: current.length,
		written,
		deleted,
		skipped,
		indexPath: parsedMirrorIndexPath(options.root),
		diagnostics,
	};
}

/**
 * Cheap, parse-free staleness check: compares current source files (by mtime and
 * size only — no parsing) against the recorded mirror index. Returns a
 * `stale-index` diagnostic (opaque virtual path) for every supported source that
 * is new or changed since the last successful refresh. Safe to call outside the
 * query hot path (e.g. from getRefreshStatus).
 */
export async function detectMirrorStaleness(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorDiagnostic[]> {
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	const supportedExtensions = supportedExtensionSet(registry);
	const current = await listCurrentFiles(options.searchPaths, supportedExtensions);
	const previous = loadMirrorIndex(options.root);
	const diagnostics: ParsedMirrorDiagnostic[] = [];
	for (const entry of current) {
		if (!registry.getForVirtualPath(entry.virtualPath)) continue;
		const prev = previous.entries[entry.virtualPath];
		if (!prev || prev.sourceMtimeNs !== entry.mtimeNs || prev.sourceSizeBytes !== entry.sizeBytes) {
			diagnostics.push({
				code: "stale-index",
				severity: "warning",
				message: "A source document has changed since the last refresh; indexes may be stale.",
				source: entry.virtualPath,
			});
		}
	}
	return diagnostics;
}

function supportedExtensionSet(registry: ParserRegistry): ReadonlySet<string> {
	const extensions = new Set<string>();
	for (const parser of registry.list()) {
		for (const extension of parser.extensions) {
			extensions.add(normalizeExtension(extension));
		}
	}
	return extensions;
}

function normalizeExtension(extension: string): string {
	const lower = extension.toLowerCase();
	return lower.startsWith(".") ? lower : `.${lower}`;
}

async function listCurrentFiles(
	searchPaths: readonly string[],
	supportedExtensions: ReadonlySet<string>,
): Promise<CurrentEntry[]> {
	const entries: CurrentEntry[] = [];
	for (const sourceRoot of planSourceRoots(searchPaths)) {
		await collectFiles(sourceRoot, sourceRoot.rootPath, entries, supportedExtensions);
	}
	entries.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath));
	return entries;
}

async function collectFiles(
	sourceRoot: SourceRoot,
	directory: string,
	entries: CurrentEntry[],
	supportedExtensions: ReadonlySet<string>,
): Promise<void> {
	let dir: Dir;
	try {
		dir = await opendir(directory);
	} catch {
		return;
	}
	for await (const entry of dir) {
		if (SKIP_DIR_NAMES.has(entry.name)) continue;
		const sourcePath = resolve(directory, entry.name);
		if (entry.isDirectory()) {
			await collectFiles(sourceRoot, sourcePath, entries, supportedExtensions);
			continue;
		}
		if (!entry.isFile()) continue;
		const extension = normalizeExtension(extname(entry.name));
		if (!supportedExtensions.has(extension)) continue;
		let fileStat: Awaited<ReturnType<typeof stat>>;
		try {
			fileStat = await stat(sourcePath, { bigint: true });
		} catch {
			continue;
		}
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
