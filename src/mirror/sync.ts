import { type Dir, existsSync, rmSync } from "node:fs";
import { opendir, readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import { createDefaultParserRegistry, type DefaultParserRegistryOptions } from "../parser/defaults.ts";
import { ParseError } from "../parser/errors.ts";
import { KORDOC_IMAGE_EXTENSIONS } from "../parser/kordoc.ts";
import { ocrImageTooSmallMessage } from "../parser/ocr-engines.ts";
import type { ParserRegistry } from "../parser/registry.ts";
import { normalizeMarkdown } from "../parser/text.ts";
import type { ParseOutput } from "../parser/types.ts";
import { writeFileAtomic } from "./atomic-write.ts";
import {
	loadMirrorIndex,
	type ParsedMirrorEntry,
	type ParsedMirrorIndex,
	type ParsedMirrorSkipEntry,
	type ParsedMirrorSkipReason,
	saveMirrorIndex,
} from "./index-store.ts";
import { parsedMirrorIndexPath, parsedOutputPath } from "./paths.ts";

/** Refuse to fully load/parse sources larger than this. Keeps Node heap stable on home folders. */
export const DEFAULT_MAX_SOURCE_BYTES = 50 * 1024 * 1024;

/** Persist partial mirror progress this often so an OOM mid-run still restarts incrementally. */
const MIRROR_CHECKPOINT_EVERY = 25;

/**
 * Files AutoRAG's own tooling writes into a search root. They are product output,
 * not corpus sources: Jikji rewrites its agent map after the parsed phase, so
 * indexing it made an otherwise untouched corpus report as stale forever.
 */
const PRODUCT_ARTIFACT_FILE_NAMES = new Set([".jikji_agent_map.md"]);

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
	/** Absolute source paths omitted from parsed mirrors (for example exact duplicate copies). */
	readonly excludeSourcePaths?: ReadonlySet<string>;
	/** Absolute source paths or directories omitted by the user. */
	readonly userExcludedSourcePaths?: ReadonlySet<string>;
	/**
	 * Retries for a transient I/O error (`EMFILE`, `EBUSY`, `EIO`, `ETIMEDOUT`) on one
	 * file's read or mirror write before the file is recorded as skipped. Default 3.
	 */
	readonly transientRetries?: number;
	/** Base delay between transient retries; doubles per retry. Default 50ms. */
	readonly transientRetryBackoffMs?: number;
}

export type ParsedMirrorDiagnosticCode =
	| "unsupported-file"
	| "parser-skipped"
	| "parser-failed"
	| "parser-crashed"
	| "source-vanished"
	| "source-unreadable"
	| "source-read-failed"
	| "mirror-write-failed"
	| "duplicate-excluded"
	| "deleted-mirror"
	| "stale-index"
	| "pdf-extract-thin"
	| "parser-warning";

/**
 * Refresh diagnostic. `source` is an opaque virtual path, never a real filesystem path.
 * `message` keeps the underlying error text, which may include a filesystem path.
 */
export interface ParsedMirrorDiagnostic {
	readonly code: ParsedMirrorDiagnosticCode;
	readonly severity: "info" | "warning";
	readonly message: string;
	readonly source: string;
	readonly reason?: string;
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

const PARSER_FAILURE_MESSAGE = "The registered parser failed on this file; it was skipped during indexing.";
const PARSER_CRASH_MESSAGE = "The registered parser crashed unexpectedly on this file; it was skipped during indexing.";

/** Default retries for a transient per-file I/O error. Mirrors the `refresh.transientRetries` config default. */
export const DEFAULT_TRANSIENT_RETRIES = 3;
const DEFAULT_TRANSIENT_BACKOFF_MS = 50;

/** Errors that usually clear on their own: descriptor exhaustion, busy files, flaky disks and mounts. */
const TRANSIENT_ERROR_CODES: Record<string, true> = { EMFILE: true, EBUSY: true, EIO: true, ETIMEDOUT: true };

/** Write errors that mean the workspace itself is unusable, not that one file is bad. */
const WORKSPACE_ERROR_CODES: Record<string, true> = {
	ENOSPC: true,
	EDQUOT: true,
	EACCES: true,
	EPERM: true,
	EROFS: true,
};

/**
 * A failure that makes continuing the refresh pointless: the workspace volume is full or the
 * workspace is not writable. Per-file failures never use this class.
 */
export class MirrorWorkspaceError extends Error {
	readonly code: string | undefined;

	constructor(workspace: string, cause: unknown) {
		const code = errorCode(cause);
		const detail = cause instanceof Error ? cause.message : String(cause);
		super(
			`Refresh aborted: workspace ${workspace} is not writable${code === undefined ? "" : ` (${code})`}: ${detail}`,
		);
		this.name = "MirrorWorkspaceError";
		this.code = code;
		this.cause = cause;
	}
}

function errorCode(error: unknown): string | undefined {
	if (!(error instanceof Error)) return undefined;
	const code = (error as NodeJS.ErrnoException).code;
	return typeof code === "string" ? code : undefined;
}

/** Run `operation`, retrying transient I/O errors up to `retries` times with doubling backoff. */
async function withTransientRetry<T>(operation: () => T | Promise<T>, retries: number, backoffMs: number): Promise<T> {
	for (let attempt = 0; ; attempt += 1) {
		try {
			return await operation();
		} catch (error) {
			const code = errorCode(error);
			if (attempt >= retries || code === undefined || !Object.hasOwn(TRANSIENT_ERROR_CODES, code)) throw error;
			if (backoffMs > 0) await sleep(backoffMs * 2 ** attempt);
		}
	}
}

const STANDALONE_OCR_IMAGE_EXTENSIONS: ReadonlySet<string> = new Set(KORDOC_IMAGE_EXTENSIONS);

export async function syncParsedMirrors(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorSyncResult> {
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	const maxSourceBytes = options.maxSourceBytes ?? DEFAULT_MAX_SOURCE_BYTES;
	const supportedExtensions = supportedExtensionSet(registry);
	const current = await listCurrentFiles(options.searchPaths, supportedExtensions);
	const previous = loadMirrorIndex(options.root);
	const nextEntries: Record<string, ParsedMirrorEntry> = {};
	const nextSkipped: Record<string, ParsedMirrorSkipEntry> = {};
	const handledPrevious = new Set<string>();
	let written = 0;
	let skipped = 0;
	let deleted = 0;
	let sinceCheckpoint = 0;
	const transientRetries = Math.max(0, options.transientRetries ?? DEFAULT_TRANSIENT_RETRIES);
	const transientBackoffMs = options.transientRetryBackoffMs ?? DEFAULT_TRANSIENT_BACKOFF_MS;
	const diagnostics: ParsedMirrorDiagnostic[] = [];

	const recordSkip = (entry: CurrentEntry, reason: ParsedMirrorSkipReason): void => {
		nextSkipped[entry.virtualPath] = {
			virtualPath: entry.virtualPath,
			sourcePath: entry.sourcePath,
			reason,
			sourceMtimeNs: entry.mtimeNs,
			sourceSizeBytes: entry.sizeBytes,
			updatedAt: new Date().toISOString(),
		};
	};

	const checkpoint = (): void => {
		// Progressive index: processed files + still-valid previous entries for unprocessed ones so
		// a crash mid-run does not force a full re-parse of already written mirrors.
		const merged: Record<string, ParsedMirrorEntry> = { ...previous.entries, ...nextEntries };
		const mergedSkipped: Record<string, ParsedMirrorSkipEntry> = { ...previous.skipped, ...nextSkipped };
		// Drop previous records for paths this pass already removed or replaced. A handled path keeps
		// a skip only when this pass recorded one; a transient read failure must not retain an old skip.
		for (const virtualPath of handledPrevious) {
			if (virtualPath in nextEntries) {
				delete mergedSkipped[virtualPath];
				continue;
			}
			delete merged[virtualPath];
			if (!(virtualPath in nextSkipped)) delete mergedSkipped[virtualPath];
		}
		saveMirrorIndex(options.root, { version: 1, entries: merged, skipped: mergedSkipped });
		sinceCheckpoint = 0;
	};

	// Isolate one file's failure: drop its stale mirror, count the skip, surface one diagnostic, and
	// keep going. A `skipReason` persists the skip so an unchanged file is not retried; read and
	// write failures pass none, so a later refresh tries the file again.
	const skipFile = (
		entry: CurrentEntry,
		diagnostic: Omit<ParsedMirrorDiagnostic, "source">,
		skipReason?: ParsedMirrorSkipReason,
	): void => {
		deleted += removePrevious(options.root, previous, entry.virtualPath);
		handledPrevious.add(entry.virtualPath);
		skipped += 1;
		if (skipReason !== undefined) recordSkip(entry, skipReason);
		diagnostics.push({ ...diagnostic, source: entry.virtualPath });
		sinceCheckpoint += 1;
		if (sinceCheckpoint >= MIRROR_CHECKPOINT_EVERY) checkpoint();
	};

	for (const entry of current) {
		if (isPathExcluded(entry.sourcePath, options.userExcludedSourcePaths)) {
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			recordSkip(entry, "user-excluded");
			continue;
		}
		if (options.excludeSourcePaths?.has(entry.sourcePath)) {
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			recordSkip(entry, "duplicate-excluded");
			diagnostics.push({
				code: "duplicate-excluded",
				severity: "info",
				message: "An older exact duplicate was excluded from indexing; the newest copy remains indexed.",
				source: entry.virtualPath,
			});
			continue;
		}
		const parser = registry.getForVirtualPath(entry.virtualPath);
		if (!parser) {
			// Collect only filters to known extensions, so this is rare (double-check safety).
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			recordSkip(entry, "parser-skipped");
			continue;
		}

		if (entry.sizeBytes > maxSourceBytes) {
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			recordSkip(entry, "parser-skipped");
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
			// Every failure below is isolated to this one file. Only a workspace-level write failure
			// (full volume, unwritable workspace) aborts the refresh.
			let bytes: Awaited<ReturnType<typeof readFile>>;
			try {
				bytes = await withTransientRetry(() => readFile(entry.sourcePath), transientRetries, transientBackoffMs);
			} catch (error) {
				skipFile(entry, sourceReadFailure(error));
				continue;
			}

			// A page image inside a PDF stays on the provider. Only a standalone image below leptonica's
			// minimum is a per-file skip, recorded the same way as the max-size gate.
			const tooSmallImage = standaloneImageTooSmallMessage(entry.virtualPath, bytes);
			if (tooSmallImage !== undefined) {
				skipFile(entry, { code: "parser-skipped", severity: "warning", message: tooSmallImage }, "parser-skipped");
				continue;
			}

			let parsed: ParseOutput;
			try {
				// Prefer sourcePath streaming into parsers; still pass bytes for parsers that need them,
				// but only after the size gate above.
				parsed = await parser.parse({ virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, bytes });
			} catch (error) {
				if (error instanceof ParseError) {
					skipFile(
						entry,
						{ code: "parser-failed", severity: "warning", message: PARSER_FAILURE_MESSAGE },
						"parser-failed",
					);
				} else {
					// A parser bug or a third-party plain Error. The message is withheld: it may name a path.
					skipFile(entry, {
						code: "parser-crashed",
						severity: "warning",
						message: PARSER_CRASH_MESSAGE,
						...(error instanceof Error ? { reason: error.name } : {}),
					});
				}
				continue;
			}
			try {
				const markdown = normalizeMarkdown(parsed.markdown);
				await withTransientRetry(() => writeFileAtomic(outputPath, markdown), transientRetries, transientBackoffMs);
			} catch (error) {
				const code = errorCode(error);
				if (code !== undefined && Object.hasOwn(WORKSPACE_ERROR_CODES, code)) {
					// Keep what is already mirrored so the next run restarts incrementally.
					try {
						checkpoint();
					} catch {
						// The index shares the failing volume; the write error below is the root cause.
					}
					throw new MirrorWorkspaceError(options.root, error);
				}
				skipFile(entry, {
					code: "mirror-write-failed",
					severity: "warning",
					message: error instanceof Error ? error.message : String(error),
					...(code !== undefined ? { reason: code } : {}),
				});
				continue;
			}
			for (const diagnostic of parsed.diagnostics ?? []) {
				diagnostics.push({
					code: diagnostic.code,
					severity: diagnostic.severity,
					message: diagnostic.message,
					source: entry.virtualPath,
				});
			}
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

	const index: ParsedMirrorIndex = { version: 1, entries: nextEntries, skipped: nextSkipped };
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
 * size only — no parsing) against the mirror index's own records. A source is
 * current when the last refresh either indexed it or deliberately skipped it at
 * this exact version, so a skipped duplicate, oversize, or unparseable file is not
 * reported as a new source. Returns a `stale-index` diagnostic (opaque virtual
 * path) for every supported source that is new or changed since the last
 * successful refresh. Safe to call outside the query hot path (e.g. from
 * getRefreshStatus).
 */
export async function detectMirrorStaleness(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorDiagnostic[]> {
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	const supportedExtensions = supportedExtensionSet(registry);
	const current = await listCurrentFiles(options.searchPaths, supportedExtensions);
	const previous = loadMirrorIndex(options.root);
	const diagnostics: ParsedMirrorDiagnostic[] = [];
	for (const entry of current) {
		if (
			options.excludeSourcePaths?.has(entry.sourcePath) ||
			isPathExcluded(entry.sourcePath, options.userExcludedSourcePaths)
		)
			continue;
		if (!registry.getForVirtualPath(entry.virtualPath)) continue;
		const prev = previous.entries[entry.virtualPath];
		if (prev !== undefined && prev.sourceMtimeNs === entry.mtimeNs && prev.sourceSizeBytes === entry.sizeBytes)
			continue;
		const prevSkip = previous.skipped?.[entry.virtualPath];
		// A "user-excluded" skip only suppresses staleness while the path stays excluded;
		// reaching here means it was re-included, so the mirror must be rebuilt and the
		// source counts as newly added (no mirror entry exists for it).
		const honoredSkip = prevSkip?.reason === "user-excluded" ? undefined : prevSkip;
		if (
			honoredSkip !== undefined &&
			honoredSkip.sourceMtimeNs === entry.mtimeNs &&
			honoredSkip.sourceSizeBytes === entry.sizeBytes
		)
			continue;
		const known = prev ?? honoredSkip;
		const reason =
			known === undefined
				? "source-added"
				: known.sourceMtimeNs !== entry.mtimeNs && known.sourceSizeBytes !== entry.sizeBytes
					? "mtime-and-size-changed"
					: known.sourceMtimeNs !== entry.mtimeNs
						? "mtime-changed"
						: "size-changed";
		diagnostics.push({
			code: "stale-index",
			severity: "warning",
			message: `Source ${entry.virtualPath} is ${reason.replaceAll("-", " ")} since the last refresh; indexes may be stale.`,
			source: entry.virtualPath,
			reason,
		});
	}
	return diagnostics;
}

function standaloneImageTooSmallMessage(virtualPath: string, bytes: Uint8Array): string | undefined {
	if (!STANDALONE_OCR_IMAGE_EXTENSIONS.has(normalizeExtension(extname(virtualPath)))) return undefined;
	return ocrImageTooSmallMessage(bytes);
}

/** Map any per-file read error to a diagnostic; unknown errors become `source-read-failed`. */
function sourceReadFailure(error: unknown): Omit<ParsedMirrorDiagnostic, "source"> {
	const code = errorCode(error);
	const message = error instanceof Error ? error.message : String(error);
	if (code === "ENOENT" || code === "ENOTDIR" || code === "EISDIR") {
		return { code: "source-vanished", severity: "info", message };
	}
	if (code === "EACCES" || code === "EPERM") {
		return { code: "source-unreadable", severity: "warning", message };
	}
	return { code: "source-read-failed", severity: "warning", message, ...(code !== undefined ? { reason: code } : {}) };
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

export function isPathExcluded(sourcePath: string, excluded: ReadonlySet<string> | undefined): boolean {
	if (excluded === undefined) return false;
	for (const excludedPath of excluded) {
		if (sourcePath === excludedPath || sourcePath.startsWith(`${excludedPath}${sep}`)) return true;
	}
	return false;
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
		if (PRODUCT_ARTIFACT_FILE_NAMES.has(entry.name)) continue;
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

function removeFile(path: string): void {
	if (existsSync(path)) rmSync(path, { force: true });
}
