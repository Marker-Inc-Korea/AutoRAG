import { type Dir, existsSync, rmSync } from "node:fs";
import { opendir, readFile, stat } from "node:fs/promises";
import { extname, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import { createDefaultParserRegistry, type DefaultParserRegistryOptions } from "../parser/defaults.ts";
import { ParseError, ParseIsolationError } from "../parser/errors.ts";
import {
	ISOLATED_PARSER_NAMES,
	IsolatedParseRunner,
	validateIsolationOptions,
	workerParserOptions,
} from "../parser/isolated.ts";
import { KORDOC_IMAGE_EXTENSIONS } from "../parser/kordoc.ts";
import { ocrImageTooSmallMessage } from "../parser/ocr-engines.ts";
import type { ParserRegistry } from "../parser/registry.ts";
import { normalizeMarkdown } from "../parser/text.ts";
import type { ParseOutput, Parser } from "../parser/types.ts";
import { abortReason } from "../process/abort-reason.ts";
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
export const MIRROR_CHECKPOINT_EVERY = 25;

/** A refresh may delete at most this share of the indexed documents (above {@link MASS_DELETE_FLOOR}). */
export const DEFAULT_MAX_DELETE_RATIO = 0.3;

/** Deletions at or below this many documents are never treated as a mass delete. */
const MASS_DELETE_FLOOR = 100;

/** Directory entries examined per search root before the walk stops. */
export const DEFAULT_MAX_SCANNED_ENTRIES = 2_000_000;

/** Deepest directory level (search root = 0) the walk descends into. */
export const DEFAULT_MAX_WALK_DEPTH = 64;

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
	 * Retry sources that previously failed to parse or read once, ignoring their skip records.
	 * Healthy, unchanged mirrors are still left alone (unlike `force`).
	 */
	readonly retrySkipped?: boolean;
	/**
	 * Stops the sync between files and kills an in-flight parse worker. The sync rejects with the signal's
	 * reason after checkpointing the mirror, so finished files are not re-parsed next time.
	 */
	readonly signal?: AbortSignal;
	/** Called before each file and once more with `processed === total` when the parse phase ends. */
	readonly onProgress?: (progress: ParsedMirrorProgress) => void;
	/**
	 * Largest share of previously indexed documents one refresh may delete because their source
	 * disappeared. A larger deletion (above 100 documents) is suppressed with `mass-delete-suppressed`
	 * unless `force` is set. Default {@link DEFAULT_MAX_DELETE_RATIO}.
	 */
	readonly maxDeleteRatio?: number;
	/** Directory entries examined per search root before the walk stops with `walk-truncated`. */
	readonly maxScannedEntries?: number;
	/** Deepest directory level the walk descends into (search root = 0), else `walk-truncated`. */
	readonly maxDepth?: number;
	/**
	 * Retries for a transient I/O error (`EMFILE`, `EBUSY`, `EIO`, `ETIMEDOUT`) on one
	 * file's read or mirror write before the file is recorded as skipped. Default 3.
	 */
	readonly transientRetries?: number;
	/** Base delay between transient retries; doubles per retry. Default 50ms. */
	readonly transientRetryBackoffMs?: number;
}

/** Parse-phase progress. Every scanned file counts as processed, whether parsed, unchanged, skipped or excluded. */
export interface ParsedMirrorProgress {
	readonly processed: number;
	readonly total: number;
	/** Running totals for the files processed so far; the final event carries the sync result's counts. */
	readonly written: number;
	readonly deleted: number;
	readonly skipped: number;
	/** The file about to be processed; absent on the final event. */
	readonly currentVirtualPath?: string;
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
	| "parser-warning"
	| "search-root-unavailable"
	| "mass-delete-suppressed"
	| "walk-truncated"
	| "directory-unreadable";

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

type RetryableSkipReason = "parser-failed" | "source-unreadable";

const RETRYABLE_SKIP_MESSAGES: Readonly<Record<RetryableSkipReason, string>> = {
	"parser-failed": PARSER_FAILURE_MESSAGE,
	"source-unreadable":
		"The source could not be read (permission denied); it is skipped until it changes or is retried.",
};

/**
 * Whether a recorded skip still stands for this source: same mtime, size, parser name and parser
 * version. Only retryable reasons qualify; the others are re-derived cheaply from config and stat
 * every run. A record without a parser stamp predates this policy and is retried once, which then
 * stamps it.
 */
function isSkipHonored(
	skip: ParsedMirrorSkipEntry,
	entry: CurrentEntry,
	parser: Parser,
	retryPass: boolean,
): skip is ParsedMirrorSkipEntry & { readonly reason: RetryableSkipReason } {
	if (retryPass) return false;
	if (skip.reason !== "parser-failed" && skip.reason !== "source-unreadable") return false;
	if (skip.sourceMtimeNs !== entry.mtimeNs || skip.sourceSizeBytes !== entry.sizeBytes) return false;
	return skip.parserName === parser.name && skip.parserVersion === parser.version;
}

export async function syncParsedMirrors(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorSyncResult> {
	if (options.parserOptions !== undefined) validateIsolationOptions(options.parserOptions);
	if (options.signal?.aborted) throw abortReason(options.signal);
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	// Worker isolation applies to the default registry only: an injected registry or a function-valued
	// parser option cannot cross the thread boundary, so those parse in-process.
	const workerOptions = options.registry === undefined ? workerParserOptions(options.parserOptions ?? {}) : undefined;
	const isolated =
		workerOptions === undefined
			? undefined
			: new IsolatedParseRunner({
					parserOptions: workerOptions,
					fileTimeoutMs: options.parserOptions?.fileTimeoutMs,
					maxHeapMb: options.parserOptions?.maxHeapMb,
				});
	try {
		return await syncWithRegistry(options, registry, isolated);
	} finally {
		await isolated?.close();
	}
}

async function syncWithRegistry(
	options: ParsedMirrorSyncOptions,
	registry: ParserRegistry,
	isolated: IsolatedParseRunner | undefined,
): Promise<ParsedMirrorSyncResult> {
	const maxSourceBytes = options.maxSourceBytes ?? DEFAULT_MAX_SOURCE_BYTES;
	const supportedExtensions = supportedExtensionSet(registry);
	const listing = await listCurrentFiles(options.searchPaths, supportedExtensions, walkLimits(options));
	const current = listing.entries;
	const previous = loadMirrorIndex(options.root);
	const nextEntries: Record<string, ParsedMirrorEntry> = {};
	const nextSkipped: Record<string, ParsedMirrorSkipEntry> = {};
	const handledPrevious = new Set<string>();
	let written = 0;
	let skipped = 0;
	let deleted = 0;
	let sinceCheckpoint = 0;
	const diagnostics: ParsedMirrorDiagnostic[] = [...listing.diagnostics];
	// Virtual path prefixes whose files the walk could not fully see. A previous entry under one of
	// them that is absent from `current` is unverifiable, not deleted, so it is kept as-is.
	const preservedPrefixes: string[] = [...listing.unreliablePrefixes];
	for (const unavailable of listing.unavailableRoots) {
		if (unavailable.reason === "empty") {
			// An empty root only matters when it used to hold documents; an explicit force clears it.
			if (options.force === true || !hasIndexedUnder(previous, unavailable.prefix)) continue;
		}
		preservedPrefixes.push(unavailable.prefix);
		diagnostics.push({
			code: "search-root-unavailable",
			severity: "warning",
			message: SEARCH_ROOT_UNAVAILABLE_MESSAGES[unavailable.reason],
			source: unavailable.prefix,
			reason: unavailable.reason,
		});
	}
	const transientRetries = Math.max(0, options.transientRetries ?? DEFAULT_TRANSIENT_RETRIES);
	const transientBackoffMs = options.transientRetryBackoffMs ?? DEFAULT_TRANSIENT_BACKOFF_MS;

	const retryPass = options.force === true || options.retrySkipped === true;

	/** `parser` stamps a retryable skip so a later refresh can honor it while the parser is unchanged. */
	const recordSkip = (entry: CurrentEntry, reason: ParsedMirrorSkipReason, parser?: Parser): void => {
		nextSkipped[entry.virtualPath] = {
			virtualPath: entry.virtualPath,
			sourcePath: entry.sourcePath,
			reason,
			sourceMtimeNs: entry.mtimeNs,
			sourceSizeBytes: entry.sizeBytes,
			updatedAt: new Date().toISOString(),
			...(parser === undefined ? {} : { parserName: parser.name, parserVersion: parser.version }),
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
	// keep going. A `skipReason` persists the skip so an unchanged file is not retried; `parser`
	// stamps a retryable one. Transient read and write failures pass none, so a later refresh
	// tries the file again.
	const skipFile = (
		entry: CurrentEntry,
		diagnostic: Omit<ParsedMirrorDiagnostic, "source">,
		skipReason?: ParsedMirrorSkipReason,
		parser?: Parser,
	): void => {
		deleted += removePrevious(options.root, previous, entry.virtualPath);
		handledPrevious.add(entry.virtualPath);
		skipped += 1;
		if (skipReason !== undefined) recordSkip(entry, skipReason, parser);
		diagnostics.push({ ...diagnostic, source: entry.virtualPath });
		sinceCheckpoint += 1;
		if (sinceCheckpoint >= MIRROR_CHECKPOINT_EVERY) checkpoint();
	};

	for (const [position, entry] of current.entries()) {
		options.onProgress?.({
			processed: position,
			total: current.length,
			written,
			deleted,
			skipped,
			currentVirtualPath: entry.virtualPath,
		});
		if (options.signal?.aborted) {
			checkpoint();
			throw abortReason(options.signal);
		}
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

		const previousSkip = previous.skipped?.[entry.virtualPath];
		if (previousSkip !== undefined && isSkipHonored(previousSkip, entry, parser, retryPass)) {
			// Same source and parser: keep the record and do not read or parse the file again.
			deleted += removePrevious(options.root, previous, entry.virtualPath);
			handledPrevious.add(entry.virtualPath);
			skipped += 1;
			nextSkipped[entry.virtualPath] = previousSkip;
			diagnostics.push({
				code: previousSkip.reason,
				severity: "warning",
				message: RETRYABLE_SKIP_MESSAGES[previousSkip.reason],
				source: entry.virtualPath,
			});
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
				// A vanished source has nothing to retry; an unreadable one is honored like a parser failure
				// so a permission problem is not re-hit on every refresh.
				const failure = sourceReadFailure(error);
				skipFile(entry, failure, failure.code === "source-unreadable" ? "source-unreadable" : undefined, parser);
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
				const input = { virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, bytes };
				parsed =
					isolated !== undefined && ISOLATED_PARSER_NAMES.has(parser.name)
						? await isolated.parse(parser.name, input, options.signal)
						: await parser.parse(input);
			} catch (error) {
				if (options.signal?.aborted) {
					checkpoint();
					throw error;
				}
				if (error instanceof ParseError) {
					skipFile(
						entry,
						{
							code: "parser-failed",
							severity: "warning",
							message:
								error.cause instanceof ParseIsolationError
									? `${PARSER_FAILURE_MESSAGE} ${error.cause.message}.`
									: PARSER_FAILURE_MESSAGE,
						},
						"parser-failed",
						parser,
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

	const isPreserved = (virtualPath: string): boolean =>
		preservedPrefixes.some((prefix) => virtualPath === prefix || virtualPath.startsWith(`${prefix}/`));
	const staleCandidates: string[] = [];
	for (const [virtualPath, entry] of Object.entries(previous.entries)) {
		if (handledPrevious.has(virtualPath)) continue;
		if (nextEntries[virtualPath]) continue;
		if (preservedPrefixes.length > 0 && isPreserved(virtualPath)) {
			nextEntries[virtualPath] = entry;
			continue;
		}
		staleCandidates.push(virtualPath);
	}
	if (preservedPrefixes.length > 0) {
		const currentPaths = new Set(current.map((entry) => entry.virtualPath));
		for (const [virtualPath, skip] of Object.entries(previous.skipped ?? {})) {
			if (virtualPath in nextSkipped || currentPaths.has(virtualPath) || !isPreserved(virtualPath)) continue;
			nextSkipped[virtualPath] = skip;
		}
	}

	// Only delete previous index entries for files no longer present among current supported files.
	// Entries outside this run's supported collect set that disappeared should still be cleaned.
	const indexedCount = Object.keys(previous.entries).length;
	const maxDeleteRatio = options.maxDeleteRatio ?? DEFAULT_MAX_DELETE_RATIO;
	const massDelete =
		options.force !== true &&
		staleCandidates.length > MASS_DELETE_FLOOR &&
		staleCandidates.length / indexedCount > maxDeleteRatio;
	if (massDelete) {
		for (const virtualPath of staleCandidates) {
			const kept = previous.entries[virtualPath];
			if (kept) nextEntries[virtualPath] = kept;
		}
		diagnostics.push({
			code: "mass-delete-suppressed",
			severity: "warning",
			message: `This refresh would remove ${staleCandidates.length} of ${indexedCount} indexed documents, more than the allowed ${Math.round(maxDeleteRatio * 100)}%; their mirrors were kept. If the sources are really gone, run a forced refresh.`,
			source: "parsed-mirror",
		});
	} else {
		for (const virtualPath of staleCandidates) {
			removeFile(parsedOutputPath(options.root, previous.entries[virtualPath]?.virtualPath ?? virtualPath));
			deleted += 1;
			diagnostics.push({
				code: "deleted-mirror",
				severity: "info",
				message: "A previously indexed document is gone; its parsed mirror was removed.",
				source: virtualPath,
			});
		}
	}

	options.onProgress?.({ processed: current.length, total: current.length, written, deleted, skipped });
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
	const current = (await listCurrentFiles(options.searchPaths, supportedExtensions, walkLimits(options))).entries;
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
		// source counts as newly added (no mirror entry exists for it). A "source-unreadable"
		// skip only paces refresh retries: the source was never indexed, so it stays reported.
		const honoredSkip =
			prevSkip?.reason === "user-excluded" || prevSkip?.reason === "source-unreadable" ? undefined : prevSkip;
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

type SearchRootUnavailableReason = "gone" | "not-a-directory" | "unreadable" | "empty";

const SEARCH_ROOT_UNAVAILABLE_MESSAGES: Readonly<Record<SearchRootUnavailableReason, string>> = {
	gone: "A search root no longer exists; its previously indexed documents were left untouched.",
	"not-a-directory": "A search root is no longer a directory; its previously indexed documents were left untouched.",
	unreadable:
		"A search root cannot be listed (unmounted, signed out, or permission denied); its previously indexed documents were left untouched.",
	empty: "A search root is empty although it previously held indexed documents; they were left untouched. If they were removed on purpose, run a forced refresh.",
};

interface UnavailableRoot {
	readonly prefix: string;
	readonly reason: SearchRootUnavailableReason;
}

interface WalkLimits {
	readonly maxScannedEntries: number;
	readonly maxDepth: number;
}

interface CurrentFileListing {
	readonly entries: CurrentEntry[];
	readonly diagnostics: ParsedMirrorDiagnostic[];
	/** Virtual prefixes the walk could not fully see; absence under them is not evidence of deletion. */
	readonly unreliablePrefixes: string[];
	/** Roots that were not walked because they were missing, unlistable, or empty. */
	readonly unavailableRoots: UnavailableRoot[];
}

interface WalkState {
	readonly sourceRoot: SourceRoot;
	readonly supportedExtensions: ReadonlySet<string>;
	readonly limits: WalkLimits;
	readonly listing: CurrentFileListing;
	readonly unreadableDirectories: Set<string>;
	readonly truncationReasons: Set<string>;
	visited: number;
	stopped: boolean;
}

function walkLimits(options: ParsedMirrorSyncOptions): WalkLimits {
	return {
		maxScannedEntries: options.maxScannedEntries ?? DEFAULT_MAX_SCANNED_ENTRIES,
		maxDepth: options.maxDepth ?? DEFAULT_MAX_WALK_DEPTH,
	};
}

function hasIndexedUnder(index: ParsedMirrorIndex, prefix: string): boolean {
	const inside = (virtualPath: string): boolean => virtualPath === prefix || virtualPath.startsWith(`${prefix}/`);
	return Object.keys(index.entries).some(inside) || Object.keys(index.skipped ?? {}).some(inside);
}

function errnoCode(error: unknown): string | undefined {
	if (!(error instanceof Error)) return undefined;
	const code = (error as NodeJS.ErrnoException).code;
	return typeof code === "string" ? code : undefined;
}

/** Why a search root cannot be trusted as a listing, or `undefined` when it is a readable, non-empty directory. */
async function probeSearchRoot(rootPath: string): Promise<SearchRootUnavailableReason | undefined> {
	let isDirectory: boolean;
	try {
		isDirectory = (await stat(rootPath)).isDirectory();
	} catch (error) {
		const code = errnoCode(error);
		return code === "ENOENT" || code === "ENOTDIR" ? "gone" : "unreadable";
	}
	if (!isDirectory) return "not-a-directory";
	let dir: Dir;
	try {
		dir = await opendir(rootPath);
	} catch {
		return "unreadable";
	}
	try {
		return (await dir.read()) === null ? "empty" : undefined;
	} catch {
		return "unreadable";
	} finally {
		await dir.close().catch(() => undefined);
	}
}

async function listCurrentFiles(
	searchPaths: readonly string[],
	supportedExtensions: ReadonlySet<string>,
	limits: WalkLimits,
): Promise<CurrentFileListing> {
	const listing: CurrentFileListing = { entries: [], diagnostics: [], unreliablePrefixes: [], unavailableRoots: [] };
	for (const sourceRoot of planSourceRoots(searchPaths)) {
		const reason = await probeSearchRoot(sourceRoot.rootPath);
		if (reason !== undefined) {
			listing.unavailableRoots.push({ prefix: sourceRoot.prefix, reason });
			continue;
		}
		const state: WalkState = {
			sourceRoot,
			supportedExtensions,
			limits,
			listing,
			unreadableDirectories: new Set(),
			truncationReasons: new Set(),
			visited: 0,
			stopped: false,
		};
		await collectFiles(state, sourceRoot.rootPath, 0);
	}
	listing.entries.sort((a, b) => a.virtualPath.localeCompare(b.virtualPath));
	return listing;
}

function warnDirectoryUnreadable(state: WalkState, directory: string, code: string): void {
	if (state.unreadableDirectories.has(directory)) return;
	state.unreadableDirectories.add(directory);
	state.listing.diagnostics.push({
		code: "directory-unreadable",
		severity: "warning",
		message: `A directory could not be read (${code}); its previously indexed documents were left untouched.`,
		source: sourceIdentifier(state.sourceRoot, directory),
		reason: code,
	});
}

function warnWalkTruncated(state: WalkState, reason: "max-entries" | "max-depth"): void {
	if (state.truncationReasons.has(reason)) return;
	state.truncationReasons.add(reason);
	state.listing.diagnostics.push({
		code: "walk-truncated",
		severity: "warning",
		message:
			reason === "max-entries"
				? `A search root has more than ${state.limits.maxScannedEntries} entries; the walk stopped early and previously indexed documents it did not reach were left untouched.`
				: `A search root has directories nested deeper than ${state.limits.maxDepth} levels; documents below that depth were not scanned and previously indexed ones were left untouched.`,
		source: state.sourceRoot.prefix,
		reason,
	});
}

async function collectFiles(state: WalkState, directory: string, depth: number): Promise<void> {
	const { sourceRoot, listing } = state;
	let dir: Dir;
	try {
		dir = await opendir(directory);
	} catch (error) {
		const code = errnoCode(error);
		if (code === undefined) throw error;
		// A directory that vanished between listing and opening is genuinely gone.
		if (code === "ENOENT" || code === "ENOTDIR") return;
		listing.unreliablePrefixes.push(sourceIdentifier(sourceRoot, directory));
		warnDirectoryUnreadable(state, directory, code);
		return;
	}
	try {
		for await (const entry of dir) {
			if (state.visited >= state.limits.maxScannedEntries) {
				state.stopped = true;
				listing.unreliablePrefixes.push(sourceRoot.prefix);
				warnWalkTruncated(state, "max-entries");
				return;
			}
			state.visited += 1;
			if (SKIP_DIR_NAMES.has(entry.name)) continue;
			if (PRODUCT_ARTIFACT_FILE_NAMES.has(entry.name)) continue;
			const sourcePath = resolve(directory, entry.name);
			if (entry.isDirectory()) {
				if (depth + 1 > state.limits.maxDepth) {
					listing.unreliablePrefixes.push(sourceIdentifier(sourceRoot, sourcePath));
					warnWalkTruncated(state, "max-depth");
					continue;
				}
				await collectFiles(state, sourcePath, depth + 1);
				if (state.stopped) return;
				continue;
			}
			if (!entry.isFile()) continue;
			const extension = normalizeExtension(extname(entry.name));
			if (!state.supportedExtensions.has(extension)) continue;
			let fileStat: Awaited<ReturnType<typeof stat>>;
			try {
				fileStat = await stat(sourcePath, { bigint: true });
			} catch (error) {
				const code = errnoCode(error);
				if (code === undefined) throw error;
				if (code === "ENOENT" || code === "ENOTDIR") continue;
				// Present but unstat-able (for example a directory without search permission): keep its mirror.
				listing.unreliablePrefixes.push(sourceIdentifier(sourceRoot, sourcePath));
				warnDirectoryUnreadable(state, directory, code);
				continue;
			}
			listing.entries.push({
				virtualPath: sourceIdentifier(sourceRoot, sourcePath),
				sourcePath,
				sizeBytes: Number(fileStat.size),
				mtimeNs: Number(fileStat.mtimeNs),
			});
		}
	} catch (error) {
		const code = errnoCode(error);
		if (code === undefined) throw error;
		listing.unreliablePrefixes.push(sourceIdentifier(sourceRoot, directory));
		warnDirectoryUnreadable(state, directory, code);
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
