import { createHash } from "node:crypto";
import type { BigIntStats, Dir } from "node:fs";
import { opendir, readFile, stat } from "node:fs/promises";
import { basename, dirname, extname, join, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import type { CloudFileProbe } from "../filesystem/cloud-placeholder.ts";
import { planSourceRoots, type SourceRoot, sourceIdentifier } from "../filesystem/source-paths.ts";
import { resolveMinSyncBinary } from "../minsync/binary.ts";
import { MinSyncStageSession } from "../minsync/stage.ts";
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
import type { ParseOutput } from "../parser/types.ts";
import { abortReason } from "../process/abort-reason.ts";
import { writeFileAtomic } from "./atomic-write.ts";
import { CloudGate } from "./cloud.ts";
import {
	loadMirrorIndex,
	mirrorIndexFromStage,
	mirrorStageKey,
	type ParsedMirrorIndex,
	type ParsedMirrorSkipReason,
} from "./index-store.ts";
import { parsedMirrorRoot, parsedOutputPath } from "./paths.ts";

/** Refuse to fully load/parse sources larger than this. Keeps Node heap stable on home folders. */
export const DEFAULT_MAX_SOURCE_BYTES = 50 * 1024 * 1024;

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
	/** Embedded/test binary override; the normal path resolves the installed MinSync. */
	readonly minSyncBinaryPath?: string;
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
	 * Stops between files and kills an in-flight parse worker. Rejects with the signal's
	 * reason; MinSync's durable journal retains completed items for the next run.
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
	/**
	 * Cloud-aware handling of File Provider placeholders (files whose data is not on disk; reading one
	 * downloads it). `undefined` means the platform default: macOS probes with `stat`/`brctl`, other
	 * platforms read every file as before. Tests inject their own probe.
	 */
	readonly cloudProbe?: CloudFileProbe;
	/** Most placeholders one refresh may download. Default {@link DEFAULT_CLOUD_MAX_FILES}. */
	readonly cloudMaxFiles?: number;
	/** Most placeholder bytes one refresh may download; the first file is always allowed. Default {@link DEFAULT_CLOUD_MAX_BYTES}. */
	readonly cloudMaxBytes?: number;
	/** Release an iCloud file again once it is parsed and indexed. Default true. */
	readonly cloudEvictAfterParse?: boolean;
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
	| "source-changed"
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
	| "directory-unreadable"
	| "cloud-download-deferred"
	| "cloud-placeholder-skipped"
	| "cloud-evict-failed";

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
	readonly sourceVersion: string;
	readonly statIdentity: string;
	/** Same as {@link statIdentity} without ctime; see {@link contentIdentity}. */
	readonly contentIdentity: string;
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

function parseFingerprint(options: ParsedMirrorSyncOptions): string {
	const stable = (value: unknown): unknown => {
		if (typeof value === "function") return String(value);
		if (Array.isArray(value)) return value.map(stable);
		if (value !== null && typeof value === "object") {
			return Object.fromEntries(
				Object.entries(value)
					.sort(([a], [b]) => a.localeCompare(b))
					.map(([key, item]) => [key, stable(item)]),
			);
		}
		return value;
	};
	return createHash("sha256")
		.update(JSON.stringify(stable({ format: 1, parser: outputParserOptions(options.parserOptions) })))
		.digest("hex");
}

function outputParserOptions(options: DefaultParserRegistryOptions = {}): unknown {
	const { fileTimeoutMs: _timeout, maxHeapMb: _heap, ocr, ...parser } = options;
	return { ...parser, ...(ocr === undefined ? {} : { ocr: { enabled: ocr.enabled } }) };
}

function retryFingerprint(options: ParsedMirrorSyncOptions, fingerprint: string): string {
	const parser = options.parserOptions;
	return createHash("sha256")
		.update(
			JSON.stringify({
				fingerprint,
				fileTimeoutMs: parser?.fileTimeoutMs,
				maxHeapMb: parser?.maxHeapMb,
				ocrTimeoutMs: parser?.ocr?.timeoutMs,
				ocrCachePath: parser?.ocr?.cachePath,
			}),
		)
		.digest("hex");
}

function sourceVersion(info: { size: bigint; mtimeNs: bigint }): string {
	return `${info.size}:${info.mtimeNs}`;
}

/** Strong observation within this run; remounts and metadata edits are not content revisions. */
function statIdentity(info: BigIntStats): string {
	return [info.size, info.mtimeNs, info.ctimeNs, info.ino, info.dev].join(":");
}

/**
 * Identity of a placeholder's content across the download that reading it causes. Measured on macOS 26 with
 * iCloud Drive: hydrating a file moves ctime every time, and the first hydrate of a file also shifts mtime
 * by tens of nanoseconds (size, inode and device stay put) because the provider rounds the timestamp to the
 * precision the cloud keeps. So ctime is left out and mtime is compared to the second; a real edit changes
 * size or the mtime second.
 */
function contentIdentity(info: BigIntStats): string {
	return [info.size, info.mtimeNs / 1_000_000_000n, info.ino, info.dev].join(":");
}

type SourceOmission = { reason: ParsedMirrorSkipReason; diagnostic?: Omit<ParsedMirrorDiagnostic, "source"> };

function sourcePlan(
	current: readonly CurrentEntry[],
	registry: ParserRegistry,
	options: ParsedMirrorSyncOptions,
): {
	versions: Map<string, string>;
	omitted: Map<string, SourceOmission>;
} {
	const versions = new Map<string, string>();
	const omitted = new Map<string, SourceOmission>();
	for (const entry of current) {
		const parser = registry.getForVirtualPath(entry.virtualPath);
		if (isPathExcluded(entry.sourcePath, options.userExcludedSourcePaths))
			omitted.set(entry.virtualPath, { reason: "user-excluded" });
		else if (options.excludeSourcePaths?.has(entry.sourcePath))
			omitted.set(entry.virtualPath, {
				reason: "duplicate-excluded",
				diagnostic: {
					code: "duplicate-excluded",
					severity: "info",
					message: "An older exact duplicate was excluded from indexing; the newest copy remains indexed.",
				},
			});
		else if (!parser || entry.sizeBytes > (options.maxSourceBytes ?? DEFAULT_MAX_SOURCE_BYTES))
			omitted.set(entry.virtualPath, {
				reason: "parser-skipped",
				...(parser
					? {
							diagnostic: {
								code: "parser-skipped",
								severity: "warning",
								message: "Source exceeds the configured max parse size and was skipped during indexing.",
							},
						}
					: {}),
			});
		versions.set(
			entry.virtualPath,
			JSON.stringify({
				virtualPath: entry.virtualPath,
				sourcePath: entry.sourcePath,
				parserName: parser?.name ?? "unsupported",
				parserVersion: parser?.version,
				sourceMtimeNs: entry.mtimeNs,
				sourceSizeBytes: entry.sizeBytes,
				sourceVersion: entry.sourceVersion,
				...(omitted.has(entry.virtualPath) ? { omission: omitted.get(entry.virtualPath)!.reason } : {}),
			}),
		);
	}
	return { versions, omitted };
}

async function syncWithRegistry(
	options: ParsedMirrorSyncOptions,
	registry: ParserRegistry,
	isolated: IsolatedParseRunner | undefined,
): Promise<ParsedMirrorSyncResult> {
	const listing = await listCurrentFiles(options.searchPaths, supportedExtensionSet(registry), walkLimits(options));
	const current = listing.entries;
	const binaryPath = await resolveMinSyncBinary({ root: options.root, binaryPath: options.minSyncBinaryPath });
	const previous = loadMirrorIndex(options.root, { minSyncBinaryPath: binaryPath });
	const diagnostics: ParsedMirrorDiagnostic[] = [...listing.diagnostics];
	const preservedPrefixes = [...listing.unreliablePrefixes];
	for (const unavailable of listing.unavailableRoots) {
		if (unavailable.reason === "empty" && (options.force === true || !hasIndexedUnder(previous, unavailable.prefix)))
			continue;
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
	const gate = new CloudGate({
		probe: options.cloudProbe,
		maxFiles: options.cloudMaxFiles,
		maxBytes: options.cloudMaxBytes,
		evictAfterParse: options.cloudEvictAfterParse,
	});
	const fingerprint = parseFingerprint(options);
	const failureFingerprint = retryFingerprint(options, fingerprint);
	const { versions, omitted } = sourcePlan(current, registry, options);
	const currentPaths = new Set(current.map((entry) => entry.virtualPath));
	const pathsByKey = new Map([...currentPaths, ...(previous.known ?? [])].map((path) => [mirrorStageKey(path), path]));
	const staleCandidates = new Set(
		Object.keys(previous.entries).filter(
			(path) =>
				!currentPaths.has(path) &&
				!preservedPrefixes.some((prefix) => path === prefix || path.startsWith(`${prefix}/`)),
		),
	);
	const indexedCount = Object.keys(previous.entries).length;
	const maxDeleteRatio = options.maxDeleteRatio ?? DEFAULT_MAX_DELETE_RATIO;
	const massDelete =
		options.force !== true &&
		staleCandidates.size > MASS_DELETE_FLOOR &&
		staleCandidates.size / indexedCount > maxDeleteRatio;
	if (massDelete)
		diagnostics.push({
			code: "mass-delete-suppressed",
			severity: "warning",
			message: `This refresh would remove ${staleCandidates.size} of ${indexedCount} indexed documents, more than the allowed ${Math.round(maxDeleteRatio * 100)}%; their mirrors were kept. If the sources are really gone, run a forced refresh.`,
			source: "parsed-mirror",
		});
	const unknown = new Set<string>();
	for (const path of previous.known ?? []) {
		if (
			!currentPaths.has(path) &&
			((massDelete && staleCandidates.has(path)) ||
				preservedPrefixes.some((prefix) => path === prefix || path.startsWith(`${prefix}/`)))
		)
			unknown.add(mirrorStageKey(path));
	}
	const session = new MinSyncStageSession(
		binaryPath,
		join(parsedMirrorRoot(options.root), "stage"),
		options.root,
		options.signal,
	);
	let written = 0;
	let deleted = 0;
	let skipped = 0;
	const skipFile = async (
		entry: CurrentEntry,
		diagnostic: Omit<ParsedMirrorDiagnostic, "source">,
		reason?: ParsedMirrorSkipReason,
	): Promise<void> => {
		if (previous.entries[entry.virtualPath]) deleted += 1;
		skipped += 1;
		diagnostics.push({ ...diagnostic, source: entry.virtualPath });
		if (reason === "parser-skipped")
			await session.skip(mirrorStageKey(entry.virtualPath), versions.get(entry.virtualPath)!, reason);
		else
			await session.fail(mirrorStageKey(entry.virtualPath), versions.get(entry.virtualPath)!, diagnostic.code, {
				defer: reason === "parser-failed" || reason === "source-unreadable",
				discard_output: true,
			});
	};
	try {
		const plan = await session.open({
			stage: "parse",
			source_id: "autorag-parsed-mirror",
			fingerprint,
			retry_fingerprint: failureFingerprint,
			retry_failed: options.retrySkipped === true,
			output_root: join(parsedMirrorRoot(options.root), "files"),
			items: [...versions].map(([path, version]) => ({ key: mirrorStageKey(path), version })),
			unknown: [...unknown],
			full: options.force === true,
		});
		const pending = new Set(
			plan.decisions
				.filter((decision) => decision.action === "process" || decision.action === "retry")
				.map((decision) => decision.key),
		);
		const deferred = new Map(
			plan.decisions.filter((decision) => decision.deferred === true).map((decision) => [decision.key, decision]),
		);
		for (const decision of plan.decisions) {
			if (decision.action !== "delete") continue;
			const path = pathsByKey.get(decision.key);
			await session.delete(decision.key);
			if (path === undefined) continue;
			if (previous.entries[path]) deleted += 1;
			if (!currentPaths.has(path))
				diagnostics.push({
					code: "deleted-mirror",
					severity: "info",
					message: "A previously indexed document is gone; its parsed mirror was removed.",
					source: path,
				});
		}
		await gate.classify(
			current.filter((entry) => pending.has(mirrorStageKey(entry.virtualPath))).map((entry) => entry.sourcePath),
		);
		for (const [position, entry] of current.entries()) {
			options.onProgress?.({
				processed: position,
				total: current.length,
				written,
				deleted,
				skipped,
				currentVirtualPath: entry.virtualPath,
			});
			if (options.signal?.aborted) throw abortReason(options.signal);
			const omission = omitted.get(entry.virtualPath);
			if (omission !== undefined) {
				if (pending.has(mirrorStageKey(entry.virtualPath))) {
					if (previous.entries[entry.virtualPath]) deleted += 1;
					await session.skip(mirrorStageKey(entry.virtualPath), versions.get(entry.virtualPath)!, omission.reason);
				}
				skipped += 1;
				if (omission.diagnostic) diagnostics.push({ ...omission.diagnostic, source: entry.virtualPath });
				continue;
			}
			const deferredFailure = deferred.get(mirrorStageKey(entry.virtualPath));
			if (deferredFailure !== undefined) {
				skipped += 1;
				const code = deferredFailure.error === "source-unreadable" ? "source-unreadable" : "parser-failed";
				diagnostics.push({
					code,
					severity: "warning",
					source: entry.virtualPath,
					message:
						code === "parser-failed"
							? PARSER_FAILURE_MESSAGE
							: "The source could not be read; retry is deferred until the source or settings change, or an explicit retry is requested.",
				});
				continue;
			}
			const parser = registry.getForVirtualPath(entry.virtualPath)!;
			const outputPath = parsedOutputPath(options.root, entry.virtualPath);
			const cloudDecision = pending.has(mirrorStageKey(entry.virtualPath))
				? await gate.decide(entry.sourcePath, entry.sizeBytes, dirname(entry.sourcePath))
				: ({ kind: "local" } as const);
			if (cloudDecision.kind === "skip") {
				skipped += 1;
				diagnostics.push({
					code: "cloud-placeholder-skipped",
					severity: "info",
					message: `The file is not downloaded and its provider cannot be released again, so it was not read; its last indexed copy was kept. Make it available offline to index it.`,
					source: entry.virtualPath,
					...(cloudDecision.provider === undefined ? {} : { reason: cloudDecision.provider }),
				});
				continue;
			}
			if (cloudDecision.kind === "defer") {
				skipped += 1;
				diagnostics.push({
					code: "cloud-download-deferred",
					severity: "info",
					message: "The download budget for this refresh is used up; the file stays queued for the next refresh.",
					source: entry.virtualPath,
				});
				continue;
			}
			if (pending.has(mirrorStageKey(entry.virtualPath))) {
				let bytes: Awaited<ReturnType<typeof readFile>>;
				try {
					bytes = await withTransientRetry(() => readFile(entry.sourcePath), transientRetries, transientBackoffMs);
				} catch (error) {
					const failure = sourceReadFailure(error);
					await skipFile(entry, failure, failure.code === "source-unreadable" ? "source-unreadable" : undefined);
					continue;
				}
				const tooSmallImage = standaloneImageTooSmallMessage(entry.virtualPath, bytes);
				if (tooSmallImage !== undefined) {
					await skipFile(
						entry,
						{ code: "parser-skipped", severity: "warning", message: tooSmallImage },
						"parser-skipped",
					);
					continue;
				}
				let parsed: ParseOutput;
				try {
					const input = { virtualPath: entry.virtualPath, sourcePath: entry.sourcePath, bytes };
					parsed =
						isolated !== undefined && ISOLATED_PARSER_NAMES.has(parser.name)
							? await isolated.parse(parser.name, input, options.signal)
							: await parser.parse(input);
				} catch (error) {
					if (options.signal?.aborted) throw abortReason(options.signal);
					if (error instanceof ParseError)
						await skipFile(
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
						);
					else
						await skipFile(entry, {
							code: "parser-crashed",
							severity: "warning",
							message: PARSER_CRASH_MESSAGE,
							...(error instanceof Error ? { reason: error.name } : {}),
						});
					continue;
				}
				let info: BigIntStats;
				try {
					info = await stat(entry.sourcePath, { bigint: true });
				} catch (error) {
					await skipFile(entry, sourceReadFailure(error));
					continue;
				}
				const unchanged =
					cloudDecision.kind === "download"
						? contentIdentity(info) === entry.contentIdentity
						: statIdentity(info) === entry.statIdentity;
				if (!unchanged) {
					skipped += 1;
					diagnostics.push({
						code: "source-changed",
						severity: "warning",
						message:
							"The source changed while parsing; its last good mirror was kept and it will be retried on the next refresh.",
						source: entry.virtualPath,
					});
					await session.fail(
						mirrorStageKey(entry.virtualPath),
						versions.get(entry.virtualPath)!,
						"source-changed",
					);
					continue;
				}
				const markdown = normalizeMarkdown(parsed.markdown);
				try {
					await withTransientRetry(
						() => writeFileAtomic(outputPath, markdown),
						transientRetries,
						transientBackoffMs,
					);
				} catch (error) {
					const code = errorCode(error);
					if (code !== undefined && Object.hasOwn(WORKSPACE_ERROR_CODES, code))
						throw new MirrorWorkspaceError(options.root, error);
					await skipFile(entry, {
						code: "mirror-write-failed",
						severity: "warning",
						message: error instanceof Error ? error.message : String(error),
						...(code !== undefined ? { reason: code } : {}),
					});
					continue;
				}
				await session.commit(mirrorStageKey(entry.virtualPath), versions.get(entry.virtualPath)!, {
					path: basename(outputPath),
					size: Buffer.byteLength(markdown),
				});
				for (const diagnostic of parsed.diagnostics ?? [])
					diagnostics.push({ ...diagnostic, source: entry.virtualPath });
				written += 1;
				if (cloudDecision.kind === "download" && gate.evictAfterParse) {
					const released = await gate.release(entry.sourcePath);
					if (!released.ok) {
						diagnostics.push({
							code: "cloud-evict-failed",
							severity: "warning",
							message: `The file was indexed but its local copy could not be released: ${released.message}`,
							source: entry.virtualPath,
						});
					}
				}
			}
		}
		await session.finish();
		options.onProgress?.({ processed: current.length, total: current.length, written, deleted, skipped });
		return {
			scanned: current.length,
			written,
			deleted,
			skipped,
			indexPath: join(parsedMirrorRoot(options.root), "stage"),
			diagnostics,
		};
	} catch (error) {
		throw options.signal?.aborted ? abortReason(options.signal) : error;
	} finally {
		await session.close();
	}
}

/** Ask MinSync's read-only plan which current source revisions still need work. */
export async function detectMirrorStaleness(options: ParsedMirrorSyncOptions): Promise<ParsedMirrorDiagnostic[]> {
	const registry = options.registry ?? createDefaultParserRegistry(options.parserOptions);
	const current = (await listCurrentFiles(options.searchPaths, supportedExtensionSet(registry), walkLimits(options)))
		.entries;
	if (current.length === 0) return [];
	const binaryPath = await resolveMinSyncBinary({
		root: options.root,
		binaryPath: options.minSyncBinaryPath,
		install: false,
	});
	const session = new MinSyncStageSession(
		binaryPath,
		join(parsedMirrorRoot(options.root), "stage"),
		options.root,
		options.signal,
	);
	try {
		const previous = mirrorIndexFromStage(
			options.root,
			await session.inspect({
				stage: "parse",
				source_id: "autorag-parsed-mirror",
				output_root: join(parsedMirrorRoot(options.root), "files"),
			}),
		);
		const { versions, omitted } = sourcePlan(current, registry, options);
		const fingerprint = parseFingerprint(options);
		const plan = await session.plan({
			stage: "parse",
			source_id: "autorag-parsed-mirror",
			fingerprint,
			retry_fingerprint: retryFingerprint(options, fingerprint),
			output_root: join(parsedMirrorRoot(options.root), "files"),
			items: [...versions].map(([path, version]) => ({ key: mirrorStageKey(path), version })),
		});
		const work = new Set(
			plan.decisions
				.filter(
					(decision) =>
						decision.action === "process" ||
						decision.action === "retry" ||
						(decision.deferred && decision.error === "source-unreadable"),
				)
				.map((decision) => decision.key),
		);
		const diagnostics: ParsedMirrorDiagnostic[] = [];
		for (const entry of current) {
			if (omitted.has(entry.virtualPath) || !work.has(mirrorStageKey(entry.virtualPath))) continue;
			const prev = previous.entries[entry.virtualPath];
			const skip = previous.skipped?.[entry.virtualPath];
			const known =
				prev ?? (skip?.reason === "user-excluded" || skip?.reason === "source-unreadable" ? undefined : skip);
			const parser = registry.getForVirtualPath(entry.virtualPath)!;
			const parserChanged =
				prev !== undefined &&
				(prev.parserName !== parser.name ||
					prev.parserVersion !== parser.version ||
					prev.parserFingerprint !== fingerprint);
			const reason =
				known === undefined
					? "source-added"
					: parserChanged
						? "parser-changed"
						: known.sourceMtimeNs !== entry.mtimeNs && known.sourceSizeBytes !== entry.sizeBytes
							? "mtime-and-size-changed"
							: known.sourceMtimeNs !== entry.mtimeNs
								? "mtime-changed"
								: known.sourceSizeBytes !== entry.sizeBytes
									? "size-changed"
									: "source-or-mirror-changed";
			diagnostics.push({
				code: "stale-index",
				severity: "warning",
				source: entry.virtualPath,
				reason,
				message: `Source ${entry.virtualPath} is ${reason.replaceAll("-", " ")} since the last refresh; indexes may be stale.`,
			});
		}
		return diagnostics;
	} finally {
		await session.close();
	}
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
	return (index.known ?? []).some(inside);
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
				sourceVersion: sourceVersion(fileStat),
				statIdentity: statIdentity(fileStat),
				contentIdentity: contentIdentity(fileStat),
			});
		}
	} catch (error) {
		const code = errnoCode(error);
		if (code === undefined) throw error;
		listing.unreliablePrefixes.push(sourceIdentifier(sourceRoot, directory));
		warnDirectoryUnreadable(state, directory, code);
	}
}
