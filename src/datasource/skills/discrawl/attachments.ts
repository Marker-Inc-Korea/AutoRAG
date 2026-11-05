import { createHash } from "node:crypto";
import {
	copyFileSync,
	createReadStream,
	lstatSync,
	mkdirSync,
	readFileSync,
	renameSync,
	rmSync,
	type Stats,
	statSync,
} from "node:fs";
import { extname, join, posix } from "node:path";
import { resolveAutoRAGHome } from "../../../config/home.ts";
import { acquireFileLock, type FileLockHandle } from "../../../filesystem/file-lock.ts";
import { writeFileAtomic } from "../../../mirror/atomic-write.ts";
import { sanitizeIdSegment } from "../../connector.ts";
import type {
	DiscrawlAttachmentFetchResult,
	DiscrawlAttachmentListResult,
	DiscrawlAttachmentRow,
	DiscrawlStatusResult,
} from "./types.ts";

/** Per-connection status file kept at the top of the attachments folder. */
export const DISCORD_ATTACHMENT_STATUS_FILENAME = "download-status.json";
const STATUS_VERSION = 1;
/** discrawl's private config and download cache inside the attachments folder. */
const INTERNAL_DIRNAME = ".discrawl";
const LOCK_DIRNAME = ".lock";
/** A link this close to its `ex` deadline counts as expired: the download would race the signature. */
const LINK_EXPIRY_MARGIN_MS = 60_000;
/** Lets a fetch row that discrawl stamped a moment before AutoRAG's clock still count as this run's. */
const CLOCK_SKEW_MS = 5_000;
const MAX_FILENAME_CHARS = 150;

/**
 * Where a Discord connection's attachments are saved: `<AUTORAG_HOME>/attachments/discord/<connection>`.
 * That is `~/.autorag/attachments/discord/<connection>` for every user on macOS, Linux, and Windows, unless
 * `AUTORAG_HOME` moves the whole AutoRAG home.
 */
export function discordAttachmentsRoot(connection: string, env: NodeJS.ProcessEnv = process.env): string {
	return join(resolveAutoRAGHome(env), "attachments", "discord", sanitizeIdSegment(connection));
}

/**
 * - `downloaded`: the saved file is still byte-identical to what Discord served.
 * - `not-downloaded`: never saved.
 * - `changed`: the saved file was edited. It stays as the user's own file; the original is fetched again.
 * - `missing`: the saved file was deleted, moved, or renamed; the original is fetched again.
 * - `failed`: the last download attempt failed (`error` says why).
 * - `skipped`: discrawl refused the file (for example `too_large`) and it is not retried.
 */
export type DiscordAttachmentState = "downloaded" | "not-downloaded" | "changed" | "missing" | "failed" | "skipped";

export interface DiscordAttachmentRecord {
	readonly attachmentId: string;
	readonly messageId: string;
	readonly guildId: string;
	readonly guildName?: string;
	readonly channelId: string;
	readonly channelName?: string;
	readonly authorName?: string;
	readonly createdAt?: string;
	readonly filename: string;
	readonly contentType?: string;
	readonly size: number;
	readonly state: DiscordAttachmentState;
	/** True when the file is not saved intact and Discord's signed link still works. */
	readonly downloadable: boolean;
	/** When the signed link stops working (Discord signs attachment links for 24 hours). */
	readonly linkExpiresAt?: string;
	/** Saved file, relative to the attachments folder, with `/` separators. */
	readonly path?: string;
	readonly sha256?: string;
	readonly bytes?: number;
	readonly mtimeMs?: number;
	readonly downloadedAt?: string;
	readonly error?: string;
}

export interface DiscordAttachmentSummary {
	readonly total: number;
	readonly downloaded: number;
	readonly notDownloaded: number;
	readonly changed: number;
	readonly missing: number;
	readonly failed: number;
	readonly skipped: number;
	/** Not saved intact and still downloadable now. */
	readonly downloadable: number;
	/** Not saved intact and the link already expired: these can no longer be downloaded. */
	readonly expired: number;
}

/** What one sync downloaded. */
export interface DiscordAttachmentRun {
	readonly attempted: number;
	readonly fetched: number;
	readonly failed: number;
	readonly skipped: number;
	readonly bytes: number;
}

export interface DiscordAttachmentSyncOk {
	readonly ok: true;
	readonly root: string;
	readonly statusPath: string;
	readonly summary: DiscordAttachmentSummary;
	readonly run: DiscordAttachmentRun;
	readonly records: readonly DiscordAttachmentRecord[];
}

export interface DiscordAttachmentSyncFail {
	readonly ok: false;
	readonly reason: "busy" | "discrawl-failed";
	readonly message: string;
}

export type DiscordAttachmentSyncResult = DiscordAttachmentSyncOk | DiscordAttachmentSyncFail;

export interface DiscordAttachmentSyncOptions {
	/** Download every attachment that is not saved intact and whose link still works. */
	readonly download: boolean;
	readonly signal?: AbortSignal;
}

/** The discrawl calls the store needs; `DiscrawlClient` implements them. */
export interface DiscordAttachmentClient {
	status(signal?: AbortSignal): Promise<DiscrawlStatusResult>;
	listAttachments(configPath: string, guildId?: string, signal?: AbortSignal): Promise<DiscrawlAttachmentListResult>;
	fetchAttachments(
		configPath: string,
		messageId: string,
		signal?: AbortSignal,
	): Promise<DiscrawlAttachmentFetchResult>;
}

/** What the Discord skill calls after each import; `DiscordAttachmentStore` implements it. */
export interface DiscordAttachmentSyncer {
	readonly root: string;
	sync(options: DiscordAttachmentSyncOptions): Promise<DiscordAttachmentSyncResult>;
}

export interface DiscordAttachmentStoreOptions {
	readonly client: DiscordAttachmentClient;
	/** Attachments folder; see {@link discordAttachmentsRoot}. */
	readonly root: string;
	readonly guildId?: string;
	readonly channelIds?: readonly string[];
	readonly channelNames?: readonly string[];
	readonly now?: () => number;
}

/** A record before its `downloadable` flag is computed for the current time. */
type TrackedRecord = Omit<DiscordAttachmentRecord, "downloadable">;

/** What a previous run's status file contributes: where the file was saved and how the last try ended. */
interface SavedEntry {
	readonly state: DiscordAttachmentState;
	readonly path?: string;
	readonly sha256?: string;
	readonly bytes?: number;
	readonly mtimeMs?: number;
	readonly downloadedAt?: string;
	readonly error?: string;
}

const STATES: Readonly<Record<DiscordAttachmentState, true>> = {
	downloaded: true,
	"not-downloaded": true,
	changed: true,
	missing: true,
	failed: true,
	skipped: true,
};

/**
 * Saves Discord attachments into one folder and keeps `download-status.json` beside them.
 *
 * discrawl does the downloading; AutoRAG never talks to Discord itself. AutoRAG hands discrawl a private
 * config whose `cache_dir` is `.discrawl/` inside the folder, asks it to fetch one message's attachments at a
 * time, and then moves each file to `<server>/<channel>/<original name>`. A saved file is identified by its
 * SHA-256: when the file at the recorded path is gone or its bytes differ, it no longer counts as the
 * attachment. An edited file is left in place as the user's own and the original is saved again beside it
 * under a free name, as long as Discord's signed link still works. Nothing in the folder is ever overwritten
 * or deleted.
 */
export class DiscordAttachmentStore implements DiscordAttachmentSyncer {
	readonly root: string;
	private readonly client: DiscordAttachmentClient;
	private readonly guildId: string | undefined;
	private readonly channelIds: ReadonlySet<string>;
	private readonly channelNames: ReadonlySet<string>;
	private readonly now: () => number;

	constructor(options: DiscordAttachmentStoreOptions) {
		this.client = options.client;
		this.root = options.root;
		this.guildId = options.guildId;
		this.channelIds = new Set(options.channelIds ?? []);
		this.channelNames = new Set(options.channelNames ?? []);
		this.now = options.now ?? Date.now;
	}

	get statusPath(): string {
		return join(this.root, DISCORD_ATTACHMENT_STATUS_FILENAME);
	}

	async sync(options: DiscordAttachmentSyncOptions): Promise<DiscordAttachmentSyncResult> {
		mkdirSync(this.root, { recursive: true });
		let lock: FileLockHandle;
		try {
			lock = acquireFileLock(join(this.root, LOCK_DIRNAME), {
				timeoutMs: 0,
				staleMs: 60_000,
				timeoutError: () => new Error("busy"),
			});
		} catch {
			return { ok: false, reason: "busy", message: "another AutoRAG process is updating this attachments folder" };
		}
		try {
			return await this.syncLocked(options);
		} finally {
			lock.release();
		}
	}

	private async syncLocked(options: DiscordAttachmentSyncOptions): Promise<DiscordAttachmentSyncResult> {
		const { signal } = options;
		const status = await this.client.status(signal);
		if (!status.ok) return discrawlFailure("discrawl status", status.reason, status.stderr);
		if (status.data.databasePath === undefined) {
			return { ok: false, reason: "discrawl-failed", message: "discrawl status did not report its archive path" };
		}
		const configPath = this.writeDiscrawlConfig(status.data.databasePath);
		const listed = await this.client.listAttachments(configPath, this.guildId, signal);
		if (!listed.ok) return discrawlFailure("discrawl attachments", listed.reason, listed.stderr);

		const previous = readStatusFile(this.statusPath);
		const records = new Map<string, TrackedRecord>();
		for (const row of listed.data) {
			if (!this.channelAllowed(row)) continue;
			records.set(row.attachmentId, await this.verify(row, previous.get(row.attachmentId)));
		}

		let run: DiscordAttachmentRun = { attempted: 0, fetched: 0, failed: 0, skipped: 0, bytes: 0 };
		if (options.download) {
			const targets = [...records.values()].filter((record) => this.isDownloadable(record));
			if (targets.length > 0) run = await this.download(configPath, targets, records, signal);
		}

		const finalRecords: DiscordAttachmentRecord[] = [...records.values()].map((record) => ({
			...record,
			downloadable: this.isDownloadable(record),
		}));
		const summary = summarize(finalRecords);
		const document = {
			version: STATUS_VERSION,
			updatedAt: new Date(this.now()).toISOString(),
			summary,
			attachments: finalRecords,
		};
		writeFileAtomic(this.statusPath, `${JSON.stringify(document, null, 2)}\n`);
		return { ok: true, root: this.root, statusPath: this.statusPath, summary, run, records: finalRecords };
	}

	/** discrawl keeps reading the operator's archive but stores downloads in this folder's private cache. */
	private writeDiscrawlConfig(databasePath: string): string {
		const internal = join(this.root, INTERNAL_DIRNAME);
		const configPath = join(internal, "config.toml");
		writeFileAtomic(
			configPath,
			[
				"# AutoRAG managed discrawl attachments config v1",
				"version = 1",
				`db_path = ${JSON.stringify(databasePath)}`,
				`cache_dir = ${JSON.stringify(internal)}`,
				`log_dir = ${JSON.stringify(join(internal, "logs"))}`,
				"",
			].join("\n"),
		);
		return configPath;
	}

	private channelAllowed(row: DiscrawlAttachmentRow): boolean {
		if (this.channelIds.size === 0 && this.channelNames.size === 0) return true;
		return (
			this.channelIds.has(row.channelId) || (row.channelName !== undefined && this.channelNames.has(row.channelName))
		);
	}

	private isDownloadable(record: TrackedRecord): boolean {
		if (record.state === "downloaded" || record.state === "skipped") return false;
		if (record.linkExpiresAt === undefined) return true;
		return Date.parse(record.linkExpiresAt) - LINK_EXPIRY_MARGIN_MS > this.now();
	}

	/** Re-checks a saved file. Unchanged size and mtime is trusted; anything else is hashed. */
	private async verify(row: DiscrawlAttachmentRow, previous: SavedEntry | undefined): Promise<TrackedRecord> {
		const base = recordFromRow(row);
		if (previous?.path === undefined || previous.sha256 === undefined) {
			if (previous?.state === "failed" || previous?.state === "skipped") {
				return {
					...base,
					state: previous.state,
					...(previous.error === undefined ? {} : { error: previous.error }),
				};
			}
			return { ...base, state: "not-downloaded" };
		}
		const saved = {
			path: previous.path,
			sha256: previous.sha256,
			...(previous.bytes === undefined ? {} : { bytes: previous.bytes }),
			...(previous.downloadedAt === undefined ? {} : { downloadedAt: previous.downloadedAt }),
		};
		const absolute = join(this.root, ...previous.path.split("/"));
		let stats: Stats;
		try {
			stats = lstatSync(absolute);
		} catch {
			return { ...base, ...saved, state: "missing" };
		}
		if (!stats.isFile()) return { ...base, ...saved, state: "missing" };
		if (stats.size === previous.bytes && stats.mtimeMs === previous.mtimeMs) {
			return { ...base, ...saved, mtimeMs: stats.mtimeMs, state: "downloaded" };
		}
		const sha256 = await hashFile(absolute);
		return {
			...base,
			...saved,
			mtimeMs: stats.mtimeMs,
			state: sha256 === previous.sha256 ? "downloaded" : "changed",
		};
	}

	private async download(
		configPath: string,
		targets: readonly TrackedRecord[],
		records: Map<string, TrackedRecord>,
		signal: AbortSignal | undefined,
	): Promise<DiscordAttachmentRun> {
		const startedAt = this.now();
		const fetchErrors = new Map<string, string>();
		for (const messageId of new Set(targets.map((target) => target.messageId))) {
			if (signal?.aborted) break;
			const fetched = await this.client.fetchAttachments(configPath, messageId, signal);
			if (!fetched.ok) fetchErrors.set(messageId, `discrawl attachments fetch failed (${fetched.reason})`);
		}
		const relisted = await this.client.listAttachments(configPath, this.guildId, signal);
		const after = new Map(relisted.ok ? relisted.data.map((row) => [row.attachmentId, row] as const) : []);
		const cacheDir = join(this.root, INTERNAL_DIRNAME);

		let fetched = 0;
		let failed = 0;
		let skipped = 0;
		let bytes = 0;
		const fail = (target: TrackedRecord, error: string): void => {
			// A failed re-download keeps the reason the file needed one, so `changed`/`missing` stay visible.
			const state = target.state === "changed" || target.state === "missing" ? target.state : "failed";
			records.set(target.attachmentId, { ...target, state, error });
			failed += 1;
		};
		for (const target of targets) {
			const row = after.get(target.attachmentId);
			// discrawl reports a file it finds intact in its cache as `reused` without restamping the row, so a
			// file left behind by an interrupted run is picked up whatever its timestamp says.
			const cached =
				row?.fetchStatus === "fetched" && row.mediaPath !== undefined
					? cachedMediaPath(cacheDir, row.mediaPath)
					: undefined;
			if (row !== undefined && cached !== undefined && isFile(cached)) {
				const saved = await this.place(target, cached);
				records.set(target.attachmentId, saved);
				fetched += 1;
				bytes += saved.bytes ?? 0;
				continue;
			}
			const attemptedNow =
				row?.fetchStatus !== undefined &&
				row.fetchedAt !== undefined &&
				Date.parse(row.fetchedAt) >= startedAt - CLOCK_SKEW_MS;
			if (row === undefined || !attemptedNow) {
				fail(target, fetchErrors.get(target.messageId) ?? "discrawl did not attempt this attachment");
			} else if (row.fetchStatus === "fetched") {
				// discrawl stores identical bytes under one cache file, so a sibling attachment may have moved it.
				const twin = [...records.values()].find(
					(record) => record.state === "downloaded" && record.sha256 === row.contentSha256 && record.path,
				);
				if (twin?.path === undefined) {
					fail(target, "discrawl reported the file but it was not in its cache");
					continue;
				}
				const saved = await this.place(target, join(this.root, ...twin.path.split("/")), "copy");
				records.set(target.attachmentId, saved);
				fetched += 1;
				bytes += saved.bytes ?? 0;
			} else if (row.fetchStatus === "failed") {
				fail(target, row.fetchError ?? "download failed");
			} else {
				records.set(target.attachmentId, { ...target, state: "skipped", error: row.fetchStatus });
				skipped += 1;
			}
		}
		// Every file this run needs has been moved out. What is left are duplicates discrawl fetched for
		// attachments already saved (`--missing` checks its own cache, which AutoRAG empties).
		rmSync(join(cacheDir, "media"), { recursive: true, force: true });
		return { attempted: targets.length, fetched, failed, skipped, bytes };
	}

	/** Puts a downloaded file at `<server>/<channel>/<name>`, never over an existing file. */
	private async place(target: TrackedRecord, source: string, mode: "move" | "copy" = "move"): Promise<TrackedRecord> {
		const relativeDir = posix.join(
			safeFilename(target.guildId === "@me" ? "DM" : target.guildName || target.guildId, target.guildId),
			safeFilename(target.channelName || target.channelId, target.channelId),
		);
		const directory = join(this.root, ...relativeDir.split("/"));
		mkdirSync(directory, { recursive: true });
		const name = freeName(directory, safeFilename(target.filename, target.attachmentId));
		const destination = join(directory, name);
		if (mode === "move") renameSync(source, destination);
		else copyFileSync(source, destination);
		const stats = statSync(destination);
		const { error: _cleared, ...rest } = target;
		return {
			...rest,
			state: "downloaded",
			path: posix.join(relativeDir, name),
			sha256: await hashFile(destination),
			bytes: stats.size,
			mtimeMs: stats.mtimeMs,
			downloadedAt: new Date(this.now()).toISOString(),
		};
	}
}

function discrawlFailure(step: string, reason: string, stderr: string): DiscordAttachmentSyncFail {
	const detail = stderr.trim().length > 0 ? `: ${stderr.trim().slice(0, 500)}` : "";
	return { ok: false, reason: "discrawl-failed", message: `${step} failed (${reason})${detail}` };
}

function recordFromRow(row: DiscrawlAttachmentRow): Omit<TrackedRecord, "state"> {
	const linkExpiresAt = signedLinkExpiry(row.url) ?? signedLinkExpiry(row.proxyUrl);
	return {
		attachmentId: row.attachmentId,
		messageId: row.messageId,
		guildId: row.guildId,
		...(row.guildName === undefined ? {} : { guildName: row.guildName }),
		channelId: row.channelId,
		...(row.channelName === undefined ? {} : { channelName: row.channelName }),
		...(row.authorName === undefined ? {} : { authorName: row.authorName }),
		...(row.createdAt === undefined ? {} : { createdAt: row.createdAt }),
		filename: row.filename,
		...(row.contentType === undefined ? {} : { contentType: row.contentType }),
		size: row.size,
		...(linkExpiresAt === undefined ? {} : { linkExpiresAt }),
	};
}

/** Discord signs attachment links with `ex=<hex unix seconds>`. */
export function signedLinkExpiry(url: string | undefined): string | undefined {
	if (url === undefined) return undefined;
	let ex: string | null;
	try {
		ex = new URL(url).searchParams.get("ex");
	} catch {
		return undefined;
	}
	if (ex === null || !/^[0-9a-f]+$/i.test(ex)) return undefined;
	return new Date(Number.parseInt(ex, 16) * 1000).toISOString();
}

function summarize(records: readonly DiscordAttachmentRecord[]): DiscordAttachmentSummary {
	const count = (state: DiscordAttachmentState): number => records.filter((record) => record.state === state).length;
	return {
		total: records.length,
		downloaded: count("downloaded"),
		notDownloaded: count("not-downloaded"),
		changed: count("changed"),
		missing: count("missing"),
		failed: count("failed"),
		skipped: count("skipped"),
		downloadable: records.filter((record) => record.downloadable).length,
		expired: records.filter(
			(record) => record.state !== "downloaded" && record.state !== "skipped" && !record.downloadable,
		).length,
	};
}

/** Reads only what a later run relies on; anything malformed is ignored and simply re-verified. */
function readStatusFile(path: string): Map<string, SavedEntry> {
	const out = new Map<string, SavedEntry>();
	let parsed: unknown;
	try {
		parsed = JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return out;
	}
	const attachments = (parsed as { attachments?: unknown } | null)?.attachments;
	if (!Array.isArray(attachments)) return out;
	for (const entry of attachments) {
		if (entry === null || typeof entry !== "object") continue;
		const record = entry as Record<string, unknown>;
		const { attachmentId, state } = record;
		if (typeof attachmentId !== "string" || typeof state !== "string" || !Object.hasOwn(STATES, state)) continue;
		const savedPath = typeof record.path === "string" && isContainedRelative(record.path) ? record.path : undefined;
		out.set(attachmentId, {
			state: state as DiscordAttachmentState,
			...(savedPath === undefined ? {} : { path: savedPath }),
			...(typeof record.sha256 === "string" ? { sha256: record.sha256 } : {}),
			...(typeof record.bytes === "number" ? { bytes: record.bytes } : {}),
			...(typeof record.mtimeMs === "number" ? { mtimeMs: record.mtimeMs } : {}),
			...(typeof record.downloadedAt === "string" ? { downloadedAt: record.downloadedAt } : {}),
			...(typeof record.error === "string" ? { error: record.error } : {}),
		});
	}
	return out;
}

/** A recorded path must stay inside the folder; a hand-edited `../` entry is ignored. */
function isContainedRelative(path: string): boolean {
	if (path.length === 0 || path.startsWith("/") || /^[A-Za-z]:/.test(path) || path.includes("\\")) return false;
	return path.split("/").every((segment) => segment.length > 0 && segment !== "." && segment !== "..");
}

/** discrawl's `media_path` is relative to `<cache_dir>/media`; anything escaping it is rejected. */
function cachedMediaPath(cacheDir: string, mediaPath: string): string | undefined {
	return isContainedRelative(mediaPath) ? join(cacheDir, "media", ...mediaPath.split("/")) : undefined;
}

function isFile(path: string): boolean {
	try {
		return lstatSync(path).isFile();
	} catch {
		return false;
	}
}

/** The original Discord file name, made safe for macOS, Linux, and Windows. */
export function safeFilename(name: string, attachmentId: string): string {
	let cleaned = name
		.normalize("NFC")
		.replace(/[\p{Cc}<>:"/\\|?*]+/gu, "_")
		.replace(/^[\s.]+|[\s.]+$/g, "");
	if (cleaned.length === 0) return `attachment-${attachmentId}`;
	if (/^(con|prn|aux|nul|com\d|lpt\d)(\..*)?$/i.test(cleaned)) cleaned = `_${cleaned}`;
	if (cleaned.length <= MAX_FILENAME_CHARS) return cleaned;
	const extension = extname(cleaned).slice(0, 20);
	return `${cleaned.slice(0, MAX_FILENAME_CHARS - extension.length)}${extension}`;
}

/** `name`, or `name (2)`, `name (3)`, ... when something already sits at that path. */
function freeName(directory: string, name: string): string {
	const extension = extname(name);
	const stem = name.slice(0, name.length - extension.length);
	for (let index = 1; ; index += 1) {
		const candidate = index === 1 ? name : `${stem} (${index})${extension}`;
		if (!pathExists(join(directory, candidate))) return candidate;
	}
}

function pathExists(path: string): boolean {
	try {
		lstatSync(path);
		return true;
	} catch {
		return false;
	}
}

async function hashFile(path: string): Promise<string> {
	const hash = createHash("sha256");
	for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
	return hash.digest("hex");
}
