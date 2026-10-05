import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { basename, join, resolve, sep } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { walkFileSearch } from "./walk.ts";

/**
 * Drives the user's fsearch-cli (FSearch) as a separate process per AutoRAG
 * workspace.
 *
 * License model: FSearch and fsearch-cli are GPL-2.0-or-later and are NOT
 * bundled with AutoRAG. The user installs fsearch-cli separately
 * (`brew install NomaDamas/tap/fsearch-cli`, or the Linux build from
 * https://github.com/NomaDamas/fsearch-mac) and AutoRAG only spawns it,
 * communicating over its CLI and daemon socket — mere aggregation per the
 * FSF FAQ, never a combined work.
 *
 * Permission model: the database indexes only the configured search folders
 * and lives under `<workspace>/.autorag/fsearch/`, isolated from any FSearch
 * app database the user keeps. A `fsearch-cli watch` daemon per workspace
 * keeps the index live from FSEvents (macOS) or inotify/fanotify (Linux) and
 * serves searches over its socket; without the daemon, `fsearch-cli search`
 * loads the database file directly. When fsearch-cli is not installed,
 * searches degrade to a bounded slow filesystem walk over the same folders.
 */

export interface FSearchEntry {
	readonly path: string;
	readonly name: string;
	readonly type: "file" | "folder";
	readonly size: number | undefined;
	readonly dateModified: string | undefined;
}

export type FSearchSort =
	| "name-ascending"
	| "name-descending"
	| "path-ascending"
	| "path-descending"
	| "size-ascending"
	| "size-descending"
	| "date-modified-ascending"
	| "date-modified-descending";

export interface FSearchSearchRequest {
	/** FSearch query syntax (wildcards, AND/OR/NOT, `ext:`, `size:`, `path:`). */
	readonly query: string;
	readonly regex?: boolean;
	readonly matchCase?: boolean;
	readonly matchPath?: boolean;
	readonly kind?: "files" | "folders";
	/** Restrict to items under this absolute folder (client-side prefix filter). */
	readonly path?: string;
	readonly sort?: FSearchSort;
	readonly offset?: number;
	readonly maxResults?: number;
}

export type FSearchFailureReason = "unsupported-platform" | "binary-missing" | "index-failed" | "search-failed";

export interface FSearchFailure {
	readonly ok: false;
	readonly reason: FSearchFailureReason;
	readonly message: string;
}

/** Which backend answered a search: the fsearch-cli index or the slow walk. */
export type FSearchBackend = "fsearch-cli" | "walk";

export type FSearchSearchResult =
	| {
			readonly ok: true;
			readonly backend: FSearchBackend;
			readonly results: readonly FSearchEntry[];
			/** Total matches reported by fsearch-cli (may exceed results length). */
			readonly total?: number;
			readonly truncated?: boolean;
			/** Degradation note, e.g. why the slow walk answered instead. */
			readonly note?: string;
	  }
	| FSearchFailure;

export type FSearchIndexResult =
	| {
			readonly ok: true;
			readonly indexedFolders: readonly string[];
			readonly indexedItems: number;
			/** Whether the live watch daemon serves the database after indexing. */
			readonly watchLive?: boolean;
	  }
	| FSearchFailure;

export interface FSearchRunResult {
	readonly code: number | null;
	readonly stdout: string;
	readonly stderr: string;
}

export type FSearchRunner = (command: string, args: readonly string[], timeoutMs: number) => Promise<FSearchRunResult>;
/** Launches a detached daemon; returns its pid when known. */
export type FSearchLauncher = (command: string, args: readonly string[]) => number | undefined;

export interface FSearchOptions {
	/** Set false to disable FSearch on macOS/Linux. */
	readonly enabled?: boolean;
	/** fsearch-cli binary to spawn; defaults to PATH lookup. */
	readonly binaryPath?: string;
	/** Per fsearch-cli invocation timeout. Default 30s. */
	readonly timeoutMs?: number;
	/** How long indexing waits for the watch daemon to serve. Default 10s. */
	readonly startupTimeoutMs?: number;
	/** How long an index run may take. Default 10 minutes. */
	readonly indexTimeoutMs?: number;
	/** Keep a `fsearch-cli watch` daemon live after indexing. Default true. */
	readonly watch?: boolean;
}

export interface FSearchClientOptions extends FSearchOptions {
	/** Workspace root; state lives under `<root>/.autorag/fsearch`. */
	readonly root: string;
	/** Folders FSearch indexes (the configured search paths). */
	readonly folders: readonly string[];
	/** Additional configured folders omitted from the index. */
	readonly excludeFolders?: readonly string[];
	readonly platform?: NodeJS.Platform;
	readonly run?: FSearchRunner;
	readonly launch?: FSearchLauncher;
	readonly isProcessAlive?: (pid: number) => boolean;
	readonly killProcess?: (pid: number) => void;
	readonly pollIntervalMs?: number;
	/** Slow-walk bounds used when fsearch-cli is missing. */
	readonly walkMaxVisited?: number;
	readonly walkDeadlineMs?: number;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_STARTUP_TIMEOUT_MS = 10_000;
const STOP_TIMEOUT_MS = 2_000;
const DEFAULT_INDEX_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_POLL_INTERVAL_MS = 250;
const DEFAULT_MAX_RESULTS = 100;
const MAX_RESULTS_CAP = 1000;
const MAX_OUTPUT_BYTES = 8 * 1024 * 1024;

const SORT_KEYS: Record<FSearchSort, readonly [key: string, descending: boolean]> = {
	"name-ascending": ["name", false],
	"name-descending": ["name", true],
	"path-ascending": ["path", false],
	"path-descending": ["path", true],
	"size-ascending": ["size", false],
	"size-descending": ["size", true],
	"date-modified-ascending": ["mtime", false],
	"date-modified-descending": ["mtime", true],
};

/**
 * Short, stable daemon socket path for a database. fsearch-cli's default
 * `<db>.sock` overflows the 104-byte unix sun_path limit for deep workspace
 * paths — the daemon then binds a truncated name its own clients cannot
 * derive, and daemon/client rendezvous silently breaks (observed on macOS
 * with `/private/var/folders/...` workspaces). An explicit short path keyed
 * by the database's realpath keeps the daemon, stats, and search rendezvous
 * consistent regardless of which symlinked spelling a caller used.
 */
export function fsearchSocketPath(dbPath: string): string {
	let canonical: string;
	try {
		canonical = realpathSync(dbPath);
	} catch {
		canonical = resolve(dbPath);
	}
	const hash = createHash("sha256").update(canonical).digest("hex").slice(0, 12);
	return `/tmp/autorag-fsearch-${process.getuid?.() ?? 0}-${hash}.sock`;
}

/**
 * fsearch-cli argv for one search. The query goes after `--` so a leading
 * dash is never parsed as a switch. fsearch-cli has no offset flag, so the
 * caller fetches offset+maxResults (or the cap when a path filter needs
 * client-side prefix filtering) and pages locally.
 */
export function buildFSearchSearchArgs(dbPath: string, socketPath: string, request: FSearchSearchRequest): string[] {
	const args = ["search", "--db", dbPath, "--socket", socketPath, "--json"];
	if (request.regex) args.push("-r");
	if (request.matchCase) args.push("-c");
	if (request.matchPath) args.push("-p");
	if (request.kind === "files") args.push("-f");
	if (request.kind === "folders") args.push("-F");
	if (request.sort !== undefined) {
		const [key, descending] = SORT_KEYS[request.sort];
		args.push("--sort", key);
		if (descending) args.push("--desc");
	}
	const limit =
		request.path !== undefined
			? MAX_RESULTS_CAP
			: Math.min((request.offset ?? 0) + (request.maxResults ?? DEFAULT_MAX_RESULTS), MAX_RESULTS_CAP);
	args.push("--limit", String(limit), "--", request.query);
	return args;
}

/** fsearch-cli argv for one index build over the given folders. */
export function buildFSearchIndexArgs(
	dbPath: string,
	folders: readonly string[],
	excludeFolders: readonly string[],
): string[] {
	const args = ["index", "--db", dbPath];
	for (const folder of folders) args.push("--include", folder);
	for (const folder of excludeFolders) args.push("--exclude-path", folder);
	return args;
}

/** Parse `fsearch-cli search --json` JSON-lines output, dropping the trailing done line. */
export function parseFSearchJsonLines(stdout: string): { entries: FSearchEntry[]; total: number | undefined } {
	const entries: FSearchEntry[] = [];
	let total: number | undefined;
	for (const line of stdout.split("\n")) {
		const text = line.trim();
		if (text.length === 0) continue;
		let record: {
			done?: unknown;
			num_results?: unknown;
			path?: unknown;
			type?: unknown;
			size?: unknown;
			mtime?: unknown;
		};
		try {
			record = JSON.parse(text);
		} catch {
			throw new Error(`fsearch-cli did not return JSON lines: ${text.slice(0, 2000)}`);
		}
		if (record.done === true) {
			if (typeof record.num_results === "number") total = record.num_results;
			continue;
		}
		const path = String(record.path ?? "");
		entries.push({
			path,
			name: basename(path),
			type: record.type === "folder" ? "folder" : "file",
			size: typeof record.size === "number" ? record.size : undefined,
			dateModified: typeof record.mtime === "number" ? new Date(record.mtime * 1000).toISOString() : undefined,
		});
	}
	return { entries, total };
}

/** Parse `fsearch-cli stats` JSON output. */
export function parseFSearchStats(stdout: string): { files: number; folders: number; live: boolean } {
	let record: { files?: unknown; folders?: unknown; live?: unknown };
	try {
		record = JSON.parse(stdout.trim());
	} catch {
		throw new Error(`fsearch-cli stats did not return JSON: ${stdout.trim().slice(0, 2000)}`);
	}
	return {
		files: typeof record.files === "number" ? record.files : 0,
		folders: typeof record.folders === "number" ? record.folders : 0,
		live: record.live === true,
	};
}

export class FSearchClient {
	private readonly options: FSearchClientOptions;
	private readonly platform: NodeJS.Platform;
	private readonly run: FSearchRunner;
	private readonly launch: FSearchLauncher;
	private readonly isAlive: (pid: number) => boolean;
	private readonly kill: (pid: number) => void;
	private readonly stateDir: string;
	private readonly dbPath: string;
	private readonly pidPath: string;
	private binary: { readonly ok: true; readonly path: string } | FSearchFailure | undefined;
	private operation: Promise<unknown> = Promise.resolve();

	constructor(options: FSearchClientOptions) {
		this.options = options;
		this.platform = options.platform ?? process.platform;
		this.run = options.run ?? runProcess;
		this.launch = options.launch ?? launchDetached;
		this.isAlive = options.isProcessAlive ?? defaultIsProcessAlive;
		this.kill = options.killProcess ?? ((pid) => process.kill(pid, "SIGTERM"));
		this.stateDir = join(options.root, ".autorag", "fsearch");
		this.dbPath = join(this.stateDir, "fsearch.db");
		this.pidPath = join(this.stateDir, "watch.pid");
	}

	isSupported(): boolean {
		return this.platform === "darwin" || this.platform === "linux";
	}

	/** Search the workspace database, building it on first use; walks when fsearch-cli is missing. */
	search(request: FSearchSearchRequest): Promise<FSearchSearchResult> {
		return this.serialize(async () => {
			const unsupported = this.unsupportedFailure();
			if (unsupported !== undefined) return unsupported;
			const binary = await this.resolveBinary();
			if (!binary.ok) {
				// Issue #1763 graceful fallback: no fsearch-cli → bounded slow walk.
				try {
					const walked = await walkFileSearch(this.options.folders, request, {
						maxVisited: this.options.walkMaxVisited,
						deadlineMs: this.options.walkDeadlineMs,
					});
					return {
						ok: true,
						backend: "walk",
						results: walked.entries,
						truncated: walked.truncated || undefined,
						note: `fsearch-cli is unavailable (${binary.message}); answered with a slow filesystem walk over the configured search folders${walked.truncated ? " (truncated)" : ""}.`,
					};
				} catch (error) {
					return { ok: false, reason: "search-failed", message: (error as Error).message };
				}
			}
			if (!existsSync(this.dbPath)) {
				const indexed = await this.indexLocked(binary.path);
				if (!indexed.ok) return indexed;
			}
			const args = buildFSearchSearchArgs(this.dbPath, this.socketPath(), request);
			const result = await this.run(binary.path, args, this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
			if (result.code !== 0) return this.failure("search-failed", binary.path, args, result);
			let parsed: { entries: FSearchEntry[]; total: number | undefined };
			try {
				parsed = parseFSearchJsonLines(result.stdout);
			} catch (error) {
				return { ok: false, reason: "search-failed", message: (error as Error).message };
			}
			let entries = parsed.entries;
			if (request.path !== undefined) {
				const prefix = request.path.endsWith(sep) ? request.path : `${request.path}${sep}`;
				entries = entries.filter((entry) => entry.path === request.path || entry.path.startsWith(prefix));
			}
			if (request.offset !== undefined && request.offset > 0) entries = entries.slice(request.offset);
			const maxResults = Math.min(request.maxResults ?? DEFAULT_MAX_RESULTS, MAX_RESULTS_CAP);
			if (entries.length > maxResults) entries = entries.slice(0, maxResults);
			return { ok: true, backend: "fsearch-cli", results: entries, total: parsed.total };
		});
	}

	/** (Re)build the database over the configured folders and keep the watch daemon live. */
	index(): Promise<FSearchIndexResult> {
		return this.serialize(async () => {
			const unsupported = this.unsupportedFailure();
			if (unsupported !== undefined) return unsupported;
			const binary = await this.resolveBinary();
			if (!binary.ok) return binary;
			return this.indexLocked(binary.path);
		});
	}

	/** Terminate this workspace's watch daemon when it is running. No-op otherwise. */
	stop(): Promise<void> {
		return this.serialize(async () => {
			const pid = this.readWatchPid();
			if (pid !== undefined && this.isAlive(pid)) this.kill(pid);
			rmSync(this.pidPath, { force: true });
		});
	}

	private async indexLocked(binaryPath: string): Promise<FSearchIndexResult> {
		mkdirSync(this.stateDir, { recursive: true });
		const indexArgs = buildFSearchIndexArgs(this.dbPath, this.options.folders, [
			...new Set([
				join(this.options.root, ".autorag"),
				...this.options.folders.map((folder) => join(folder, ".autorag")),
				...(this.options.excludeFolders ?? []),
			]),
		]);
		const indexed = await this.run(binaryPath, indexArgs, this.options.indexTimeoutMs ?? DEFAULT_INDEX_TIMEOUT_MS);
		if (indexed.code !== 0) return this.failure("index-failed", binaryPath, indexArgs, indexed);
		const statsArgs = ["stats", "--db", this.dbPath, "--socket", this.socketPath()];
		const stats = await this.run(binaryPath, statsArgs, this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		if (stats.code !== 0) return this.failure("index-failed", binaryPath, statsArgs, stats);
		let counts: { files: number; folders: number; live: boolean };
		try {
			counts = parseFSearchStats(stats.stdout);
		} catch (error) {
			return { ok: false, reason: "index-failed", message: (error as Error).message };
		}
		const watchLive = this.options.watch === false ? undefined : await this.ensureWatch(binaryPath);
		return {
			ok: true,
			indexedFolders: [...this.options.folders],
			indexedItems: counts.files + counts.folders,
			watchLive,
		};
	}

	/**
	 * Restart the live watch daemon against the freshly indexed database.
	 * Index-time always recycles the tracked daemon (Everything parity: a
	 * config-changed index must be the one served); a daemon that serves the
	 * socket without a pid file is adopted, not replaced, because it cannot be
	 * signaled. Searches work without any daemon (fsearch-cli loads the
	 * database file), so a daemon that never serves degrades answer latency,
	 * not correctness.
	 */
	private async ensureWatch(binaryPath: string): Promise<boolean> {
		const pollMs = this.options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
		const tracked = this.readWatchPid();
		if (tracked !== undefined) {
			if (this.isAlive(tracked)) this.kill(tracked);
			rmSync(this.pidPath, { force: true });
			// Wait for the socket to be released so the relaunch does not exit on contention.
			const stopDeadline = Date.now() + STOP_TIMEOUT_MS;
			while ((await this.serving(binaryPath)) && Date.now() < stopDeadline) await sleep(pollMs);
		}
		// A daemon that already serves this database without a pid file (e.g.
		// the workspace was recreated without stopFsearch) outranks a launch:
		// spawning another watch would exit on socket contention anyway.
		if (await this.serving(binaryPath)) return true;
		// Clear a stale socket file from a dead daemon so the bind cannot fail.
		rmSync(this.socketPath(), { force: true });
		const pid = this.launch(binaryPath, ["watch", "--db", this.dbPath, "--socket", this.socketPath(), "--quiet"]);
		if (pid === undefined) return false;
		writeFileSync(this.pidPath, String(pid));
		const deadline = Date.now() + (this.options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS);
		for (;;) {
			if (await this.serving(binaryPath)) return true;
			if (Date.now() >= deadline) {
				// Never point the pid file at a daemon that already exited.
				if (!this.isAlive(pid)) rmSync(this.pidPath, { force: true });
				return false;
			}
			await sleep(pollMs);
		}
	}

	/** True when a watch daemon answers searches for this database over its socket. */
	private async serving(binaryPath: string): Promise<boolean> {
		const stats = await this.run(
			binaryPath,
			["stats", "--db", this.dbPath, "--socket", this.socketPath()],
			this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
		);
		if (stats.code !== 0) return false;
		try {
			return parseFSearchStats(stats.stdout).live;
		} catch {
			return false;
		}
	}

	/** Computed per call: the database realpath only exists after indexing ran. */
	private socketPath(): string {
		return fsearchSocketPath(this.dbPath);
	}

	private readWatchPid(): number | undefined {
		let raw: string;
		try {
			raw = readFileSync(this.pidPath, "utf8");
		} catch {
			return undefined;
		}
		const pid = Number.parseInt(raw.trim(), 10);
		return Number.isInteger(pid) && pid > 0 ? pid : undefined;
	}

	private unsupportedFailure(): FSearchFailure | undefined {
		return this.isSupported()
			? undefined
			: {
					ok: false,
					reason: "unsupported-platform",
					message: `FSearch is macOS/Linux-only; this host is ${this.platform}.`,
				};
	}

	private async resolveBinary(): Promise<{ readonly ok: true; readonly path: string } | FSearchFailure> {
		if (this.binary !== undefined) return this.binary;
		const path = this.options.binaryPath ?? "fsearch-cli";
		const probe = await this.run(path, ["--version"], this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		this.binary =
			probe.code === 0
				? { ok: true, path }
				: {
						ok: false,
						reason: "binary-missing",
						message: `${path} --version exit ${probe.code}: ${(probe.stderr || probe.stdout).trim()}`,
					};
		return this.binary;
	}

	private serialize<T>(task: () => Promise<T>): Promise<T> {
		const next = this.operation.then(task, task);
		this.operation = next.catch(() => undefined);
		return next;
	}

	private failure(
		reason: FSearchFailureReason,
		binaryPath: string,
		args: readonly string[],
		result: FSearchRunResult,
	): FSearchFailure {
		return {
			ok: false,
			reason,
			message: `${binaryPath} ${args.join(" ")} exit ${result.code}: ${(result.stderr || result.stdout).trim()}`,
		};
	}
}

function runProcess(command: string, args: readonly string[], timeoutMs: number): Promise<FSearchRunResult> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	let resolvePromise!: (result: FSearchRunResult) => void;
	const promise = new Promise<FSearchRunResult>((resolve) => {
		resolvePromise = resolve;
	});
	const child = spawn(command, args, { windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
	const stdout: Buffer[] = [];
	const stderr: Buffer[] = [];
	let bytes = 0;
	let overflow = false;
	const collect = (target: Buffer[]) => (chunk: Buffer) => {
		bytes += chunk.length;
		if (bytes > MAX_OUTPUT_BYTES) {
			overflow = true;
			child.kill();
			return;
		}
		target.push(chunk);
	};
	child.stdout.on("data", collect(stdout));
	child.stderr.on("data", collect(stderr));
	const timer = setTimeout(() => child.kill(), timeoutMs);
	child.on("error", (error) => {
		clearTimeout(timer);
		resolvePromise({ code: null, stdout: "", stderr: `${command}: ${error.message}` });
	});
	child.on("close", (code, signal) => {
		clearTimeout(timer);
		const err = Buffer.concat(stderr).toString("utf8");
		resolvePromise({
			code,
			stdout: Buffer.concat(stdout).toString("utf8"),
			stderr: overflow
				? `${err}\noutput exceeded ${MAX_OUTPUT_BYTES} bytes`
				: signal !== null
					? `${err}\nkilled by ${signal} after ${timeoutMs}ms`
					: err,
		});
	});
	return promise;
}

function launchDetached(command: string, args: readonly string[]): number | undefined {
	const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
	child.on("error", () => {
		// The stats poll reports the daemon as not serving; searches still load the database file.
	});
	child.unref();
	return child.pid;
}

function defaultIsProcessAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}
