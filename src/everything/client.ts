import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { mkdirSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { setTimeout as sleep } from "node:timers/promises";
import { type EnsureEverythingBinariesResult, ensureEverythingBinaries } from "./bundle.ts";

/**
 * Drives a private, user-level Everything instance per AutoRAG workspace.
 *
 * Permission model: the instance indexes only the configured search folders
 * (Everything "folder indexing"), which needs no administrator rights. It never
 * requests elevation, never installs the Everything service, never indexes
 * whole NTFS/ReFS volumes, and never enables the HTTP/ETP servers. Settings and
 * the database live under `<workspace>/.autorag/everything/`, and the named
 * instance keeps it isolated from any Everything the user already runs.
 *
 * Search goes through the bundled ES CLI over Everything's IPC window.
 */

export interface EverythingEntry {
	readonly path: string;
	readonly type: "file" | "folder";
	readonly size: number | undefined;
	readonly dateModified: string | undefined;
}

export type EverythingSort =
	| "name-ascending"
	| "name-descending"
	| "path-ascending"
	| "path-descending"
	| "size-ascending"
	| "size-descending"
	| "date-modified-ascending"
	| "date-modified-descending";

export interface EverythingSearchRequest {
	/** Everything search syntax (wildcards, `ext:`, `dm:`, `size:`, `|`, `!`, quotes for phrases). */
	readonly query: string;
	readonly regex?: boolean;
	readonly matchCase?: boolean;
	readonly matchPath?: boolean;
	readonly wholeWord?: boolean;
	readonly kind?: "files" | "folders";
	/** Restrict to items under this absolute folder. */
	readonly path?: string;
	readonly sort?: EverythingSort;
	readonly offset?: number;
	readonly maxResults?: number;
}

export type EverythingFailureReason =
	| "unsupported-platform"
	| "bundle-missing"
	| "install-failed"
	| "not-running"
	| "startup-failed"
	| "index-failed"
	| "search-failed";

export interface EverythingFailure {
	readonly ok: false;
	readonly reason: EverythingFailureReason;
	readonly message: string;
}

export type EverythingSearchResult =
	| { readonly ok: true; readonly results: readonly EverythingEntry[] }
	| EverythingFailure;

export type EverythingIndexResult =
	| { readonly ok: true; readonly indexedFolders: readonly string[]; readonly indexedItems: number }
	| EverythingFailure;

export interface EverythingRunResult {
	readonly code: number | null;
	readonly stdout: string;
	readonly stderr: string;
}

export type EverythingRunner = (
	command: string,
	args: readonly string[],
	timeoutMs: number,
) => Promise<EverythingRunResult>;
export type EverythingLauncher = (command: string, args: readonly string[]) => void;

export interface EverythingOptions {
	/** Set false to disable Everything on Windows. */
	readonly enabled?: boolean;
	/** Per ES invocation timeout. Default 30s. */
	readonly timeoutMs?: number;
	/** How long to wait for a freshly launched instance to answer IPC. Default 20s. */
	readonly startupTimeoutMs?: number;
	/** How long indexing waits for the database to finish loading. Default 10 minutes. */
	readonly indexTimeoutMs?: number;
}

export interface EverythingClientOptions extends EverythingOptions {
	/** Workspace root; state lives under `<root>/.autorag/everything`. */
	readonly root: string;
	/** Folders Everything indexes (the configured search paths). */
	readonly folders: readonly string[];
	readonly platform?: NodeJS.Platform;
	readonly resolveBinaries?: () => Promise<EnsureEverythingBinariesResult>;
	readonly run?: EverythingRunner;
	readonly launch?: EverythingLauncher;
	readonly pollIntervalMs?: number;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_STARTUP_TIMEOUT_MS = 20_000;
const DEFAULT_INDEX_TIMEOUT_MS = 10 * 60_000;
const DEFAULT_POLL_INTERVAL_MS = 250;
const DEFAULT_MAX_RESULTS = 100;
const MAX_OUTPUT_BYTES = 8 * 1024 * 1024;
const VERSION_PATTERN = /^\d+\.\d+\.\d+\.\d+$/;

/** Stable, workspace-scoped Everything instance name. */
export function everythingInstanceName(root: string): string {
	return `autorag-${createHash("sha256").update(resolve(root).toLowerCase()).digest("hex").slice(0, 12)}`;
}

/**
 * Everything.ini that confines indexing to the given folders. List values are
 * double-quoted with `\` escaped, per the Everything INI list syntax.
 */
export function buildEverythingIni(input: {
	readonly folders: readonly string[];
	readonly excludeFolders: readonly string[];
}): string {
	const list = (values: readonly string[]) => values.map((value) => `"${value.replace(/\\/g, "\\\\")}"`).join(",");
	const perFolder = (value: string) => input.folders.map(() => value).join(",");
	return [
		"[Everything]",
		"app_data=0",
		"run_as_admin=0",
		"run_in_background=1",
		"show_tray_icon=0",
		"show_in_taskbar=0",
		"check_for_updates_on_startup=0",
		"beta_updates=0",
		"language=1033",
		"allow_http_server=0",
		"allow_etp_server=0",
		"search_history_enabled=0",
		"run_history_enabled=0",
		"auto_include_fixed_volumes=0",
		"auto_include_removable_volumes=0",
		"auto_include_fixed_refs_volumes=0",
		"auto_include_removable_refs_volumes=0",
		"ntfs_volume_paths=",
		"ntfs_volume_includes=",
		"refs_volume_paths=",
		"refs_volume_includes=",
		"index_size=1",
		"index_date_modified=1",
		`folders=${list(input.folders)}`,
		`folder_monitor_changes=${perFolder("1")}`,
		`folder_rescan_if_full_list=${perFolder("1")}`,
		`folder_update_types=${perFolder("0")}`,
		"exclude_list_enabled=1",
		`exclude_folders=${list(input.excludeFolders)}`,
		"",
	].join("\r\n");
}

/**
 * ES argv for one search. `-argv` makes ES split arguments exactly like
 * CommandLineToArgvW, which is how Node quotes child-process arguments, so a
 * query's own double quotes survive as Everything phrase quotes. The query is
 * placed after `--` so a leading dash is never parsed as an ES switch.
 */
export function buildEverythingSearchArgs(
	instance: string,
	request: EverythingSearchRequest,
	timeoutMs?: number,
): string[] {
	const args = [
		"-argv",
		"-instance",
		instance,
		"-cp",
		"65001",
		"-json",
		"-full-path-and-name",
		"-size",
		"-date-modified",
		"-date-format",
		"1",
	];
	// Without -timeout, ES returns zero results when the instance is still
	// starting or rebuilding its index instead of waiting for a real answer.
	if (timeoutMs !== undefined) args.push("-timeout", String(timeoutMs));
	if (request.matchCase) args.push("-case");
	if (request.matchPath) args.push("-match-path");
	if (request.wholeWord) args.push("-whole-word");
	if (request.kind === "files") args.push("/a-d");
	if (request.kind === "folders") args.push("/ad");
	if (request.path !== undefined) args.push("-path", request.path);
	if (request.sort !== undefined) args.push("-sort", request.sort);
	if (request.offset !== undefined && request.offset > 0) args.push("-offset", String(request.offset));
	args.push("-n", String(request.maxResults ?? DEFAULT_MAX_RESULTS));
	// -regex is a mode switch; the query still goes after -- so a pattern
	// starting with a dash is never parsed as another ES switch.
	if (request.regex) args.push("-regex", "--", request.query);
	else args.push("--", request.query);
	return args;
}

/** Parse `es -json -full-path-and-name -size -date-modified` output. */
export function parseEverythingJson(stdout: string): EverythingEntry[] {
	const text = stdout.trim();
	if (text.length === 0) return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(text);
	} catch {
		throw new Error(`ES did not return JSON: ${text.slice(0, 2000)}`);
	}
	if (!Array.isArray(parsed)) throw new Error(`ES did not return a JSON array: ${text.slice(0, 2000)}`);
	return parsed.map((item) => {
		const record = item as { filename?: unknown; size?: unknown; date_modified?: unknown };
		const filename = String(record.filename ?? "");
		const isFolder = filename.endsWith("\\") && filename.length > 3;
		return {
			path: isFolder ? filename.slice(0, -1) : filename,
			type: filename.endsWith("\\") ? "folder" : "file",
			size: typeof record.size === "number" ? record.size : undefined,
			dateModified:
				typeof record.date_modified === "string" || typeof record.date_modified === "number"
					? String(record.date_modified)
					: undefined,
		};
	});
}

export class EverythingClient {
	readonly instanceName: string;
	private readonly options: EverythingClientOptions;
	private readonly platform: NodeJS.Platform;
	private readonly run: EverythingRunner;
	private readonly launch: EverythingLauncher;
	private readonly stateDir: string;
	private binaries: EnsureEverythingBinariesResult | undefined;
	private operation: Promise<unknown> = Promise.resolve();

	constructor(options: EverythingClientOptions) {
		this.options = options;
		this.platform = options.platform ?? process.platform;
		this.instanceName = everythingInstanceName(options.root);
		this.run = options.run ?? runProcess;
		this.launch = options.launch ?? launchDetached;
		this.stateDir = join(options.root, ".autorag", "everything");
	}

	isSupported(): boolean {
		return this.platform === "win32";
	}

	/**
	 * Search the running instance. Read-only: never starts Everything or waits
	 * for it to index (refresh does that), so a query never pays startup cost.
	 */
	search(request: EverythingSearchRequest): Promise<EverythingSearchResult> {
		return this.serialize(async () => {
			const binaries = await this.resolveBinaries();
			if (!binaries.ok) return binaries;
			const ping = await this.ping(binaries.esPath);
			if (ping.code !== 0) {
				return {
					ok: false,
					reason: "not-running",
					message: `Everything instance ${this.instanceName} is not running; run \`autorag refresh\` to start and index it.`,
				};
			}
			const es = { ok: true as const, esPath: binaries.esPath };
			const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
			const args = buildEverythingSearchArgs(this.instanceName, request, timeoutMs);
			// ES needs a moment beyond its own -timeout to print the error.
			const result = await this.run(es.esPath, args, timeoutMs + 5_000);
			if (result.code !== 0) return this.failure("search-failed", es.esPath, args, result);
			try {
				return { ok: true, results: parseEverythingJson(result.stdout) };
			} catch (error) {
				return { ok: false, reason: "search-failed", message: (error as Error).message };
			}
		});
	}

	/**
	 * (Re)write the folder configuration, restart the instance so Everything
	 * loads it, and wait until the database finished indexing.
	 */
	index(): Promise<EverythingIndexResult> {
		return this.serialize(async () => {
			const es = await this.ensureRunning(true);
			if (!es.ok) return es;
			const timeoutMs = this.options.indexTimeoutMs ?? DEFAULT_INDEX_TIMEOUT_MS;
			const saveArgs = ["-instance", this.instanceName, "-save-db"];
			const saved = await this.run(es.esPath, saveArgs, timeoutMs);
			if (saved.code !== 0) return this.failure("index-failed", es.esPath, saveArgs, saved);
			const countArgs = ["-instance", this.instanceName, "-get-result-count", "-timeout", String(timeoutMs), "*"];
			const counted = await this.run(es.esPath, countArgs, timeoutMs + 5_000);
			if (counted.code !== 0) return this.failure("index-failed", es.esPath, countArgs, counted);
			return {
				ok: true,
				indexedFolders: [...this.options.folders],
				indexedItems: Number.parseInt(counted.stdout.trim(), 10) || 0,
			};
		});
	}

	/** Exit this workspace's Everything instance if it is running. */
	stop(): Promise<void> {
		return this.serialize(async () => {
			const binaries = await this.resolveBinaries();
			if (!binaries.ok) return;
			await this.run(
				binaries.esPath,
				["-instance", this.instanceName, "-exit"],
				this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
			);
		});
	}

	private serialize<T>(task: () => Promise<T>): Promise<T> {
		const next = this.operation.then(task, task);
		this.operation = next.catch(() => undefined);
		return next;
	}

	private async resolveBinaries(): Promise<EnsureEverythingBinariesResult> {
		if (!this.isSupported()) {
			return {
				ok: false,
				reason: "unsupported-platform",
				message: `Everything is Windows-only; this host is ${this.platform}.`,
			};
		}
		if (this.binaries?.ok) return this.binaries;
		this.binaries = await (
			this.options.resolveBinaries ?? (() => ensureEverythingBinaries({ root: this.options.root }))
		)();
		return this.binaries;
	}

	private async ensureRunning(
		reconfigure: boolean,
	): Promise<{ readonly ok: true; readonly esPath: string } | EverythingFailure> {
		const binaries = await this.resolveBinaries();
		if (!binaries.ok) return binaries;
		const timeoutMs = this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
		let ping = await this.ping(binaries.esPath);
		if (ping.code === 0 && !reconfigure) return { ok: true, esPath: binaries.esPath };
		if (ping.code === 0) {
			// Everything rewrites its ini on exit, so stop it before writing ours.
			const exitArgs = ["-instance", this.instanceName, "-exit"];
			const exited = await this.run(binaries.esPath, exitArgs, timeoutMs);
			if (exited.code !== 0) return this.failure("startup-failed", binaries.esPath, exitArgs, exited);
		}
		mkdirSync(this.stateDir, { recursive: true });
		const iniPath = join(this.stateDir, "Everything.ini");
		writeFileSync(
			iniPath,
			buildEverythingIni({
				folders: this.options.folders,
				excludeFolders: [
					...new Set([
						join(this.options.root, ".autorag"),
						...this.options.folders.map((folder) => join(folder, ".autorag")),
					]),
				],
			}),
		);
		const launchArgs = [
			"-instance",
			this.instanceName,
			"-config",
			iniPath,
			"-db",
			join(this.stateDir, "Everything.db"),
			"-startup",
		];
		try {
			this.launch(binaries.everythingPath, launchArgs);
		} catch (error) {
			return {
				ok: false,
				reason: "startup-failed",
				message: `${binaries.everythingPath} ${launchArgs.join(" ")} failed to start: ${(error as Error).message}`,
			};
		}
		const deadline = Date.now() + (this.options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS);
		const pollMs = this.options.pollIntervalMs ?? DEFAULT_POLL_INTERVAL_MS;
		for (;;) {
			ping = await this.ping(binaries.esPath);
			if (ping.code === 0) return { ok: true, esPath: binaries.esPath };
			if (Date.now() >= deadline) break;
			await sleep(pollMs);
		}
		return {
			ok: false,
			reason: "startup-failed",
			message: `Everything instance ${this.instanceName} (${binaries.everythingPath}) did not answer ES IPC within ${this.options.startupTimeoutMs ?? DEFAULT_STARTUP_TIMEOUT_MS}ms; last ES exit ${ping.code}: ${(ping.stderr || ping.stdout).trim()}`,
		};
	}

	/** Exit code 0 only when the instance answered with its version. */
	private async ping(esPath: string): Promise<EverythingRunResult> {
		const result = await this.run(
			esPath,
			["-instance", this.instanceName, "-get-everything-version"],
			this.options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
		);
		return result.code === 0 && !VERSION_PATTERN.test(result.stdout.trim()) ? { ...result, code: -1 } : result;
	}

	private failure(
		reason: EverythingFailureReason,
		esPath: string,
		args: readonly string[],
		result: EverythingRunResult,
	): EverythingFailure {
		return {
			ok: false,
			reason,
			message: `${esPath} ${args.join(" ")} exit ${result.code}: ${(result.stderr || result.stdout).trim()}`,
		};
	}
}

function runProcess(command: string, args: readonly string[], timeoutMs: number): Promise<EverythingRunResult> {
	// tsconfig lib is ES2022, which predates Promise.withResolvers.
	let resolvePromise!: (result: EverythingRunResult) => void;
	const promise = new Promise<EverythingRunResult>((resolve) => {
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

function launchDetached(command: string, args: readonly string[]): void {
	const child = spawn(command, args, { detached: true, stdio: "ignore", windowsHide: true });
	child.on("error", () => {
		// The startup poll reports the instance as unreachable with ES's error.
	});
	child.unref();
}
