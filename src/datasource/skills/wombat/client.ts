import { spawn } from "node:child_process";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import { TREE_KILL_GRACE_MS, terminateProcessTree } from "../../../process/terminate-tree.ts";
import { boundDiagnosticText } from "../../connector.ts";
import { DEFAULT_CONNECTOR_SYNC_TIMEOUT_MS } from "../../connector-timeouts.ts";
import type {
	WombatDoctorResult,
	WombatFailure,
	WombatFailureReason,
	WombatOk,
	WombatOptions,
	WombatSearchHit,
	WombatSearchMode,
	WombatSearchOptions,
	WombatSearchResult,
	WombatSyncResult,
} from "./types.ts";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: WombatFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};
type BoundedOutput = {
	readonly text: string;
	readonly bytes: number;
	readonly exceeded: boolean;
};

const DEFAULT_BINARY = "wombat";
const DEFAULT_SEARCH_TIMEOUT_MS = 60_000;
/** `sync --index` encodes embeddings and SPLADE vectors; a first run over years of sessions takes a long time. */
const DEFAULT_INDEX_TIMEOUT_MS = DEFAULT_CONNECTOR_SYNC_TIMEOUT_MS;
const DEFAULT_MAX_BUFFER_BYTES = 8 * 1_048_576;
const SAFE_ENV_KEYS: Readonly<Record<string, true>> = {
	HOME: true,
	USERPROFILE: true,
	LANG: true,
	LC_ALL: true,
	PATH: true,
	TMPDIR: true,
	TMP: true,
	TEMP: true,
	NO_COLOR: true,
};
/** Variables wombat and the MinSync it spawns document; nothing else from the parent environment is forwarded. */
const SAFE_WOMBAT_ENV_KEYS: Readonly<Record<string, true>> = {
	WOMBAT_DATA_DIR: true,
	WOMBAT_CONFIG: true,
	WOMBAT_MINSYNC_BIN: true,
	WOMBAT_EMBEDDER: true,
	WOMBAT_CHUNKER: true,
	CODEX_HOME: true,
	CLAUDE_CONFIG_DIR: true,
	XDG_DATA_HOME: true,
	HF_HOME: true,
	HF_ENDPOINT: true,
	MINSYNC_SERVER: true,
};
/** An embedder id or endpoint that would ship session text off the machine is refused before the CLI starts. */
const REMOTE_EMBEDDER_PREFIXES = ["openai:", "tei:"];

export interface WombatSearchClient {
	search(mode: WombatSearchMode, query: string, options?: WombatSearchOptions): Promise<WombatSearchResult>;
}

export interface WombatSyncClient extends WombatSearchClient {
	doctor(signal?: AbortSignal): Promise<WombatDoctorResult>;
	sync(signal?: AbortSignal): Promise<WombatSyncResult>;
}

export class WombatClient implements WombatSyncClient {
	private readonly options: WombatOptions;

	constructor(options: WombatOptions = {}) {
		this.options = options;
	}

	async doctor(signal?: AbortSignal): Promise<WombatDoctorResult> {
		const result = await this.run(["doctor", "--json"], signal, this.options.timeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		if (parsed === undefined || typeof parsed.ready !== "boolean") {
			return failure({ ...result, ok: false, reason: "invalid-output" });
		}
		const freshness = parsed.freshness !== null && typeof parsed.freshness === "object" ? parsed.freshness : {};
		const fresh = freshness as Record<string, unknown>;
		return ok(
			{
				ready: parsed.ready,
				version: string(parsed, "version"),
				minsyncVersion: string(parsed, "minsync_version"),
				indexedSessions: number(fresh, "indexed_sessions") ?? 0,
				indexedTurns: number(fresh, "indexed_turns") ?? 0,
			},
			result,
		);
	}

	/** Parse new and changed sessions into the mirror, then let MinSync index it incrementally. */
	async sync(signal?: AbortSignal): Promise<WombatSyncResult> {
		const args = ["sync", "--index", "--json"];
		for (const agent of this.options.agents ?? []) args.push("--agent", agent);
		const result = await this.run(
			args,
			signal,
			this.options.indexTimeoutMs ?? this.options.timeoutMs ?? DEFAULT_INDEX_TIMEOUT_MS,
		);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		if (parsed === undefined || number(parsed, "total_sessions") === undefined) {
			return failure({ ...result, ok: false, reason: "invalid-output" });
		}
		const indexed =
			parsed.indexed !== null && typeof parsed.indexed === "object"
				? (parsed.indexed as Record<string, unknown>)
				: undefined;
		return ok(
			{
				discovered: number(parsed, "discovered"),
				processed: number(parsed, "processed"),
				written: number(parsed, "written"),
				skipped: number(parsed, "skipped"),
				deleted: number(parsed, "deleted"),
				failed: number(parsed, "failed"),
				totalSessions: number(parsed, "total_sessions") ?? 0,
				totalMessages: number(parsed, "total_messages") ?? 0,
				...(indexed === undefined ? {} : { chunkCount: number(indexed, "chunkCount") }),
				indexOk: indexed === undefined ? true : indexed.ok !== false,
			},
			result,
		);
	}

	async search(mode: WombatSearchMode, query: string, options?: WombatSearchOptions): Promise<WombatSearchResult> {
		if (query.trim().length === 0) return { ok: true, hits: [], stdout: "", stderr: "", code: 0 };
		const args = [
			"search",
			"--mode",
			mode,
			"--limit",
			String(options?.topK ?? 20),
			...(this.options.agents?.length === 1 ? ["--agent", this.options.agents[0] as string] : []),
			"--json",
			query.trim(),
		];
		const result = await this.run(args, options?.signal, this.options.timeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS);
		if (!result.ok) return failure(result);
		const parsed = parseJson(result.stdout);
		if (!Array.isArray(parsed)) return failure({ ...result, ok: false, reason: "invalid-output" });
		const hits = parsed.map((row, index) => normalizeHit(row, mode, index));
		if (hits.some((hit) => hit === undefined)) return failure({ ...result, ok: false, reason: "invalid-output" });
		return {
			ok: true,
			hits: hits as WombatSearchHit[],
			stdout: result.stdout,
			stderr: result.stderr,
			code: result.code ?? 0,
		};
	}

	private async run(
		args: readonly string[],
		signal: AbortSignal | undefined,
		timeoutMs: number,
	): Promise<ProcessResult> {
		const env: NodeJS.ProcessEnv = {};
		for (const [key, value] of Object.entries(process.env)) {
			if (value !== undefined && (Object.hasOwn(SAFE_ENV_KEYS, key) || Object.hasOwn(SAFE_WOMBAT_ENV_KEYS, key)))
				env[key] = value;
		}
		for (const [key, value] of Object.entries(this.options.env ?? {})) {
			if (value === undefined) delete env[key];
			else if (Object.hasOwn(SAFE_ENV_KEYS, key) || Object.hasOwn(SAFE_WOMBAT_ENV_KEYS, key)) env[key] = value;
		}
		const embedder = env.WOMBAT_EMBEDDER;
		if (embedder !== undefined && REMOTE_EMBEDDER_PREFIXES.some((prefix) => embedder.startsWith(prefix))) {
			return { ok: false, reason: "remote-embedding-rejected", stdout: "", stderr: "", code: null };
		}
		// Global flags precede the subcommand: wombat declares them on the top-level command.
		const globals = [
			...(this.options.dataDir === undefined ? [] : ["--data-dir", this.options.dataDir]),
			...(this.options.configPath === undefined ? [] : ["--config", this.options.configPath]),
			...(this.options.includeSubagents === true ? ["--include-subagents"] : []),
			...(this.options.includeEphemeral === true ? ["--include-ephemeral"] : []),
		];
		return spawnWombat(
			this.options.binaryPath ?? DEFAULT_BINARY,
			[...globals, ...args],
			env,
			this.options,
			signal,
			timeoutMs,
		);
	}
}

function spawnWombat(
	binary: string,
	args: readonly string[],
	env: NodeJS.ProcessEnv,
	options: WombatOptions,
	signal: AbortSignal | undefined,
	timeoutMs: number,
): Promise<ProcessResult> {
	return new Promise((resolveResult) => {
		const portable = portableSpawnCommand(binary, args);
		const child = spawn(portable.command, portable.args, {
			env,
			detached: process.platform !== "win32",
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BoundedOutput = { text: "", bytes: 0, exceeded: false },
			stderr: BoundedOutput = { text: "", bytes: 0, exceeded: false },
			settled = false,
			reason: WombatFailureReason | undefined,
			pipeGuard: ReturnType<typeof setTimeout> | undefined;
		const max = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
		const finish = (code: number | null): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			clearTimeout(pipeGuard);
			signal?.removeEventListener("abort", abort);
			resolveResult({
				ok: code === 0 && reason === undefined,
				...(reason === undefined ? (code === 0 ? {} : { reason: "nonzero-exit" as const }) : { reason }),
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		};
		const timer = setTimeout(() => {
			reason = "timeout";
			terminateProcessTree(child, "SIGKILL");
		}, timeoutMs);
		const abort = () => {
			reason ??= "aborted";
			terminateProcessTree(child, "SIGKILL");
		};
		if (signal?.aborted) abort();
		signal?.addEventListener("abort", abort, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendOutput(stdout, chunk, max);
			if (stdout.exceeded && reason === undefined) {
				reason = "stdout-too-large";
				terminateProcessTree(child, "SIGKILL");
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendOutput(stderr, chunk, max);
			if (stderr.exceeded && reason === undefined) {
				reason = "stderr-too-large";
				terminateProcessTree(child, "SIGKILL");
			}
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			clearTimeout(pipeGuard);
			signal?.removeEventListener("abort", abort);
			resolveResult({
				ok: false,
				reason: error.code === "ENOENT" ? "binary-missing" : "spawn-error",
				stdout: stdout.text,
				stderr: "",
				code: null,
			});
		});
		// A descendant that inherited the stdio pipes (MinSync serve) keeps `close` from firing after the
		// direct child exits. Reap the group and bound the wait so the promise can never stay pending.
		child.on("exit", () => {
			if (settled) return;
			terminateProcessTree(child, "SIGKILL");
			pipeGuard = setTimeout(() => finish(child.exitCode), TREE_KILL_GRACE_MS);
		});
		child.on("close", (code) => finish(code));
	});
}

function parseJson(text: string): unknown {
	try {
		return JSON.parse(text.trim());
	} catch {
		return undefined;
	}
}

function parseObject(text: string): Record<string, unknown> | undefined {
	const value = parseJson(text);
	return value !== null && typeof value === "object" && !Array.isArray(value)
		? (value as Record<string, unknown>)
		: undefined;
}

function normalizeHit(value: unknown, mode: WombatSearchMode, index: number): WombatSearchHit | undefined {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
	const row = value as Record<string, unknown>;
	const text = (key: string): string => (typeof row[key] === "string" ? (row[key] as string) : "");
	const chunkId = text("chunk_id") || text("chunkId");
	const snippet = text("snippet") || text("content");
	const sessionId = text("session_id") || text("sessionId");
	const agent = text("agent");
	if (!isSafeOpaqueId(chunkId) || !isSafeOpaqueId(sessionId) || !isSafeOpaqueId(agent) || snippet.length === 0) {
		return undefined;
	}
	const optional = (key: string): string | undefined => {
		const found = text(key);
		return found.length === 0 ? undefined : found;
	};
	return {
		chunkId,
		snippet,
		score: typeof row.score === "number" && Number.isFinite(row.score) ? row.score : 1 / (index + 1),
		mode,
		agent,
		sessionId,
		catalogKey: text("catalog_key") || `${agent}:${sessionId}`,
		title: text("title"),
		cwd: optional("cwd"),
		sourcePath: optional("source_path"),
		startTurn: typeof row.start_turn === "number" ? row.start_turn : undefined,
		endTurn: typeof row.end_turn === "number" ? row.end_turn : undefined,
		startedAt: optional("started_at"),
		endedAt: optional("ended_at"),
	};
}

function number(value: Record<string, unknown>, key: string): number | undefined {
	return typeof value[key] === "number" && Number.isFinite(value[key]) ? (value[key] as number) : undefined;
}

function string(value: Record<string, unknown>, key: string): string | undefined {
	return typeof value[key] === "string" ? (value[key] as string) : undefined;
}

function ok<T>(data: T, result: ProcessResult): WombatOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}

function failure(result: ProcessResult): WombatFailure {
	return {
		ok: false,
		reason: result.reason ?? "nonzero-exit",
		stdout: result.stdout,
		stderr: boundDiagnosticText(result.stderr),
		code: result.code,
	};
}

function appendOutput(current: BoundedOutput, chunk: string, maxBytes: number): BoundedOutput {
	if (current.exceeded) return current;
	const bytes = Buffer.from(chunk, "utf8");
	const remaining = Math.max(0, maxBytes - current.bytes);
	const prefix = truncateUtf8(chunk, remaining);
	const total = current.bytes + bytes.byteLength;
	return {
		text: current.text + prefix,
		bytes: Math.min(total, maxBytes),
		exceeded: total > maxBytes,
	};
}

function isSafeOpaqueId(value: string): boolean {
	return value.length > 0 && value !== "." && value !== ".." && !/[\\/#\u0000-\u001f\u007f]/u.test(value);
}

function truncateUtf8(value: string, maxBytes: number): string {
	if (Buffer.byteLength(value) <= maxBytes) return value;
	const decoder = new TextDecoder("utf-8", { fatal: true });
	const bytes = Buffer.from(value, "utf8");
	for (let end = Math.min(maxBytes, bytes.length); end >= 0; end -= 1) {
		try {
			return decoder.decode(bytes.subarray(0, end));
		} catch {
			// Try the next shorter UTF-8 prefix.
		}
	}
	return "";
}
