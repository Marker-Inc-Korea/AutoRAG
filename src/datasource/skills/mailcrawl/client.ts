import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { ManagedCliConfigManager, ManagedCliRegistry } from "../../../cli/managed-cli-config.ts";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import {
	createMailcrawlManagedCliProvider,
	ensurePrivateMailcrawlDataDir,
	validateMailcrawlInstanceId,
} from "./config.ts";
import type {
	MailcrawlFailure,
	MailcrawlFailureReason,
	MailcrawlIndexInfo,
	MailcrawlOk,
	MailcrawlOptions,
	MailcrawlSearchHit,
	MailcrawlSearchMode,
	MailcrawlSearchOptions,
	MailcrawlSearchResult,
	MailcrawlSyncResult,
} from "./types.ts";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: MailcrawlFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};
type BoundedOutput = {
	readonly text: string;
	readonly bytes: number;
	readonly exceeded: boolean;
};
const DEFAULT_BINARY = "mailcrawl";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_BUFFER_BYTES = 1_048_576;
const SAFE_ENV_KEYS = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP", "NO_COLOR"]);
const SAFE_MAILCRAWL_ENV_KEYS = new Set([
	"MAILCRAWL_DATA_DIR",
	"MAILCRAWL_EMBEDDER",
	"MAILCRAWL_KIWI_MODEL",
	"MAILCRAWL_JA_HELPER",
	"MAILCRAWL_ZH_HELPER",
]);
const REMOTE_EMBEDDING_ENV_KEYS = new Set([
	"mailcrawl_embedder_base_url",
	"mailcrawl_embedding_base_url",
	"mailcrawl_allow_remote_embeddings",
]);

export interface MailcrawlSearchClient {
	search(mode: MailcrawlSearchMode, query: string, options?: MailcrawlSearchOptions): Promise<MailcrawlSearchResult>;
}

export interface MailcrawlIndexClient extends MailcrawlSearchClient {
	index(signal?: AbortSignal): Promise<MailcrawlOk<MailcrawlIndexInfo> | MailcrawlFailure>;
}

export class MailcrawlClient implements MailcrawlSearchClient {
	private readonly options: MailcrawlOptions;
	private readonly manager: ManagedCliConfigManager | undefined;

	constructor(options: MailcrawlOptions = {}) {
		this.options = options;
		if (options.instanceId !== undefined) validateMailcrawlInstanceId(options.instanceId);
		if (options.managedCliConfigManager) this.manager = options.managedCliConfigManager;
		else if (options.workspacePath !== undefined) {
			const registry = new ManagedCliRegistry();
			registry.register(createMailcrawlManagedCliProvider(options.binaryPath));
			this.manager = new ManagedCliConfigManager({ workspace: options.workspacePath, registry });
		}
	}

	async sync(signal?: AbortSignal): Promise<MailcrawlSyncResult> {
		const args = [
			"sync",
			"--json",
			...(this.options.account === undefined ? [] : ["--account", this.options.account]),
			...(this.options.mailbox === undefined ? [] : ["--mailbox", this.options.mailbox]),
			...(this.options.backend === undefined ? [] : ["--backend", this.options.backend]),
			...(this.options.source === undefined ? [] : ["--source", this.options.source]),
			...(this.options.fixture === undefined ? [] : ["--fixture", this.options.fixture]),
			...(this.options.himalayaConfig === undefined ? [] : ["--himalaya-config", this.options.himalayaConfig]),
		];
		const result = await this.run(args, signal);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		if (parsed === undefined || !hasAnyNumber(parsed, ["added", "updated", "deleted", "unchanged", "chunksAdded"])) {
			return failure({ ...result, ok: false, reason: "invalid-output" });
		}
		const messages = number(parsed, "added") ?? number(parsed, "updated") ?? number(parsed, "unchanged") ?? 0;
		return ok(
			{
				messages,
				added: number(parsed, "added"),
				updated: number(parsed, "updated"),
				deleted: number(parsed, "deleted"),
				unchanged: number(parsed, "unchanged"),
				chunksAdded: number(parsed, "chunksAdded"),
				archiveRevision: string(parsed, "archiveRevision"),
			},
			result,
		);
	}

	async search(
		mode: MailcrawlSearchMode,
		query: string,
		options?: MailcrawlSearchOptions,
	): Promise<MailcrawlSearchResult> {
		if (query.trim().length === 0) return { ok: true, hits: [], stdout: "", stderr: "", code: 0 };
		const args = [
			"search",
			"--mode",
			mode === "keyword" ? "bm25" : mode,
			...(this.options.account === undefined ? [] : ["--account", this.options.account]),
			...(this.options.mailbox === undefined ? [] : ["--mailbox", this.options.mailbox]),
			"--limit",
			String(options?.topK ?? 20),
			"--json",
			query.trim(),
		];
		const result = await this.run(args, options?.signal);
		if (!result.ok) return failure(result);
		const parsed = parseJson(result.stdout);
		if (parsed === undefined) return failure({ ...result, ok: false, reason: "invalid-output" });
		const rows = Array.isArray(parsed) ? parsed : objectArray(parsed, ["hits", "results", "items"]);
		if (rows === undefined) return failure({ ...result, ok: false, reason: "invalid-output" });
		const hits = rows.map((row, index) => normalizeHit(row, mode, index));
		if (hits.some((hit) => hit === undefined)) {
			return failure({ ...result, ok: false, reason: "invalid-output" });
		}
		return {
			ok: true,
			hits: hits as MailcrawlSearchHit[],
			stdout: result.stdout,
			stderr: result.stderr,
			code: result.code ?? 0,
		};
	}

	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const violation = findRemoteEmbeddingViolation({ ...process.env, ...(this.options.env ?? {}) });
		if (violation !== undefined) {
			return { ok: false, reason: "remote-embedding-rejected", stdout: "", stderr: "", code: null };
		}
		let launch: { readonly env: Readonly<Record<string, string>>; readonly cwd?: string } | undefined;
		try {
			if (this.manager)
				launch = await this.manager.materialize("mailcrawl", {
					instance: this.options.instanceId,
					...(this.options.dataDir === undefined ? {} : { config: { dataDir: this.options.dataDir } }),
				});
			else if (this.options.dataDir !== undefined) ensurePrivateMailcrawlDataDir(this.options.dataDir);
		} catch {
			return { ok: false, reason: "spawn-error", stdout: "", stderr: "", code: null };
		}
		const env: NodeJS.ProcessEnv = {};
		for (const [key, value] of Object.entries(process.env)) {
			if (value !== undefined && (SAFE_ENV_KEYS.has(key) || SAFE_MAILCRAWL_ENV_KEYS.has(key))) env[key] = value;
		}
		for (const [key, value] of Object.entries(this.options.env ?? {})) {
			if (value === undefined) delete env[key];
			else if (SAFE_ENV_KEYS.has(key) || SAFE_MAILCRAWL_ENV_KEYS.has(key)) env[key] = value;
		}
		if (this.manager === undefined && this.options.dataDir !== undefined)
			env.MAILCRAWL_DATA_DIR = this.options.dataDir;
		return spawnMailcrawl(this.options.binaryPath ?? DEFAULT_BINARY, args, env, this.options, signal, launch);
	}

	async index(signal?: AbortSignal): Promise<MailcrawlOk<MailcrawlIndexInfo> | MailcrawlFailure> {
		const result = await this.run(["index", "--json"], signal);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		if (
			parsed === undefined ||
			(number(parsed, "embedded") === undefined &&
				number(parsed, "reused") === undefined &&
				string(parsed, "generation") === undefined)
		) {
			return failure({ ...result, ok: false, reason: "invalid-output" });
		}
		return ok(
			{
				embedded: number(parsed, "embedded"),
				reused: number(parsed, "reused"),
				generation: string(parsed, "generation"),
			},
			result,
		);
	}
}

function spawnMailcrawl(
	binary: string,
	args: readonly string[],
	env: NodeJS.ProcessEnv,
	options: MailcrawlOptions,
	signal: AbortSignal | undefined,
	launch: { readonly env: Readonly<Record<string, string>>; readonly cwd?: string } | undefined,
): Promise<ProcessResult> {
	return new Promise((resolveResult) => {
		const portable = portableSpawnCommand(binary, args);
		const child = spawn(portable.command, portable.args, {
			env: { ...env, ...(launch?.env ?? {}) },
			...(launch?.cwd === undefined ? {} : { cwd: launch.cwd }),
			detached: process.platform !== "win32",
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BoundedOutput = { text: "", bytes: 0, exceeded: false },
			stderr: BoundedOutput = { text: "", bytes: 0, exceeded: false },
			settled = false,
			reason: MailcrawlFailureReason | undefined;
		const max = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
		const timer = setTimeout(() => {
			reason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		const abort = () => {
			reason = "aborted";
			terminate(child);
		};
		if (signal?.aborted) abort();
		signal?.addEventListener("abort", abort, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendOutput(stdout, chunk, max);
			if (stdout.exceeded && reason === undefined) {
				reason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendOutput(stderr, chunk, max);
			if (stderr.exceeded && reason === undefined) {
				reason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			signal?.removeEventListener("abort", abort);
			resolveResult({
				ok: false,
				reason: error.code === "ENOENT" ? "binary-missing" : "spawn-error",
				stdout: stdout.text,
				stderr: "",
				code: null,
			});
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			signal?.removeEventListener("abort", abort);
			resolveResult({
				ok: code === 0 && reason === undefined,
				...(reason === undefined ? (code === 0 ? {} : { reason: "nonzero-exit" as const }) : { reason }),
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		});
	});
}

function terminate(child: ChildProcess): void {
	if (child.killed) return;
	if (process.platform !== "win32" && child.pid !== undefined) {
		try {
			process.kill(-child.pid, "SIGTERM");
			return;
		} catch {
			/* swallow process-group fallback */
		}
	}
	child.kill("SIGTERM");
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
function objectArray(value: unknown, keys: readonly string[]): readonly unknown[] | undefined {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
	for (const key of keys)
		if (Array.isArray((value as Record<string, unknown>)[key]))
			return (value as Record<string, unknown>)[key] as readonly unknown[];
	return undefined;
}
function hasAnyNumber(value: Record<string, unknown>, keys: readonly string[]): boolean {
	return keys.some((key) => number(value, key) !== undefined);
}
function normalizeHit(value: unknown, mode: MailcrawlSearchMode, index: number): MailcrawlSearchHit | undefined {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
	const row = value as Record<string, unknown>;
	const text = (key: string): string => (typeof row[key] === "string" ? (row[key] as string) : "");
	const chunkId = text("chunkId") || text("chunk_id");
	const messageId = text("messageId") || text("message_id");
	const accountId = text("accountId") || text("account_id");
	const mailbox = text("mailbox");
	const snippet = text("snippet") || text("content") || text("text");
	if (!isSafeOpaqueId(chunkId) || !isSafeOpaqueId(messageId) || !snippet) return undefined;
	const to = Array.isArray(row.to) ? row.to.filter((item): item is string => typeof item === "string") : [];
	return {
		chunkId,
		messageId,
		threadId: text("threadId") || text("thread_id"),
		accountId,
		mailbox,
		subject: text("subject"),
		from: text("from"),
		to,
		date: text("date"),
		snippet,
		score: typeof row.score === "number" ? row.score : 1 / (index + 1),
		mode,
	};
}
function number(value: Record<string, unknown>, key: string): number | undefined {
	return typeof value[key] === "number" && Number.isFinite(value[key]) ? (value[key] as number) : undefined;
}
function string(value: Record<string, unknown>, key: string): string | undefined {
	return typeof value[key] === "string" ? (value[key] as string) : undefined;
}
function ok<T>(data: T, result: ProcessResult): MailcrawlOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
function failure(result: ProcessResult): MailcrawlFailure {
	return {
		ok: false,
		reason: result.reason ?? "nonzero-exit",
		stdout: "",
		stderr: result.stderr.length > 0 ? "mailcrawl command failed; details suppressed for datasource privacy" : "",
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

function findRemoteEmbeddingViolation(env: NodeJS.ProcessEnv): string | undefined {
	for (const [key, value] of Object.entries(env)) {
		if (value === undefined) continue;
		const lower = key.toLowerCase();
		if (REMOTE_EMBEDDING_ENV_KEYS.has(lower)) return key;
		if (lower === "mailcrawl_embedder" && /^https?:\/\//iu.test(value.trim())) return key;
	}
	return undefined;
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
