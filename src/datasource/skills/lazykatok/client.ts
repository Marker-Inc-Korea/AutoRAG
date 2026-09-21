import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import type {
	LazykatokChunk,
	LazykatokChunkResult,
	LazykatokContext,
	LazykatokContextResult,
	LazykatokDoctorInfo,
	LazykatokDoctorResult,
	LazykatokFailure,
	LazykatokIndexInfo,
	LazykatokIndexResult,
	LazykatokOk,
	LazykatokOptions,
	LazykatokParentResult,
	LazykatokSearchHit,
	LazykatokSearchMode,
	LazykatokSearchOptions,
	LazykatokSearchResult,
	LazykatokSyncInfo,
	LazykatokSyncResult,
} from "./types.ts";
import { DEFAULT_LAZYKATOK_BINARY, DEFAULT_LAZYKATOK_MAX_BUFFER_BYTES, DEFAULT_LAZYKATOK_TIMEOUT_MS } from "./types.ts";

const SAFE_INHERITED_ENV_KEYS = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP"]);
/**
 * Environment namespaces the child may see. lazykatok still publishes its
 * configuration through the historical `KATOK_*` namespace (`KATOK_EMBEDDER`,
 * `KATOK_KAKAO_USER_ID`), so that prefix stays allowed alongside the
 * `LAZYKATOK_*` one.
 */
const SAFE_CLI_ENV_PREFIXES = ["KATOK_", "LAZYKATOK_"] as const;

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: LazykatokFailure["reason"];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

type SpawnRequest = {
	readonly options: LazykatokOptions;
	readonly args: readonly string[];
	readonly env: NodeJS.ProcessEnv;
	readonly signal?: AbortSignal;
	readonly cwd?: string;
};

/**
 * Thin external `lazykatok` CLI wrapper. Every method spawns the `lazykatok` binary as
 * a child process, parses JSON from stdout, and returns a discriminated ok/fail
 * union. No method throws for expected failures (missing binary, CLI nonzero
 * exit, timeout, oversized output, or invalid JSON). The client never opens the
 * KakaoTalk database directly.
 */
export class LazykatokClient {
	private readonly options: LazykatokOptions;

	constructor(options: LazykatokOptions = {}) {
		this.options = options;
	}

	async doctor(signal?: AbortSignal): Promise<LazykatokDoctorResult> {
		const result = await this.run(["doctor", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeDoctor(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async sync(signal?: AbortSignal): Promise<LazykatokSyncResult> {
		const result = await this.run(["sync", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeSync(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async index(signal?: AbortSignal): Promise<LazykatokIndexResult> {
		const result = await this.run(["index", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeIndex(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async search(
		mode: LazykatokSearchMode,
		query: string,
		options?: LazykatokSearchOptions,
	): Promise<LazykatokSearchResult> {
		const args = ["search", mode, query, "--json"];
		if (options?.topK !== undefined) args.push("--limit", String(options.topK));
		const result = await this.run(args, options?.signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonValue(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const hits = normalizeHits(parsed);
		return hits === undefined ? toFailure(result, "invalid-shape") : searchOk(hits, result);
	}

	async chunkGet(chunkId: string, signal?: AbortSignal): Promise<LazykatokChunkResult> {
		const result = await this.run(["chunk", "get", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeChunk(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async context(chunkId: string, signal?: AbortSignal): Promise<LazykatokContextResult> {
		const result = await this.run(["chunk", "context", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeContext(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async parent(chunkId: string, signal?: AbortSignal): Promise<LazykatokParentResult> {
		const result = await this.run(["chunk", "parent", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonValue(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeParentWindows(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	/** Single retrieval pipeline: build env, spawn, parse-free raw result. */
	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const env = controlledEnv(this.options.env);
		return spawnLazykatok({
			options: this.options,
			args: [...args, ...commonArgs(this.options)],
			env,
			signal,
		});
	}
}

function spawnLazykatok(request: SpawnRequest): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const { options, args, env, signal } = request;
		const portable = portableSpawnCommand(commandFor(options.binaryPath), args);
		const child = spawn(portable.command, [...portable.args], {
			env,
			...(request.cwd === undefined ? {} : { cwd: request.cwd }),
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: LazykatokFailure["reason"] | undefined;
		const maxBuffer = options.maxBufferBytes ?? DEFAULT_LAZYKATOK_MAX_BUFFER_BYTES;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_LAZYKATOK_TIMEOUT_MS);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminate(child);
		};
		if (signal?.aborted) abortHandler();
		signal?.addEventListener("abort", abortHandler, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendBounded(stdout, chunk, maxBuffer);
			if (stdout.capped) {
				finalReason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendBounded(stderr, chunk, maxBuffer);
			if (stderr.capped) {
				finalReason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abortHandler);
			const reason = error.code === "ENOENT" ? "binary-missing" : "spawn-error";
			resolve({ ok: false, reason, stdout: stdout.text, stderr: describeSpawnFailure(reason), code: null });
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abortHandler);
			if (finalReason !== undefined) {
				resolve({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			resolve({
				ok: code === 0,
				reason: code === 0 ? undefined : "nonzero-exit",
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		});
	});
}

function commandFor(binaryPath: string | undefined): string {
	return binaryPath === undefined ? DEFAULT_LAZYKATOK_BINARY : binaryPath;
}

/**
 * Flags common to every subcommand. lazykatok's own CLI contract only exposes
 * `--data-dir` and `--config` as global options; there is no `--workspace`
 * or global `--source` flag. AutoRAG never forces an AutoRAG-managed
 * workspace on lazykatok — without explicit options, lazykatok uses its own default
 * store (e.g. `~/Library/~/Library/Application Support/katok` on macOS).
 */
function commonArgs(options: LazykatokOptions): readonly string[] {
	const args: string[] = [];
	if (options.workspacePath !== undefined) args.push("--data-dir", options.workspacePath);
	if (options.configPath !== undefined) args.push("--config", options.configPath);
	return args;
}

/**
 * Builds the child environment by merging `process.env` with the caller's
 * overrides. `undefined` values remove a key.
 */
function controlledEnv(configuredEnv: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (value !== undefined && isAllowedLazykatokEnvKey(key)) env[key] = value;
	}
	for (const [key, value] of Object.entries(configuredEnv ?? {})) {
		if (value === undefined) {
			delete env[key];
		} else if (isAllowedLazykatokEnvKey(key)) {
			env[key] = value;
		}
	}
	return env;
}

function isAllowedLazykatokEnvKey(key: string): boolean {
	return SAFE_INHERITED_ENV_KEYS.has(key) || SAFE_CLI_ENV_PREFIXES.some((prefix) => key.startsWith(prefix));
}

function searchOk(hits: readonly LazykatokSearchHit[], result: ProcessResult): LazykatokSearchResult {
	return { ok: true, hits, data: { hits }, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
function terminate(child: ChildProcess): void {
	if (child.killed) return;
	if (process.platform !== "win32" && child.pid !== undefined) {
		try {
			process.kill(-child.pid, "SIGKILL");
			return;
		} catch {}
	}
	child.kill("SIGKILL");
}

/** Path-opaque stderr replacement for spawn failures (the raw Node error leaks the binary path). */
function describeSpawnFailure(reason: "binary-missing" | "spawn-error"): string {
	return reason === "binary-missing"
		? "the lazykatok binary could not be found"
		: "the lazykatok binary could not be started";
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk);
	const nextBytes = state.bytes + chunkBytes;
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remainingBytes = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remainingBytes), bytes: maxBytes, capped: true };
}

function parseJsonValue(stdout: string): unknown {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return undefined;
	try {
		return JSON.parse(trimmed);
	} catch {
		return undefined;
	}
}

function parseJsonObject(stdout: string): Record<string, unknown> | undefined {
	const parsed = parseJsonValue(stdout);
	if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) return undefined;
	return parsed as Record<string, unknown>;
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
	if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
	return value as Record<string, unknown>;
}

function asString(value: unknown): string | undefined {
	return typeof value === "string" ? value : undefined;
}

function asNumber(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function asBoolean(value: unknown): boolean | undefined {
	return typeof value === "boolean" ? value : undefined;
}

function normalizeDoctor(raw: Record<string, unknown>): LazykatokDoctorInfo | undefined {
	const version = asString(raw.version);
	// The real CLI carries no boolean `ready` field: it reports readiness through
	// the `freshness` block, the `archive` status, and the `source_adapter`
	// probes. A payload holding that block is a doctor payload.
	const ready = asBoolean(raw.ready) ?? (asRecord(raw.freshness) === undefined ? undefined : true);
	if (ready === undefined) return undefined;
	const metadata = stripKnown(raw, new Set(["version", "ready"]));
	return { ...(version !== undefined ? { version } : {}), ready, metadata };
}

function normalizeSync(raw: Record<string, unknown>): LazykatokSyncInfo | undefined {
	// The real `sync --json` report counts messages instead of asserting `synced`.
	const synced = asBoolean(raw.synced) ?? (asNumber(raw.total_messages) === undefined ? undefined : true);
	if (synced === undefined) return undefined;
	const messageCount = asNumber(raw.messageCount) ?? asNumber(raw.total_messages);
	const metadata = stripKnown(raw, new Set(["synced", "messageCount"]));
	return { synced, ...(messageCount !== undefined ? { messageCount } : {}), metadata };
}

function normalizeIndex(raw: Record<string, unknown>): LazykatokIndexInfo | undefined {
	// The real `index --json` report names the archive size `candidate_chunks`.
	const chunkCount = asNumber(raw.chunkCount) ?? asNumber(raw.candidate_chunks);
	if (chunkCount === undefined) return undefined;
	// `documents` is an inventory of the CLI's native semantic-store files; the
	// client's contract is path-opaque, so those paths are not surfaced.
	const metadata = stripKnown(raw, new Set(["chunkCount", "candidate_chunks", "documents"]));
	return { chunkCount, metadata };
}

function normalizeHits(raw: unknown): readonly LazykatokSearchHit[] | undefined {
	// Real lazykatok prints a bare JSON array; the legacy envelope wraps it in { hits }.
	const hits = Array.isArray(raw) ? raw : asRecord(raw)?.hits;
	if (!Array.isArray(hits)) return undefined;
	const normalized: LazykatokSearchHit[] = [];
	for (const entry of hits) {
		const record = asRecord(entry);
		if (record === undefined) return undefined;
		const hit = normalizeHit(record);
		if (hit === undefined) return undefined;
		normalized.push(hit);
	}
	return normalized;
}

/**
 * One lazykatok search hit. Accepts both the AutoRAG-legacy object envelope
 * (`{ chunkId, score, content }`) and the real lazykatok CLI fields
 * (`chunk_id`, `snippet`, `chat_name`, `sender_nickname`, `started_at`,
 * `ended_at`, `ranker`, `unit`, `rank`). Chat identity fields are surfaced
 * in metadata so callers can present a human-readable source.
 */
/**
 * Chat identity fields the CLI prints in snake_case. They are surfaced in
 * camelCase metadata so callers can present a human-readable source, while the
 * raw keys stay alongside them.
 */
function chatIdentityMetadata(record: Record<string, unknown>): Readonly<Record<string, unknown>> {
	const chatName = asString(record.chat_name);
	const senderNickname = asString(record.sender_nickname);
	const startedAt = asString(record.started_at);
	const endedAt = asString(record.ended_at);
	return {
		...(chatName !== undefined ? { chatName } : {}),
		...(senderNickname !== undefined ? { senderNickname } : {}),
		...(startedAt !== undefined ? { startedAt } : {}),
		...(endedAt !== undefined ? { endedAt } : {}),
	};
}

function normalizeHit(record: Record<string, unknown>): LazykatokSearchHit | undefined {
	const chunkId = asString(record.chunkId) ?? asString(record.chunk_id);
	const score = asNumber(record.score);
	const content = asString(record.content) ?? asString(record.snippet);
	if (chunkId === undefined || chunkId.length === 0 || score === undefined || content === undefined) return undefined;
	const metadata = stripKnown(record, new Set(["chunkId", "chunk_id", "score", "content", "snippet"]));
	return { chunkId, score, content, metadata: { ...metadata, ...chatIdentityMetadata(record) } };
}

function normalizeChunk(raw: Record<string, unknown>): LazykatokChunk | undefined {
	// Real payloads key the chunk as `chunk_id` (a parent window as `parent_id`)
	// and the body as `text`; the legacy AutoRAG envelope used `chunkId`/`content`.
	const chunkId = asString(raw.chunkId) ?? asString(raw.chunk_id) ?? asString(raw.parent_id);
	const content = asString(raw.content) ?? asString(raw.text);
	if (chunkId === undefined || chunkId.length === 0 || content === undefined) return undefined;
	const metadata = stripKnown(raw, new Set(["chunkId", "chunk_id", "parent_id", "content", "text"]));
	return { chunkId, content, metadata: { ...metadata, ...chatIdentityMetadata(raw) } };
}

function normalizeContext(raw: Record<string, unknown>): LazykatokContext | undefined {
	// Two accepted shapes: the legacy envelope (`{ chunks: [...] }`) and the real
	// `chunk context --json` payload (`{ chunk, previous, next, parent_windows }`).
	const legacyChunks = raw.chunks;
	if (Array.isArray(legacyChunks)) {
		const chunks = normalizeChunkList(legacyChunks);
		return chunks === undefined ? undefined : { chunks, metadata: stripKnown(raw, new Set(["chunks"])) };
	}
	const target = asRecord(raw.chunk);
	if (target === undefined) return undefined;
	const chunk = normalizeChunk(target);
	if (chunk === undefined) return undefined;
	// Neighbours carry chat identity but no `text`; text-less entries stay in
	// metadata rather than being surfaced as empty chunks.
	const before = normalizeChunkList(raw.previous) ?? [];
	const after = normalizeChunkList(raw.next) ?? [];
	return {
		chunks: [...before, chunk, ...after],
		metadata: stripKnown(raw, new Set(["chunk"])),
	};
}

/** Normalizes a list of chunk objects, dropping entries that carry no text of their own. */
function normalizeChunkList(value: unknown): LazykatokChunk[] | undefined {
	if (!Array.isArray(value)) return undefined;
	const chunks: LazykatokChunk[] = [];
	for (const entry of value) {
		const record = asRecord(entry);
		if (record === undefined) continue;
		const chunk = normalizeChunk(record);
		if (chunk !== undefined) chunks.push(chunk);
	}
	return chunks;
}

/** The real `chunk parent --json` payload is an array of parent windows. */
function normalizeParentWindows(raw: unknown): readonly LazykatokChunk[] | undefined {
	const windows = normalizeChunkList(raw);
	if (windows !== undefined) return windows;
	// Legacy envelope: `{ parent_windows: [...] }`.
	const nested = asRecord(raw)?.parent_windows;
	return normalizeChunkList(nested);
}

/** Returns a copy of `raw` minus the known top-level keys (preserved as typed fields). */
function stripKnown(raw: Record<string, unknown>, known: ReadonlySet<string>): Readonly<Record<string, unknown>> {
	const metadata: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(raw)) {
		if (!known.has(key)) metadata[key] = value;
	}
	return metadata;
}

function toFailure(result: ProcessResult, reason?: LazykatokFailure["reason"]): LazykatokFailure {
	return {
		ok: false,
		reason: reason ?? result.reason ?? "nonzero-exit",
		stdout: result.stdout,
		stderr: boundDiagnosticText(result.stderr),
		code: result.code,
	};
}

/**
 * lazykatok stderr reaches the operator as the CLI wrote it — paths included,
 * because that is what makes a failed search debuggable. Only the length is
 * bounded so one runaway process cannot flood a diagnostic.
 */
function boundDiagnosticText(value: string): string {
	if (value.length === 0) return "";
	return value.trim().slice(0, 4000);
}

function ok<T>(data: T, result: ProcessResult): LazykatokOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
