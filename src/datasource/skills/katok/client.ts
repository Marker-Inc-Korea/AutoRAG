import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { katokDatasourceRoot } from "./paths.ts";
import type {
	KatokChunk,
	KatokChunkResult,
	KatokContext,
	KatokContextResult,
	KatokDoctorInfo,
	KatokDoctorResult,
	KatokFailure,
	KatokIndexInfo,
	KatokIndexResult,
	KatokOk,
	KatokOptions,
	KatokParentResult,
	KatokSearchHit,
	KatokSearchMode,
	KatokSearchOptions,
	KatokSearchResult,
	KatokSyncInfo,
	KatokSyncResult,
} from "./types.ts";
import {
	DEFAULT_KATOK_BINARY,
	DEFAULT_KATOK_MAX_BUFFER_BYTES,
	DEFAULT_KATOK_SOURCE,
	DEFAULT_KATOK_TIMEOUT_MS,
} from "./types.ts";

/**
 * Environment keys whose mere presence (any value) enables remote embeddings
 * for the `katok` CLI. Compared case-insensitively so casing tricks cannot
 * smuggle remote embedder configuration into the spawned process.
 */
const REMOTE_EMBEDDING_KEYS: ReadonlySet<string> = new Set(["embedder_base_url", "allow_remote_embeddings"]);

/** The `KATOK_EMBEDDER` key (case-insensitive) is only rejected when URL-valued. */
const KATOK_EMBEDDER_KEY = "katok_embedder";
const SAFE_INHERITED_ENV_KEYS = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP"]);
const SAFE_KATOK_ENV_PREFIX = "KATOK_";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: KatokFailure["reason"];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly violatingKey?: string;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

type SpawnRequest = {
	readonly options: KatokOptions;
	readonly args: readonly string[];
	readonly env: NodeJS.ProcessEnv;
	readonly signal?: AbortSignal;
};

interface RemoteEmbeddingViolation {
	readonly key: string;
}

/**
 * Thin external `katok` CLI wrapper. Every method spawns the `katok` binary as
 * a child process, parses JSON from stdout, and returns a discriminated ok/fail
 * union. No method throws for expected failures (missing binary, rejected
 * remote-embedding config, CLI nonzero exit, timeout, oversized output, or
 * invalid JSON). The client never opens the KakaoTalk database directly.
 *
 * Remote-embedding configuration is rejected before spawn, case-insensitively,
 * so the spawned process can never egress embeddings to a remote endpoint.
 */
export class KatokClient {
	private readonly options: KatokOptions;

	constructor(options: KatokOptions = {}) {
		this.options = options;
	}

	async doctor(signal?: AbortSignal): Promise<KatokDoctorResult> {
		const result = await this.run(["doctor", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeDoctor(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async sync(signal?: AbortSignal): Promise<KatokSyncResult> {
		const result = await this.run(["sync", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeSync(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async index(signal?: AbortSignal): Promise<KatokIndexResult> {
		const result = await this.run(["index", "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeIndex(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async search(mode: KatokSearchMode, query: string, options?: KatokSearchOptions): Promise<KatokSearchResult> {
		const args = ["search", mode, query, "--json"];
		if (options?.topK !== undefined) args.push("--top-k", String(options.topK));
		if (options?.scope !== undefined) args.push("--scope", options.scope);
		const result = await this.run(args, options?.signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const hits = normalizeHits(parsed);
		return hits === undefined ? toFailure(result, "invalid-shape") : searchOk(hits, result);
	}

	async chunkGet(chunkId: string, signal?: AbortSignal): Promise<KatokChunkResult> {
		const result = await this.run(["chunk", "get", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeChunk(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async context(chunkId: string, signal?: AbortSignal): Promise<KatokContextResult> {
		const result = await this.run(["context", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeContext(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async parent(chunkId: string, signal?: AbortSignal): Promise<KatokParentResult> {
		const result = await this.run(["parent", chunkId, "--json"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeChunk(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	/** Single retrieval pipeline: build env, gate remote embeddings, spawn, parse-free raw result. */
	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const violation = findRemoteEmbeddingViolation({ ...process.env, ...(this.options.env ?? {}) });
		if (violation !== undefined) {
			return {
				ok: false,
				reason: "remote-embedding-rejected",
				stdout: "",
				stderr: "",
				code: null,
				violatingKey: violation.key,
			};
		}
		const env = controlledEnv(this.options.env);
		return spawnKatok({ options: this.options, args: [...args, ...commonArgs(this.options)], env, signal });
	}
}

function spawnKatok(request: SpawnRequest): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const { options, args, env, signal } = request;
		const child = spawn(commandFor(options.binaryPath), args, {
			env,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: KatokFailure["reason"] | undefined;
		const maxBuffer = options.maxBufferBytes ?? DEFAULT_KATOK_MAX_BUFFER_BYTES;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_KATOK_TIMEOUT_MS);
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
	return binaryPath === undefined ? DEFAULT_KATOK_BINARY : binaryPath;
}

/** Flags common to every subcommand: data source, fixture path, workspace. */
function commonArgs(options: KatokOptions): readonly string[] {
	const args: string[] = [];
	const source = options.source ?? DEFAULT_KATOK_SOURCE;
	args.push("--source", source);
	if (source === "fixture" && options.fixturePath !== undefined) {
		args.push("--fixture-path", options.fixturePath);
	}
	const workspace = resolveWorkspace(options);
	if (workspace !== undefined) args.push("--workspace", workspace);
	return args;
}

function resolveWorkspace(options: KatokOptions): string | undefined {
	if (options.workspacePath !== undefined) return options.workspacePath;
	const root = options.root ?? process.cwd();
	return katokDatasourceRoot(root);
}

/**
 * Builds the child environment by merging `process.env` with the caller's
 * overrides. `undefined` values remove a key. The remote-embedding gate runs
 * on this merged env so the spawned process can never receive a remote
 * embedder configuration.
 */
function controlledEnv(configuredEnv: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (value !== undefined && isAllowedKatokEnvKey(key)) env[key] = value;
	}
	for (const [key, value] of Object.entries(configuredEnv ?? {})) {
		if (value === undefined) {
			delete env[key];
		} else if (isAllowedKatokEnvKey(key)) {
			env[key] = value;
		}
	}
	return env;
}

function isAllowedKatokEnvKey(key: string): boolean {
	return SAFE_INHERITED_ENV_KEYS.has(key) || key.startsWith(SAFE_KATOK_ENV_PREFIX);
}

function findRemoteEmbeddingViolation(env: NodeJS.ProcessEnv): RemoteEmbeddingViolation | undefined {
	for (const [key, value] of Object.entries(env)) {
		if (value === undefined) continue;
		const lower = key.toLowerCase();
		if (REMOTE_EMBEDDING_KEYS.has(lower)) return { key };
		if (lower === KATOK_EMBEDDER_KEY && isUrlValue(value)) return { key };
	}
	return undefined;
}

function isUrlValue(value: string): boolean {
	return /^https?:\/\//i.test(value.trim());
}

function searchOk(hits: readonly KatokSearchHit[], result: ProcessResult): KatokSearchResult {
	return { ok: true, hits, data: { hits }, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}

/** Path-opaque stderr replacement for spawn failures (the raw Node error leaks the binary path). */
function describeSpawnFailure(reason: "binary-missing" | "spawn-error"): string {
	return reason === "binary-missing" ? "the katok binary could not be found" : "the katok binary could not be started";
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk);
	const nextBytes = state.bytes + chunkBytes;
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remainingBytes = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remainingBytes), bytes: maxBytes, capped: true };
}

function parseJsonObject(stdout: string): Record<string, unknown> | undefined {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return undefined;
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
	} catch {
		return undefined;
	}
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

function normalizeDoctor(raw: Record<string, unknown>): KatokDoctorInfo | undefined {
	const version = asString(raw.version);
	const ready = asBoolean(raw.ready);
	if (ready === undefined) return undefined;
	const metadata = stripKnown(raw, new Set(["version", "ready"]));
	return { ...(version !== undefined ? { version } : {}), ready, metadata };
}

function normalizeSync(raw: Record<string, unknown>): KatokSyncInfo | undefined {
	const synced = asBoolean(raw.synced);
	if (synced === undefined) return undefined;
	const messageCount = asNumber(raw.messageCount);
	const metadata = stripKnown(raw, new Set(["synced", "messageCount"]));
	return { synced, ...(messageCount !== undefined ? { messageCount } : {}), metadata };
}

function normalizeIndex(raw: Record<string, unknown>): KatokIndexInfo | undefined {
	const chunkCount = asNumber(raw.chunkCount);
	if (chunkCount === undefined) return undefined;
	const metadata = stripKnown(raw, new Set(["chunkCount"]));
	return { chunkCount, metadata };
}

function normalizeHits(raw: Record<string, unknown>): readonly KatokSearchHit[] | undefined {
	const hits = raw.hits;
	if (!Array.isArray(hits)) return undefined;
	const normalized: KatokSearchHit[] = [];
	for (const entry of hits) {
		const record = asRecord(entry);
		if (record === undefined) return undefined;
		const chunkId = asString(record.chunkId);
		const score = asNumber(record.score);
		const content = asString(record.content);
		if (chunkId === undefined || chunkId.length === 0 || score === undefined || content === undefined)
			return undefined;
		const metadata = stripKnown(record, new Set(["chunkId", "score", "content"]));
		normalized.push({ chunkId, score, content, metadata });
	}
	return normalized;
}

function normalizeChunk(raw: Record<string, unknown>): KatokChunk | undefined {
	const chunkId = asString(raw.chunkId);
	const content = asString(raw.content);
	if (chunkId === undefined || chunkId.length === 0 || content === undefined) return undefined;
	const metadata = stripKnown(raw, new Set(["chunkId", "content"]));
	return { chunkId, content, metadata };
}

function normalizeContext(raw: Record<string, unknown>): KatokContext | undefined {
	const chunksValue = raw.chunks;
	if (!Array.isArray(chunksValue)) return undefined;
	const chunks: KatokChunk[] = [];
	for (const entry of chunksValue) {
		const record = asRecord(entry);
		if (record === undefined) return undefined;
		const chunk = normalizeChunk(record);
		if (chunk === undefined) return undefined;
		chunks.push(chunk);
	}
	const metadata = stripKnown(raw, new Set(["chunks"]));
	return { chunks, metadata };
}

/** Returns a copy of `raw` minus the known top-level keys (preserved as typed fields). */
function stripKnown(raw: Record<string, unknown>, known: ReadonlySet<string>): Readonly<Record<string, unknown>> {
	const metadata: Record<string, unknown> = {};
	for (const [key, value] of Object.entries(raw)) {
		if (!known.has(key)) metadata[key] = value;
	}
	return metadata;
}

function toFailure(result: ProcessResult, reason?: KatokFailure["reason"]): KatokFailure {
	return {
		ok: false,
		reason: reason ?? result.reason ?? "nonzero-exit",
		stdout: sanitizeDiagnosticText(result.stdout),
		stderr: sanitizeDiagnosticText(result.stderr),
		code: result.code,
		...(result.violatingKey !== undefined ? { violatingKey: result.violatingKey } : {}),
	};
}

function sanitizeDiagnosticText(value: string): string {
	if (value.length === 0) return "";
	return "katok command failed; details suppressed for datasource privacy";
}

function ok<T>(data: T, result: ProcessResult): KatokOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
