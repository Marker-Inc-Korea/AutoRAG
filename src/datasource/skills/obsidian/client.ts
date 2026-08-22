import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import { isAbsolute, resolve } from "node:path";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import { obsidianQmdCacheDir, obsidianQmdConfigDir, stripEdgeDashes, toQmdCollectionName } from "./paths.ts";
import type {
	QmdEmbedInfo,
	QmdEmbedResult,
	QmdEnsureInfo,
	QmdEnsureResult,
	QmdFailure,
	QmdFailureReason,
	QmdOk,
	QmdOptions,
	QmdSearchHit,
	QmdSearchMode,
	QmdSearchOptions,
	QmdSearchResult,
	QmdUpdateInfo,
	QmdUpdateResult,
} from "./types.ts";
import { DEFAULT_QMD_BINARY, DEFAULT_QMD_MAX_BUFFER_BYTES, DEFAULT_QMD_TIMEOUT_MS } from "./types.ts";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: QmdFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

const SAFE_INHERITED_ENV_KEYS = new Set([
	"HOME",
	"LANG",
	"LC_ALL",
	"PATH",
	"TMPDIR",
	"TMP",
	"TEMP",
	"USER",
	"LOGNAME",
	"SHELL",
	"XDG_RUNTIME_DIR",
	"HTTP_PROXY",
	"HTTPS_PROXY",
	"NO_PROXY",
	"http_proxy",
	"https_proxy",
	"no_proxy",
]);
const SAFE_QMD_ENV_PREFIX = "QMD_";

/**
 * Thin external `qmd` CLI wrapper for Obsidian vault indexing/search.
 * Every method spawns the `qmd` binary, never throws for expected failures,
 * and isolates config/cache under the AutoRAG workspace.
 */
export class QmdClient {
	private readonly options: QmdOptions;

	constructor(options: QmdOptions = {}) {
		this.options = options;
	}

	async ensureCollection(signal?: AbortSignal): Promise<QmdEnsureResult> {
		const vaultPath = resolveVaultPath(this.options.vaultPath);
		if (vaultPath === undefined) {
			return failure("not-configured", "vault path not configured");
		}
		const instanceId = this.options.instanceId ?? "default";
		const workspaceRoot = this.options.workspaceRoot ?? process.cwd();
		const collectionName = this.options.collectionName ?? toQmdCollectionName(instanceId);
		const configDir = obsidianQmdConfigDir(workspaceRoot, instanceId);
		const cacheDir = obsidianQmdCacheDir(workspaceRoot, instanceId);
		try {
			mkdirSync(configDir, { recursive: true });
			mkdirSync(cacheDir, { recursive: true });
			writeFileSync(resolve(configDir, "index.yml"), renderIndexYaml(collectionName, vaultPath), "utf8");
		} catch (error) {
			return failure("spawn-error", error instanceof Error ? error.message : "failed to write qmd config");
		}
		const probe = await this.run(["status"], signal);
		if (!probe.ok && probe.reason === "binary-missing") return toFailure(probe);
		const data: QmdEnsureInfo = { collectionName, vaultPath, configDir };
		return ok(data, probe.ok ? probe : { ok: true, stdout: "", stderr: "", code: 0 });
	}

	async update(signal?: AbortSignal): Promise<QmdUpdateResult> {
		const ensured = await this.ensureCollection(signal);
		if (!ensured.ok) return ensured;
		const result = await this.run(["update"], signal);
		if (!result.ok) return toFailure(result);
		return ok(parseUpdateInfo(result.stdout, result.stderr), result);
	}

	async embed(signal?: AbortSignal): Promise<QmdEmbedResult> {
		const ensured = await this.ensureCollection(signal);
		if (!ensured.ok) return ensured;
		const result = await this.run(["embed"], signal);
		if (!result.ok) return toFailure(result);
		const data: QmdEmbedInfo = { embedded: true };
		return ok(data, result);
	}

	async search(mode: QmdSearchMode, query: string, options?: QmdSearchOptions): Promise<QmdSearchResult> {
		const trimmed = query.trim();
		if (trimmed.length === 0) {
			return { ok: true, hits: [], data: { hits: [] }, stdout: "", stderr: "", code: 0 };
		}
		const collection = this.options.collectionName ?? toQmdCollectionName(this.options.instanceId ?? "default");
		const topK = options?.topK ?? 20;
		const args = [mode, trimmed, "--json", "-n", String(topK), "-c", collection];
		const result = await this.run(args, options?.signal);
		if (!result.ok) return toFailure(result);
		const hits = parseSearchHits(result.stdout);
		if (hits === undefined) return toFailure({ ...result, ok: false, reason: "invalid-json" });
		return { ok: true, hits, data: { hits }, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
	}

	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const env = controlledEnv(this.options);
		return spawnQmd({
			binaryPath: this.options.binaryPath,
			args,
			env,
			timeoutMs: this.options.timeoutMs ?? DEFAULT_QMD_TIMEOUT_MS,
			maxBufferBytes: this.options.maxBufferBytes ?? DEFAULT_QMD_MAX_BUFFER_BYTES,
			signal,
		});
	}
}

function renderIndexYaml(collectionName: string, vaultPath: string): string {
	const escapedPath = vaultPath.replace(/\\/g, "\\\\");
	return [
		"collections:",
		`  ${collectionName}:`,
		`    path: ${JSON.stringify(escapedPath)}`,
		'    pattern: "**/*.md"',
		"    ignore:",
		'      - ".obsidian/**"',
		'      - ".trash/**"',
		'      - ".git/**"',
		"",
	].join("\n");
}

function resolveVaultPath(vaultPath: string | undefined): string | undefined {
	if (vaultPath === undefined || vaultPath.trim().length === 0) return undefined;
	const trimmed = vaultPath.trim();
	return isAbsolute(trimmed) ? trimmed : resolve(trimmed);
}

function controlledEnv(options: QmdOptions): NodeJS.ProcessEnv {
	const instanceId = options.instanceId ?? "default";
	const workspaceRoot = options.workspaceRoot ?? process.cwd();
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (value !== undefined && isAllowedEnvKey(key)) env[key] = value;
	}
	for (const [key, value] of Object.entries(options.env ?? {})) {
		if (value === undefined) delete env[key];
		else if (isAllowedEnvKey(key)) env[key] = value;
	}
	env.QMD_CONFIG_DIR = obsidianQmdConfigDir(workspaceRoot, instanceId);
	env.XDG_CACHE_HOME = obsidianQmdCacheDir(workspaceRoot, instanceId);
	env.QMD_TRUST_LOCAL_CONFIG = env.QMD_TRUST_LOCAL_CONFIG ?? "1";
	return env;
}

function isAllowedEnvKey(key: string): boolean {
	return SAFE_INHERITED_ENV_KEYS.has(key) || key.startsWith(SAFE_QMD_ENV_PREFIX);
}

function spawnQmd(request: {
	readonly binaryPath: string | undefined;
	readonly args: readonly string[];
	readonly env: NodeJS.ProcessEnv;
	readonly timeoutMs: number;
	readonly maxBufferBytes: number;
	readonly signal?: AbortSignal;
}): Promise<ProcessResult> {
	return new Promise((resolveResult) => {
		const portable = portableSpawnCommand(request.binaryPath ?? DEFAULT_QMD_BINARY, request.args);
		const child = spawn(portable.command, [...portable.args], {
			env: request.env,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: QmdFailureReason | undefined;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, request.timeoutMs);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminate(child);
		};
		if (request.signal?.aborted) abortHandler();
		request.signal?.addEventListener("abort", abortHandler, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendBounded(stdout, chunk, request.maxBufferBytes);
			if (stdout.capped) {
				finalReason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendBounded(stderr, chunk, request.maxBufferBytes);
			if (stderr.capped) {
				finalReason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.signal?.removeEventListener("abort", abortHandler);
			const reason: QmdFailureReason = error.code === "ENOENT" ? "binary-missing" : "spawn-error";
			resolveResult({
				ok: false,
				reason,
				stdout: stdout.text,
				stderr: reason === "binary-missing" ? "qmd binary not found" : "qmd spawn failed",
				code: null,
			});
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.signal?.removeEventListener("abort", abortHandler);
			if (finalReason !== undefined) {
				resolveResult({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			resolveResult({
				ok: code === 0,
				reason: code === 0 ? undefined : "nonzero-exit",
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		});
	});
}

function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk, "utf8");
	if (state.bytes >= maxBytes) return { ...state, capped: true };
	const remaining = maxBytes - state.bytes;
	if (chunkBytes <= remaining) {
		return { text: state.text + chunk, bytes: state.bytes + chunkBytes, capped: false };
	}
	const partial = Buffer.from(chunk, "utf8").subarray(0, remaining).toString("utf8");
	return { text: state.text + partial, bytes: maxBytes, capped: true };
}

function ok<T>(data: T, result: ProcessResult): QmdOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}

function toFailure(result: ProcessResult): QmdFailure {
	return {
		ok: false,
		reason: result.reason ?? "nonzero-exit",
		stdout: result.stdout,
		stderr: result.stderr,
		code: result.code,
	};
}

function failure(reason: QmdFailureReason, message: string): QmdFailure {
	return { ok: false, reason, stdout: "", stderr: message, code: null };
}

function parseUpdateInfo(stdout: string, stderr: string): QmdUpdateInfo {
	const text = `${stdout}\n${stderr}`;
	const match =
		/Indexed:\s*(\d+)\s*new,\s*(\d+)\s*updated,\s*(\d+)\s*unchanged,\s*(\d+)\s*removed/i.exec(text) ??
		/(\d+)\s*new,\s*(\d+)\s*updated,\s*(\d+)\s*unchanged,\s*(\d+)\s*removed/i.exec(text);
	if (match !== null) {
		return {
			indexed: Number(match[1] ?? 0),
			updated: Number(match[2] ?? 0),
			unchanged: Number(match[3] ?? 0),
			removed: Number(match[4] ?? 0),
			needsEmbedding: /needsEmbedding|needs embedding|embed/i.test(text) || undefined,
		};
	}
	try {
		const parsed: unknown = JSON.parse(stdout);
		if (isRecord(parsed)) {
			return {
				indexed: numberField(parsed, "indexed") ?? numberField(parsed, "new") ?? 0,
				updated: numberField(parsed, "updated") ?? 0,
				unchanged: numberField(parsed, "unchanged") ?? 0,
				removed: numberField(parsed, "removed") ?? 0,
				needsEmbedding: typeof parsed.needsEmbedding === "boolean" ? parsed.needsEmbedding : undefined,
			};
		}
	} catch {}
	return { indexed: 0, updated: 0, unchanged: 0, removed: 0 };
}

function parseSearchHits(stdout: string): readonly QmdSearchHit[] | undefined {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return [];
	try {
		const parsed: unknown = JSON.parse(trimmed);
		const rows = extractHitRows(parsed);
		if (rows === undefined) return undefined;
		return rows.map(normalizeHit).filter((hit): hit is QmdSearchHit => hit !== undefined);
	} catch {
		return undefined;
	}
}

function extractHitRows(parsed: unknown): readonly unknown[] | undefined {
	if (Array.isArray(parsed)) return parsed;
	if (!isRecord(parsed)) return undefined;
	for (const key of ["results", "hits", "documents", "items"] as const) {
		const value = parsed[key];
		if (Array.isArray(value)) return value;
	}
	if ("docid" in parsed || "file" in parsed || "score" in parsed) return [parsed];
	return undefined;
}

function normalizeHit(raw: unknown): QmdSearchHit | undefined {
	if (!isRecord(raw)) return undefined;
	const score = numberField(raw, "score") ?? numberField(raw, "relevance") ?? 0;
	const docid = stringField(raw, "docid") ?? stringField(raw, "id");
	const file = stringField(raw, "file") ?? stringField(raw, "path") ?? stringField(raw, "displayPath");
	const title = stringField(raw, "title");
	const content =
		stringField(raw, "snippet") ??
		stringField(raw, "content") ??
		stringField(raw, "text") ??
		stringField(raw, "body") ??
		title ??
		file ??
		"";
	const chunkId = toChunkId(docid, file, title);
	if (chunkId.length === 0) return undefined;
	const metadata: Record<string, unknown> = {};
	if (file !== undefined) metadata.path = file;
	if (title !== undefined) metadata.title = title;
	if (docid !== undefined) metadata.docid = docid;
	return {
		chunkId,
		score,
		content,
		...(title !== undefined ? { title } : {}),
		...(file !== undefined ? { file } : {}),
		...(docid !== undefined ? { docid } : {}),
		...(Object.keys(metadata).length > 0 ? { metadata } : {}),
	};
}

function toChunkId(docid: string | undefined, file: string | undefined, title: string | undefined): string {
	if (docid !== undefined && docid.length > 0) return docid.replace(/^#/, "");
	if (file !== undefined && file.length > 0) {
		return stripEdgeDashes(
			file
				.replace(/^qmd:\/\//, "")
				.replace(/\\/g, "/")
				.replace(/[^A-Za-z0-9._/-]+/g, "-"),
		).slice(0, 120);
	}
	if (title !== undefined && title.length > 0) {
		return stripEdgeDashes(title.toLowerCase().replace(/[^a-z0-9._-]+/g, "-")).slice(0, 80);
	}
	return "";
}

function stringField(record: Record<string, unknown>, key: string): string | undefined {
	const value = record[key];
	return typeof value === "string" && value.length > 0 ? value : undefined;
}

function numberField(record: Record<string, unknown>, key: string): number | undefined {
	const value = record[key];
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
