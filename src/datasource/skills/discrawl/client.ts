import { spawn } from "node:child_process";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import { TREE_KILL_GRACE_MS, terminateProcessTree } from "../../../process/terminate-tree.ts";
import type {
	DiscrawlDoctorInfo,
	DiscrawlDoctorResult,
	DiscrawlEmbeddingConfigResult,
	DiscrawlEmbeddingRuntime,
	DiscrawlEmbedInfo,
	DiscrawlEmbedResult,
	DiscrawlFailure,
	DiscrawlMetadataInfo,
	DiscrawlMetadataResult,
	DiscrawlOk,
	DiscrawlOptions,
	DiscrawlSearchHit,
	DiscrawlSearchMode,
	DiscrawlSearchOptions,
	DiscrawlSearchResult,
	DiscrawlStatusInfo,
	DiscrawlStatusResult,
	DiscrawlSyncInfo,
	DiscrawlSyncResult,
} from "./types.ts";
import { DEFAULT_DISCRAWL_BINARY, DEFAULT_DISCRAWL_MAX_BUFFER_BYTES, DEFAULT_DISCRAWL_TIMEOUT_MS } from "./types.ts";

/**
 * Environment keys that would hand the CLI a Discord *user* token. Automating
 * a user account violates Discord's Community Guidelines (rule 14) and can
 * result in account termination, so these are rejected before spawn rather
 * than forwarded.
 *
 * No Discord credential is forwarded at all: AutoRAG reads discrawl's native
 * archive, and the CLI owns its own authentication.
 */
const USER_TOKEN_KEYS: ReadonlySet<string> = new Set([
	"discord_user_token",
	"discord_token",
	"discord_self_token",
	"discord_account_token",
]);

const SAFE_INHERITED_ENV_KEYS = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP"]);
const SAFE_DISCRAWL_ENV_PREFIX = "DISCRAWL_";

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: DiscrawlFailure["reason"];
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
	readonly options: DiscrawlOptions;
	readonly args: readonly string[];
	readonly env: NodeJS.ProcessEnv;
	readonly signal?: AbortSignal;
	readonly cwd?: string;
};

/**
 * Thin external `discrawl` CLI wrapper. Every method spawns the `discrawl`
 * binary as a child process, parses JSON from stdout, and returns a
 * discriminated ok/fail union. No method throws for expected failures (missing
 * binary, rejected user token, CLI nonzero exit, timeout, oversized output, or
 * invalid JSON). The client never opens the Discord archive database directly.
 *
 * `--json` is a *global* discrawl flag and must precede the subcommand;
 * `discrawl search --json` is rejected by the CLI as an unknown flag.
 */
export class DiscrawlClient {
	private readonly options: DiscrawlOptions;

	constructor(options: DiscrawlOptions = {}) {
		this.options = options;
	}

	async configureEmbeddings(runtime: DiscrawlEmbeddingRuntime): Promise<DiscrawlEmbeddingConfigResult> {
		if (this.options.configPath !== undefined) return { configured: false, rebuildRequired: false };
		const workspace = this.options.workspacePath ?? this.options.root;
		if (workspace === undefined) return { configured: false, rebuildRequired: false };
		const path = `${workspace}/.autorag/datasources/discrawl/config.toml`;
		let existing = "";
		try {
			existing = readFileSync(path, "utf8");
		} catch {}
		const managed = existing.startsWith("# AutoRAG managed discrawl embeddings v1\n");
		if (existing.length > 0 && !managed) return { configured: false, rebuildRequired: false };
		const previous = managed ? parseManagedEmbedding(existing) : undefined;
		const config = managedDiscrawlConfig({ ...runtime, baseUrl: nativeGatewayBaseUrl(runtime.baseUrl) });
		mkdirSync(dirname(path), { recursive: true });
		if (existing !== config) writeFileSync(path, config, "utf8");
		return {
			configured: true,
			rebuildRequired:
				previous !== undefined &&
				(previous.provider !== runtime.provider ||
					previous.model !== runtime.model ||
					previous.dimensions !== runtime.dimensions),
		};
	}

	async prepareEmbeddings(): Promise<boolean> {
		const runtime = this.options.embeddingRuntime;
		if (runtime === undefined) return false;
		const configured = await this.configureEmbeddings(runtime);
		if (!configured.configured) return false;
		const metadata = await this.metadata();
		if (!metadata.ok) return configured.rebuildRequired;
		return (
			configured.rebuildRequired ||
			(metadata.data.embeddingProvider !== undefined && metadata.data.embeddingProvider !== runtime.provider) ||
			(metadata.data.embeddingModel !== undefined && metadata.data.embeddingModel !== runtime.model) ||
			(metadata.data.embeddingDimensions !== undefined && metadata.data.embeddingDimensions !== runtime.dimensions)
		);
	}

	async metadata(signal?: AbortSignal): Promise<DiscrawlMetadataResult> {
		const result = await this.runJson(["metadata"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeMetadata(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async doctor(signal?: AbortSignal): Promise<DiscrawlDoctorResult> {
		const result = await this.run(["doctor"], signal);
		if (!result.ok) return toFailure(result);
		const data = normalizeDoctor(parseKeyValueLines(result.stdout));
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async status(signal?: AbortSignal): Promise<DiscrawlStatusResult> {
		const result = await this.runJson(["status"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeStatus(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	/**
	 * Imports new messages from the local Discord Desktop cache through
	 * `discrawl wiretap`. AutoRAG never drives discrawl's bot sync — that path
	 * belongs to the CLI, which owns its own credential — so no Discord token is
	 * involved in an AutoRAG refresh.
	 */
	async sync(signal?: AbortSignal): Promise<DiscrawlSyncResult> {
		const result = await this.runJson(["wiretap"], signal);
		if (!result.ok) return toFailure(result);
		const parsed = parseJsonObject(result.stdout);
		if (parsed === undefined) return toFailure(result, "invalid-json");
		const data = normalizeSync(parsed);
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async embed(limit?: number, rebuild = false, signal?: AbortSignal): Promise<DiscrawlEmbedResult> {
		const args = ["embed"];
		if (rebuild) args.push("--rebuild");
		if (limit !== undefined) args.push("--limit", String(limit));
		const result = await this.run(args, signal);
		if (!result.ok) return toFailure(result);
		const data = normalizeEmbed(parseKeyValueLines(result.stdout));
		return data === undefined ? toFailure(result, "invalid-shape") : ok(data, result);
	}

	async search(
		mode: DiscrawlSearchMode,
		query: string,
		options?: DiscrawlSearchOptions,
	): Promise<DiscrawlSearchResult> {
		const args = ["search", "--mode", mode];
		if (options?.topK !== undefined) args.push("--limit", String(options.topK));
		if (this.options.guildId !== undefined) args.push("--guild", this.options.guildId);
		args.push(query);
		const result = await this.runJson(args, options?.signal);
		if (!result.ok) return toFailure(result);
		const hits = normalizeHits(result.stdout);
		return hits === undefined ? toFailure(result, "invalid-shape") : searchOk(hits, result);
	}

	private runJson(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		return this.run(["--json", ...args], signal);
	}

	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const violatingKey = findUserTokenKey({ ...process.env, ...(this.options.env ?? {}) });
		if (violatingKey !== undefined) {
			return { ok: false, reason: "user-token-rejected", stdout: "", stderr: "", code: null, violatingKey };
		}
		const env = controlledEnv(this.options.env);
		return spawnDiscrawl({
			options: this.options,
			args: [...commonArgs(this.options), ...args],
			env,
			signal,
		});
	}
}

export function discrawlWorkspace(options: DiscrawlOptions): string | undefined {
	return options.workspacePath;
}

/**
 * Global discrawl flags. discrawl exposes `--config` as a global option and
 * AutoRAG never forces an AutoRAG-managed config on it — without an explicit
 * `configPath`, discrawl uses its own default store
 * (`~/Library/Application Support/discrawl` on macOS).
 */
function commonArgs(options: DiscrawlOptions): readonly string[] {
	if (options.configPath !== undefined) return ["--config", options.configPath];
	if (options.embeddingRuntime !== undefined && (options.workspacePath ?? options.root) !== undefined) {
		return ["--config", managedConfigPath((options.workspacePath ?? options.root) as string)];
	}
	return [];
}

function managedConfigPath(workspace: string): string {
	return `${workspace}/.autorag/datasources/discrawl/config.toml`;
}

function spawnDiscrawl(request: SpawnRequest): Promise<ProcessResult> {
	return new Promise((resolveResult) => {
		const { options, args, env, signal } = request;
		const portable = portableSpawnCommand(options.binaryPath ?? DEFAULT_DISCRAWL_BINARY, args);
		const child = spawn(portable.command, [...portable.args], {
			env,
			...(request.cwd === undefined ? {} : { cwd: request.cwd }),
			detached: process.platform !== "win32",
			windowsHide: true,
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: DiscrawlFailure["reason"] | undefined;
		let pipeGuard: ReturnType<typeof setTimeout> | undefined;
		const maxBuffer = options.maxBufferBytes ?? DEFAULT_DISCRAWL_MAX_BUFFER_BYTES;
		const settleWith = (result: ProcessResult): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			if (pipeGuard !== undefined) clearTimeout(pipeGuard);
			signal?.removeEventListener("abort", abortHandler);
			resolveResult(result);
		};
		const finish = (code: number | null): void => {
			if (finalReason !== undefined) {
				settleWith({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			settleWith({
				ok: code === 0,
				reason: code === 0 ? undefined : "nonzero-exit",
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		};
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminateProcessTree(child, "SIGKILL");
		}, options.timeoutMs ?? DEFAULT_DISCRAWL_TIMEOUT_MS);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminateProcessTree(child, "SIGKILL");
		};
		if (signal?.aborted) abortHandler();
		signal?.addEventListener("abort", abortHandler, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendBounded(stdout, chunk, maxBuffer);
			if (stdout.capped) {
				finalReason = "stdout-too-large";
				terminateProcessTree(child, "SIGKILL");
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendBounded(stderr, chunk, maxBuffer);
			if (stderr.capped) {
				finalReason = "stderr-too-large";
				terminateProcessTree(child, "SIGKILL");
			}
		});
		child.on("exit", () => {
			if (settled) return;
			terminateProcessTree(child, "SIGKILL");
			pipeGuard = setTimeout(() => finish(child.exitCode), TREE_KILL_GRACE_MS);
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			const reason = error.code === "ENOENT" ? "binary-missing" : "spawn-error";
			settleWith({ ok: false, reason, stdout: stdout.text, stderr: describeSpawnFailure(reason), code: null });
		});
		child.on("close", (code) => finish(code));
	});
}

export function nativeGatewayBaseUrl(baseUrl: string): string {
	if (baseUrl.endsWith("/v1")) return baseUrl;
	let trimmed = baseUrl;
	while (trimmed.endsWith("/")) {
		trimmed = trimmed.slice(0, -1);
	}
	return `${trimmed}/v1`;
}

function managedDiscrawlConfig(runtime: DiscrawlEmbeddingRuntime): string {
	return [
		"# AutoRAG managed discrawl embeddings v1",
		"[search.embeddings]",
		"enabled = true",
		`provider = ${JSON.stringify(runtime.provider)}`,
		`model = ${JSON.stringify(runtime.model)}`,
		`base_url = ${JSON.stringify(runtime.baseUrl)}`,
		`dimensions = ${runtime.dimensions}`,
		"",
	].join("\n");
}

function parseManagedEmbedding(config: string): { provider?: string; model?: string; dimensions?: number } {
	const value = (key: string): string | undefined => {
		const match = config.match(new RegExp(`^${key}\\s*=\\s*(?:\\"([^\\"]*)\\"|([^\\s#]+))$`, "m"));
		return match?.[1] ?? match?.[2];
	};
	const dimensions = Number(value("dimensions"));
	return {
		provider: value("provider"),
		model: value("model"),
		...(Number.isFinite(dimensions) ? { dimensions } : {}),
	};
}

function controlledEnv(configuredEnv: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (value !== undefined && isAllowedDiscrawlEnvKey(key)) env[key] = value;
	}
	for (const [key, value] of Object.entries(configuredEnv ?? {})) {
		if (value === undefined) {
			delete env[key];
		} else if (isAllowedDiscrawlEnvKey(key)) {
			env[key] = value;
		}
	}
	return env;
}

function isAllowedDiscrawlEnvKey(key: string): boolean {
	return SAFE_INHERITED_ENV_KEYS.has(key) || key.startsWith(SAFE_DISCRAWL_ENV_PREFIX);
}

function findUserTokenKey(env: NodeJS.ProcessEnv): string | undefined {
	for (const [key, value] of Object.entries(env)) {
		if (value === undefined || value.length === 0) continue;
		if (USER_TOKEN_KEYS.has(key.toLowerCase())) return key;
	}
	return undefined;
}

/** Path-opaque stderr replacement (the raw Node error leaks the binary path). */
function describeSpawnFailure(reason: "binary-missing" | "spawn-error"): string {
	return reason === "binary-missing"
		? "the discrawl binary could not be found"
		: "the discrawl binary could not be started";
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

/** `doctor` and `embed` emit `key=value` lines rather than JSON. */
function parseKeyValueLines(stdout: string): Record<string, string> {
	const out: Record<string, string> = {};
	for (const line of stdout.split("\n")) {
		const trimmed = line.trim();
		if (trimmed.length === 0) continue;
		const eq = trimmed.indexOf("=");
		if (eq <= 0) continue;
		out[trimmed.slice(0, eq)] = trimmed.slice(eq + 1);
	}
	return out;
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

function normalizeDoctor(raw: Record<string, string>): DiscrawlDoctorInfo | undefined {
	if (Object.keys(raw).length === 0) return undefined;
	const configOk = raw.config === "ok";
	const databaseOk = raw.database === "ok";
	const ftsOk = raw.fts === "ok";
	const embeddingsOk = raw.embeddings === "ok" && raw.embeddings_probe !== "error";
	return {
		ready: configOk && databaseOk && ftsOk,
		configOk,
		databaseOk,
		ftsOk,
		embeddingsOk,
		...(raw.embeddings_model !== undefined ? { embeddingModel: raw.embeddings_model } : {}),
		...(raw.embeddings_provider !== undefined ? { embeddingProvider: raw.embeddings_provider } : {}),
		metadata: { ...raw },
	};
}

function normalizeMetadata(raw: Record<string, unknown>): DiscrawlMetadataInfo | undefined {
	const embeddings = asRecord(raw.embeddings) ?? raw;
	const provider = asString(embeddings.provider) ?? asString(raw.embeddings_provider);
	const model = asString(embeddings.model) ?? asString(raw.embeddings_model);
	const dimensionsValue =
		asNumber(embeddings.dimensions) ?? asNumber(raw.embeddings_dimensions) ?? asNumber(raw.dimension);
	const identityValue = asString(embeddings.identity) ?? asString(raw.embedding_identity);
	if (provider === undefined && model === undefined && dimensionsValue === undefined && identityValue === undefined)
		return {};
	return {
		...(provider !== undefined ? { embeddingProvider: provider } : {}),
		...(model !== undefined ? { embeddingModel: model } : {}),
		...(dimensionsValue !== undefined ? { embeddingDimensions: dimensionsValue } : {}),
		...(identityValue !== undefined ? { embeddingIdentity: identityValue } : {}),
		metadata: { ...raw },
	};
}

function normalizeStatus(raw: Record<string, unknown>): DiscrawlStatusInfo | undefined {
	const counts = raw.counts;
	if (!Array.isArray(counts)) return undefined;
	const byId = new Map<string, number>();
	for (const entry of counts) {
		const record = asRecord(entry);
		const id = asString(record?.id);
		const value = asNumber(record?.value);
		if (id !== undefined && value !== undefined) byId.set(id, value);
	}
	const messages = byId.get("messages");
	if (messages === undefined) return undefined;
	return {
		messages,
		channels: byId.get("channels") ?? 0,
		guilds: byId.get("guilds") ?? 0,
		...(asString(raw.database_path) !== undefined ? { databasePath: asString(raw.database_path) as string } : {}),
		metadata: { ...raw },
	};
}

function normalizeSync(raw: Record<string, unknown>): DiscrawlSyncInfo | undefined {
	const messages = asNumber(raw.messages);
	if (messages === undefined) return undefined;
	const guilds = asNumber(raw.guilds);
	const channels = asNumber(raw.channels);
	return {
		messages,
		...(guilds !== undefined ? { guilds } : {}),
		...(channels !== undefined ? { channels } : {}),
		metadata: { ...raw },
	};
}

function normalizeEmbed(raw: Record<string, string>): DiscrawlEmbedInfo | undefined {
	const processed = Number.parseInt(raw.processed ?? "", 10);
	if (!Number.isFinite(processed)) return undefined;
	const numeric = (key: string): number => {
		const parsed = Number.parseInt(raw[key] ?? "", 10);
		return Number.isFinite(parsed) ? parsed : 0;
	};
	return {
		processed,
		succeeded: numeric("succeeded"),
		failed: numeric("failed"),
		remainingBacklog: numeric("remaining_backlog"),
		...(raw.model !== undefined ? { model: raw.model } : {}),
		...(raw.provider !== undefined ? { provider: raw.provider } : {}),
		metadata: { ...raw },
	};
}

/**
 * `discrawl --json search` emits either an array of hits or an object wrapping
 * one under `results`/`messages`/`hits`. Both shapes are accepted; anything
 * else is reported as invalid rather than silently returning no hits.
 */
function normalizeHits(stdout: string): readonly DiscrawlSearchHit[] | undefined {
	const trimmed = stdout.trim();
	if (trimmed.length === 0) return [];
	let parsed: unknown;
	try {
		parsed = JSON.parse(trimmed);
	} catch {
		return undefined;
	}
	const rows = Array.isArray(parsed)
		? parsed
		: (() => {
				const record = asRecord(parsed);
				if (record === undefined) return undefined;
				for (const key of ["results", "messages", "hits", "rows"]) {
					if (Array.isArray(record[key])) return record[key] as unknown[];
				}
				return undefined;
			})();
	if (rows === undefined) return undefined;
	const hits: DiscrawlSearchHit[] = [];
	for (const [index, entry] of rows.entries()) {
		const record = asRecord(entry);
		if (record === undefined) return undefined;
		const messageId = asString(record.id) ?? asString(record.message_id) ?? asString(record.messageId);
		const content = asString(record.content) ?? asString(record.text) ?? "";
		if (messageId === undefined || messageId.length === 0) return undefined;
		const score = asNumber(record.score) ?? 1 / (index + 1);
		hits.push({
			messageId,
			content,
			score,
			...optionalString(record, ["channel_id", "channelId"], "channelId"),
			...optionalString(record, ["channel_name", "channelName", "channel"], "channelName"),
			...optionalString(record, ["guild_id", "guildId"], "guildId"),
			...optionalString(record, ["guild_name", "guildName", "guild"], "guildName"),
			...optionalString(record, ["author_name", "authorName", "author"], "authorName"),
			...optionalString(record, ["timestamp", "created_at", "createdAt"], "timestamp"),
			metadata: { ...record },
		});
	}
	return hits;
}

function optionalString(
	record: Record<string, unknown>,
	keys: readonly string[],
	target: string,
): Record<string, string> {
	for (const key of keys) {
		const value = asString(record[key]);
		if (value !== undefined && value.length > 0) return { [target]: value };
	}
	return {};
}

function searchOk(hits: readonly DiscrawlSearchHit[], result: ProcessResult): DiscrawlSearchResult {
	return { ok: true, hits, data: { hits }, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}

function toFailure(result: ProcessResult, reason?: DiscrawlFailure["reason"]): DiscrawlFailure {
	return {
		ok: false,
		reason: reason ?? result.reason ?? "nonzero-exit",
		stdout: result.stdout,
		stderr: sanitizeDiagnosticText(result.stderr),
		code: result.code,
		...(result.violatingKey !== undefined ? { violatingKey: result.violatingKey } : {}),
	};
}

/**
 * discrawl stderr can contain absolute archive/cache paths. Those are dropped
 * from diagnostics while the exit code is preserved, matching the katok and
 * rclone connectors.
 */
function sanitizeDiagnosticText(value: string): string {
	if (value.length === 0) return "";
	if (value.includes("/") || value.includes("\\")) {
		return "discrawl command failed; path details suppressed";
	}
	return value.trim().slice(0, 500);
}

function ok<T>(data: T, result: ProcessResult): DiscrawlOk<T> {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
