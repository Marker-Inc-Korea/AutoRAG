import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { resolveEmbeddingPrefixes } from "../embedding/prefixes.ts";
import {
	configuredLexicalLanguage,
	configuredMaxChunkSize,
	configuredVectorDimension,
	DEFAULT_MINSYNC_EMBEDDER_DIMENSION,
	DEFAULT_MINSYNC_EMBEDDER_ID,
	type MinSyncEmbeddingIdentity,
	minSyncConfigPath,
	minSyncEmbeddingIdentityPath,
	rewriteEmbedderConfig,
} from "./embedder-config.ts";
import { minSyncBinaryMissingError } from "./errors.ts";
import { kiwiAvailable } from "./kiwi.ts";
import { MINSYNC_LEXICAL_LANGUAGE, type MinSyncLexicalLanguage } from "./lexical-language.ts";
import { ensureLocalEmbedder } from "./local-embedder.ts";
import { type ProcessResult, spawnProcess } from "./process.ts";
import type {
	MinSyncEmbedderConfig,
	MinSyncFileFailure,
	MinSyncQueryHit,
	MinSyncSyncProgress,
	MinSyncSyncResult,
} from "./types.ts";

export interface MinSyncClientOptions {
	readonly binaryPath: string;
	readonly workspacePath: string;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
}

/** MinSync's native query modes: vector, BM25, and hybrid. */
export type MinSyncQueryMode = "vector" | "bm25" | "hybrid";

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

/** `minsync sync` exits 3 with "another sync is in progress" while another run holds `.minsync/lock`. */
const MINSYNC_LOCK_HELD_EXIT_CODE = 3;

/** `minsync sync` exits 6 when it completed but skipped files it could not read, chunk, or embed (`partial: true`). */
const MINSYNC_PARTIAL_SYNC_EXIT_CODE = 6;

export class MinSyncQueryError extends Error {
	readonly code: number | null;
	readonly stderr: string;
	readonly diagnostic = {
		code: "embedder-unavailable" as const,
		message: "Semantic embedder is unavailable.",
		retryable: true,
	};

	constructor(code: number | null, stderr: string) {
		super(stderr || `MinSync query failed with exit code ${code ?? "unknown"}`);
		this.name = "MinSyncQueryError";
		this.code = code;
		this.stderr = stderr;
	}
}

export class MinSyncClient {
	private readonly binaryPath: string;
	private readonly workspacePath: string;
	private readonly embedder: MinSyncEmbedderConfig | undefined;
	private readonly maxChunkSize: number | undefined;

	constructor(options: MinSyncClientOptions) {
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath;
		this.embedder = options.embedder;
		this.maxChunkSize = options.maxChunkSize;
	}

	/**
	 * What MinSync is told to embed with. No `id`, or a `native:` id, is MinSync's own Qwen3 engine, which
	 * MinSync provisions and runs (in-process, or through a `minsync serve` daemon it finds on its own).
	 * Anything else is an endpoint the operator configured, passed through untouched.
	 */
	private effectiveEmbedder(): { config: MinSyncEmbedderConfig; identity?: MinSyncEmbeddingIdentity } {
		const configured = this.embedder;
		const isNative =
			configured?.baseUrl === undefined && (configured?.id === undefined || configured.id.startsWith("native:"));
		if (!isNative) return { config: configured ?? {} };
		const id = configured?.id ?? DEFAULT_MINSYNC_EMBEDDER_ID;
		const dimension = configured?.dimension ?? DEFAULT_MINSYNC_EMBEDDER_DIMENSION;
		// An operator value, including "", wins; otherwise the model's own default applies (Qwen3's instruction).
		const prefixes = resolveEmbeddingPrefixes(configured);
		return {
			config: {
				...configured,
				id,
				dimension,
				...(configured?.queryPrefix === undefined && prefixes.query !== "" ? { queryPrefix: prefixes.query } : {}),
			},
			identity: {
				provider: "native",
				model: id.slice("native:".length),
				artifactRevision: "default",
				dimension,
				queryPrefix: prefixes.query,
				passagePrefix: prefixes.passage,
				runtimeBuild: "minsync-native",
			},
		};
	}

	private readIdentity(): MinSyncEmbeddingIdentity | undefined {
		try {
			const value: unknown = JSON.parse(readFileSync(minSyncEmbeddingIdentityPath(this.workspacePath), "utf8"));
			return isIdentity(value) ? value : undefined;
		} catch {
			return undefined;
		}
	}

	private writeIdentity(identity: MinSyncEmbeddingIdentity): void {
		writeFileSync(minSyncEmbeddingIdentityPath(this.workspacePath), `${JSON.stringify(identity, null, 2)}\n`);
	}

	private identityMismatch(identity: MinSyncEmbeddingIdentity): boolean {
		const previous = this.readIdentity();
		if (previous === undefined) return true;
		// Document vectors do not depend on the query prefix, so changing it alone must not force a re-embed.
		const { queryPrefix: _previousQuery, ...previousDocuments } = previous;
		const { queryPrefix: _currentQuery, ...currentDocuments } = identity;
		return JSON.stringify(previousDocuments) !== JSON.stringify(currentDocuments);
	}

	async sync(
		force = false,
		signal?: AbortSignal,
		onProgress?: (progress: MinSyncSyncProgress) => void,
	): Promise<MinSyncSyncResult> {
		if (!existsSync(this.binaryPath)) throw minSyncBinaryMissingError();
		const effective = this.effectiveEmbedder();
		try {
			await ensureLocalEmbedder({ baseUrl: effective.config.baseUrl, timeoutMs: effective.config.timeoutMs });
		} catch (error) {
			const reason = error instanceof Error ? error.message : "Semantic embedder is unavailable.";
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason,
				diagnostic: { code: "embedder-unavailable", message: reason, retryable: true },
			};
		}
		const embedder = effective.config;
		if (embedder.apiKeyEnv) {
			const envName = embedder.apiKeyEnv;
			if (!API_KEY_ENV_PATTERN.test(envName)) {
				return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "invalid-api-key-env" };
			}
			const envValue = process.env[envName];
			if (typeof envValue !== "string" || envValue.length === 0) {
				return {
					ok: false,
					synced: 0,
					workspacePath: this.workspacePath,
					reason: `missing-api-key-env:${envName}`,
				};
			}
		}
		// The embedder `timeoutMs` is a per-request embedding timeout: it is
		// written to MinSync's config.toml as `timeout_seconds` and governs
		// individual embedding HTTP requests inside the MinSync binary. It must
		// never double as a process-wide kill timer: a full reindex legitimately
		// runs for minutes or hours, and a small embedder timeout previously
		// SIGTERM'd `minsync sync` mid-run. Spawn init, check, and sync without a
		// process timeout so a long sync always runs to completion; only the
		// caller's `signal` can stop them, and it takes the whole process tree.
		const spawnOptions = signal === undefined ? {} : { signal };
		const lexical = this.resolveLexicalLanguage();
		const initialized = existsSync(minSyncConfigPath(this.workspacePath));

		if (!initialized) {
			const initArgs = ["init", "--format", "json"];
			if (this.embedder?.id) {
				initArgs.push("--embedder", this.embedder.id);
			}
			initArgs.push("--language", lexical.language);
			const init = await this.spawn(initArgs, spawnOptions);
			if (!init.ok || !existsSync(minSyncConfigPath(this.workspacePath))) {
				return {
					ok: false,
					synced: 0,
					workspacePath: this.workspacePath,
					reason: "init-failed",
				};
			}
		}
		// Read before the rewrite below: the rewrite changes what these report.
		const chunkSizeChanged =
			this.maxChunkSize !== undefined && configuredMaxChunkSize(this.workspacePath) !== this.maxChunkSize;
		const dimensionChanged =
			embedder.dimension !== undefined && configuredVectorDimension(this.workspacePath) !== embedder.dimension;
		const configPath = minSyncConfigPath(this.workspacePath);
		const originalConfig = readConfigSnapshot(configPath);
		const configRewritten = rewriteEmbedderConfig(this.workspacePath, embedder, {
			maxChunkSize: this.maxChunkSize,
			lexicalLanguage: lexical.language,
		});
		const restoreConfig = () => {
			if (configRewritten && originalConfig !== undefined) writeFileSync(configPath, originalConfig);
		};
		try {
			const result = await this.checkAndSync(
				force || chunkSizeChanged || dimensionChanged,
				signal,
				effective,
				restoreConfig,
				onProgress,
			);
			return result.ok && lexical.fallback !== undefined ? { ...result, lexicalFallback: lexical.fallback } : result;
		} catch (error) {
			// An aborted run must not leave this run's embedder settings in the managed config.
			restoreConfig();
			throw error;
		}
	}

	/**
	 * The tokenizer to run this sync with: always {@link MINSYNC_LEXICAL_LANGUAGE}, which fails every sync when the
	 * Kiwi native library is missing. A workspace that is new, or tokenized differently, then gets whitespace
	 * tokenization instead and the sync reports `lexicalFallback`. An index already built with the multilingual
	 * tokenizer is left alone: switching it would re-embed every chunk, and the probe can miss a library that the OS
	 * loader finds on its own.
	 */
	private resolveLexicalLanguage(): {
		readonly language: MinSyncLexicalLanguage;
		readonly fallback?: { readonly requested: MinSyncLexicalLanguage; readonly used: MinSyncLexicalLanguage };
	} {
		const requested = MINSYNC_LEXICAL_LANGUAGE;
		if (kiwiAvailable() || configuredLexicalLanguage(this.workspacePath) === requested)
			return { language: requested };
		return { language: "simple", fallback: { requested, used: "simple" } };
	}

	private lockHeldResult(stderr: string): MinSyncSyncResult {
		return {
			ok: false,
			synced: 0,
			workspacePath: this.workspacePath,
			reason: "lock-held",
			diagnostic: {
				code: "minsync-lock-held",
				message: stderr ? stderr.trim() : "Another MinSync sync holds the workspace lock.",
				retryable: true,
			},
		};
	}

	private async checkAndSync(
		forceFull: boolean,
		signal: AbortSignal | undefined,
		effective: { config: MinSyncEmbedderConfig; identity?: MinSyncEmbeddingIdentity },
		restoreConfig: () => void,
		onProgress?: (progress: MinSyncSyncProgress) => void,
	): Promise<MinSyncSyncResult> {
		const spawnOptions = signal === undefined ? {} : { signal };
		const cursorPath = join(this.workspacePath, ".minsync", "cursor.json");
		const check = await this.spawn(["check", "--format", "json"], spawnOptions);
		// MinSync takes the workspace lock for `check` as well as `sync`, so a held lock surfaces at either stage.
		if (!check.ok && check.code === MINSYNC_LOCK_HELD_EXIT_CODE) {
			restoreConfig();
			return this.lockHeldResult(check.stderr);
		}
		if (!check.ok) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: "check-failed",
				diagnostic: {
					code: "sync-failed",
					message: check.stderr ? check.stderr.trim() : "MinSync check failed.",
					retryable: true,
				},
			};
		}
		const checkFailure = readCheckFailure(check.stdout);
		if (checkFailure) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: checkFailure,
				diagnostic: {
					// Only an embedder preflight failure is an embedder problem; a vector store or generic
					// preflight failure is not.
					code: checkFailure.includes("embedder") ? "embedder-unavailable" : "sync-failed",
					message: checkFailure,
					retryable: true,
				},
			};
		}
		const identityChanged = effective.identity !== undefined && this.identityMismatch(effective.identity);
		const fullReindex = forceFull || identityChanged;
		const syncArgs =
			existsSync(cursorPath) && !fullReindex
				? ["sync", "--format", "json", "--progress", "jsonl"]
				: ["sync", "--full", "--format", "json", "--progress", "jsonl"];
		const result = await this.spawn(syncArgs, {
			...spawnOptions,
			...(onProgress === undefined
				? {}
				: {
						onStdoutLine: (line: string) => {
							const update = readSyncProgress(line);
							if (update === undefined) return;
							try {
								onProgress(update);
							} catch {
								// Progress is observational; a host callback must not interrupt the sync.
							}
						},
					}),
		});
		if (!result.ok && result.code === MINSYNC_LOCK_HELD_EXIT_CODE) {
			restoreConfig();
			return this.lockHeldResult(result.stderr);
		}
		const partial =
			!result.ok && result.code === MINSYNC_PARTIAL_SYNC_EXIT_CODE ? readPartialSync(result.stdout) : undefined;
		if (!result.ok && partial === undefined) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: "sync-failed",
				diagnostic: {
					code: "sync-failed",
					message: result.stderr ? result.stderr.trim() : "MinSync sync failed.",
					retryable: true,
				},
			};
		}
		if (!existsSync(cursorPath)) {
			restoreConfig();
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "not-ready: missing cursor" };
		}
		if (effective.identity) this.writeIdentity(effective.identity);
		const filesFailed = partial?.filesFailed ?? [];
		return {
			ok: true,
			synced: readSyncedCount(result.stdout),
			workspacePath: this.workspacePath,
			...(filesFailed.length > 0 ? { filesFailed } : {}),
			...(identityChanged
				? {
						diagnostic: {
							code: "embedding-identity-mismatch" as const,
							message: "Embedding identity changed; MinSync performed a full reindex.",
						},
					}
				: {}),
		};
	}

	async query(
		text: string,
		topK: number,
		mode: MinSyncQueryMode = "vector",
		signal?: AbortSignal,
	): Promise<readonly MinSyncQueryHit[]> {
		if (!existsSync(this.binaryPath)) throw minSyncBinaryMissingError();
		const result = await this.spawn(["query", "--format", "json", "--mode", mode, "-k", String(topK), text], {
			...(signal === undefined ? {} : { signal }),
		});
		if (!result.ok) throw new MinSyncQueryError(result.code, result.stderr);
		return parseQueryHits(result.stdout);
	}

	private async spawn(
		args: readonly string[],
		options: {
			readonly timeoutMs?: number;
			readonly signal?: AbortSignal;
			readonly onStdoutLine?: (line: string) => void;
		} = {},
	): Promise<ProcessResult> {
		return spawnProcess(this.binaryPath, args, this.workspacePath, { ...options, env: this.childEnv() });
	}

	/**
	 * MinSync's `openai:` embedder reads its bearer token from `OPENAI_API_KEY` only. The operator names the
	 * variable that holds their key with `apiKeyEnv`, so that value is what the child receives as
	 * `OPENAI_API_KEY`. A local `native:` embedder needs no key and the environment is left as it is.
	 */
	private childEnv(): NodeJS.ProcessEnv | undefined {
		const name = this.embedder?.apiKeyEnv;
		if (name === undefined || !API_KEY_ENV_PATTERN.test(name)) return undefined;
		const key = process.env[name];
		return key === undefined || key.length === 0 ? undefined : { ...process.env, OPENAI_API_KEY: key };
	}
}

function readConfigSnapshot(configPath: string): string | undefined {
	try {
		return readFileSync(configPath, "utf8");
	} catch (error) {
		if ((error as NodeJS.ErrnoException).code === "ENOENT") return undefined;
		throw error;
	}
}

function readPartialSync(stdout: string): { readonly filesFailed: readonly MinSyncFileFailure[] } | undefined {
	const parsed = parseSyncOutput(stdout);
	if (!isRecord(parsed) || parsed.partial !== true) return undefined;
	const failures = Array.isArray(parsed.files_failed) ? parsed.files_failed : [];
	return {
		filesFailed: failures.filter(isRecord).map((failure) => ({
			path: typeof failure.path === "string" ? failure.path : "(unknown)",
			stage: typeof failure.stage === "string" ? failure.stage : "unknown",
			error: typeof failure.error === "string" ? failure.error : "unknown error",
		})),
	};
}

function readSyncedCount(stdout: string): number {
	const parsed = parseSyncOutput(stdout);
	if (!isRecord(parsed)) return 0;
	for (const key of ["files_processed", "synced"]) {
		const count = parsed[key];
		if (typeof count === "number" && Number.isFinite(count)) return count;
	}
	return 0;
}

function readSyncProgress(line: string): MinSyncSyncProgress | undefined {
	const event = parseJson(line);
	if (!isRecord(event) || !["start", "batch", "heartbeat", "done"].includes(String(event.event))) return undefined;
	const total = event.total ?? event.total_files;
	const processed = event.event === "start" ? (event.resumed ?? 0) : event.processed;
	if (
		typeof processed !== "number" ||
		!Number.isInteger(processed) ||
		processed < 0 ||
		typeof total !== "number" ||
		!Number.isInteger(total) ||
		total < processed
	)
		return undefined;
	return { processed, total };
}

function parseSyncOutput(stdout: string): unknown {
	const direct = parseJson(stdout);
	if (direct !== undefined) return direct;
	const lines = stdout.trim().split("\n");
	for (let index = lines.length - 1; index >= 0; index -= 1) {
		const parsed = parseJson(lines[index]);
		if (parsed !== undefined) return parsed;
	}
	return undefined;
}

function readCheckFailure(stdout: string): string | undefined {
	const parsed = parseJson(stdout);
	if (!isRecord(parsed)) return "check-failed: invalid response";
	if (parsed.embedder_ok !== true) return "check-failed: embedder unavailable";
	if (parsed.vectorstore_ok !== true) return "check-failed: vector store unavailable";
	if (parsed.all_passed === false) return "check-failed: preflight unhealthy";
	return undefined;
}

function parseQueryHits(stdout: string): readonly MinSyncQueryHit[] {
	const parsed = parseJson(stdout);
	const candidates = Array.isArray(parsed) ? parsed : isRecord(parsed) ? parsed.results : [];
	if (!Array.isArray(candidates)) return [];
	return candidates.filter(isMinSyncQueryHit).map((hit) => {
		const docId = isRecord(hit) ? hit.doc_id : undefined;
		return typeof docId === "string" && docId.length > 0
			? { path: hit.path, score: hit.score, text: hit.text, docId }
			: { path: hit.path, score: hit.score, text: hit.text };
	});
}

function parseJson(text: string): unknown {
	try {
		return JSON.parse(text);
	} catch (error) {
		if (error instanceof SyntaxError) return undefined;
		throw error;
	}
}

function isMinSyncQueryHit(value: unknown): value is MinSyncQueryHit {
	if (!isRecord(value)) return false;
	return typeof value.path === "string" && typeof value.score === "number" && typeof value.text === "string";
}

function isIdentity(value: unknown): value is MinSyncEmbeddingIdentity {
	if (!isRecord(value)) return false;
	return ["provider", "model", "artifactRevision", "dimension", "queryPrefix", "passagePrefix", "runtimeBuild"].every(
		(key) => key in value,
	);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
