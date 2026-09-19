import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import type { EnsuredRuntime } from "../embedding-runtime/index.ts";
import {
	configuredMaxChunkSize,
	configuredVectorDimension,
	type MinSyncEmbeddingIdentity,
	minSyncConfigPath,
	minSyncEmbeddingIdentityPath,
	rewriteEmbedderConfig,
} from "./embedder-config.ts";
import { ensureLocalEmbedder } from "./local-embedder.ts";
import { spawnProcess } from "./process.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";

export const MINSYNC_OLLAMA_MIGRATION_MESSAGE =
	"This workspace uses the legacy 768-dimensional Ollama/TEI embedding path. Reindex explicitly, or pin an explicit profile config before using the new default runtime.";

export interface MinSyncRuntime {
	ensureRuntime(options?: {
		readonly profileId?: MinSyncEmbedderConfig["profile"];
		readonly cachedOnly?: boolean;
	}): Promise<EnsuredRuntime>;
}

export interface MinSyncClientOptions {
	readonly binaryPath: string;
	readonly workspacePath: string;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
	readonly runtime?: MinSyncRuntime;
}

/** MinSync v0.4.2 supports vector, BM25, and hybrid query modes. */
export type MinSyncQueryMode = "vector" | "bm25" | "hybrid";

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

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
	private readonly runtime: MinSyncRuntime | undefined;

	constructor(options: MinSyncClientOptions) {
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath;
		this.embedder = options.embedder;
		this.maxChunkSize = options.maxChunkSize;
		this.runtime = options.runtime;
	}

	private async effectiveEmbedder(): Promise<{
		config: MinSyncEmbedderConfig;
		identity?: MinSyncEmbeddingIdentity;
		runtimeUnavailable?: boolean;
		runtimeReason?: string;
	}> {
		const configured = this.embedder;
		const useRuntime =
			this.runtime !== undefined &&
			(configured === undefined || (configured.profile !== undefined && configured.baseUrl === undefined));
		if (!useRuntime) return { config: configured ?? {} };
		try {
			const ensured = await this.runtime?.ensureRuntime({ profileId: configured?.profile, cachedOnly: true });
			if (!ensured) throw new Error("Embedding runtime did not return a runtime");
			return {
				config: {
					id: `tei:${ensured.profile.model}`,
					baseUrl: ensured.baseUrl,
					dimension: ensured.profile.dimension,
					queryPrefix: ensured.profile.queryPrefix,
					passagePrefix: ensured.profile.passagePrefix,
					// A profile selects the embedding model only. Batching, concurrency,
					// retry, and timeout settings belong to the operator, so they must
					// still reach MinSync's config instead of falling back to MinSync's
					// own defaults.
					...(configured?.batchSize !== undefined ? { batchSize: configured.batchSize } : {}),
					...(configured?.maxRetries !== undefined ? { maxRetries: configured.maxRetries } : {}),
					...(configured?.maxConcurrent !== undefined ? { maxConcurrent: configured.maxConcurrent } : {}),
					...(configured?.timeoutMs !== undefined ? { timeoutMs: configured.timeoutMs } : {}),
				},
				identity: {
					provider: ensured.identity.provider,
					model: ensured.identity.model,
					artifactRevision: ensured.identity.modelRevision,
					dimension: ensured.identity.dimension,
					queryPrefix: ensured.profile.queryPrefix,
					passagePrefix: ensured.profile.passagePrefix,
					runtimeBuild: ensured.identity.runtimeBuild,
				},
			};
		} catch (error) {
			return {
				config: {},
				runtimeUnavailable: true,
				runtimeReason: error instanceof Error ? error.message : undefined,
			};
		}
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
		return previous === undefined || JSON.stringify(previous) !== JSON.stringify(identity);
	}

	async sync(force = false): Promise<MinSyncSyncResult> {
		if (!existsSync(this.binaryPath)) {
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "missing-binary" };
		}
		let effective: {
			config: MinSyncEmbedderConfig;
			identity?: MinSyncEmbeddingIdentity;
			runtimeUnavailable?: boolean;
			runtimeReason?: string;
		} = { config: {} };
		try {
			effective = await this.effectiveEmbedder();
			if (effective.runtimeUnavailable)
				throw new Error(
					effective.runtimeReason ??
						"Semantic embedder is unavailable; run autorag models prefetch (or models import).",
				);
			await ensureLocalEmbedder({ baseUrl: effective.config.baseUrl, timeoutMs: effective.config.timeoutMs });
		} catch (error) {
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason:
					error instanceof Error
						? error.message
						: (effective.runtimeReason ??
							"Semantic embedder is unavailable; run autorag models prefetch (or models import)."),
				diagnostic: {
					code: "embedder-unavailable",
					message: "Semantic embedder is unavailable; run autorag models prefetch (or models import).",
					retryable: true,
				},
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
		const spawnOpts = embedder.timeoutMs !== undefined ? { timeoutMs: embedder.timeoutMs } : {};
		const initialized = existsSync(minSyncConfigPath(this.workspacePath));
		const cursorPath = join(this.workspacePath, ".minsync", "cursor.json");
		if (!initialized) {
			const initArgs = ["init", "--format", "json"];
			if (embedder.id) {
				initArgs.push("--embedder", embedder.id);
			}
			const init = await this.spawn(initArgs, spawnOpts);
			if (!init.ok || !existsSync(minSyncConfigPath(this.workspacePath))) {
				return {
					ok: false,
					synced: 0,
					workspacePath: this.workspacePath,
					reason: "init-failed",
				};
			}
		}
		const configuredChunkSize = configuredMaxChunkSize(this.workspacePath);
		const configuredDimension = configuredVectorDimension(this.workspacePath);
		const configPath = minSyncConfigPath(this.workspacePath);
		const shouldRewriteConfig =
			this.embedder !== undefined || this.maxChunkSize !== undefined || effective.identity !== undefined;
		const originalConfig = shouldRewriteConfig ? readConfigSnapshot(configPath) : undefined;
		const configRewritten =
			shouldRewriteConfig &&
			rewriteEmbedderConfig(this.workspacePath, embedder, { maxChunkSize: this.maxChunkSize });
		const restoreConfig = () => {
			if (configRewritten && originalConfig !== undefined) writeFileSync(configPath, originalConfig);
		};
		const check = await this.spawn(["check", "--format", "json"], spawnOpts);
		if (!check.ok) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: "check-failed",
				diagnostic: {
					code: "embedder-unavailable",
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
					code: "embedder-unavailable",
					message: checkFailure,
					retryable: true,
				},
			};
		}
		const chunkSizeChanged = this.maxChunkSize !== undefined && configuredChunkSize !== this.maxChunkSize;
		const dimensionChanged = embedder.dimension !== undefined && configuredDimension !== embedder.dimension;
		const identityChanged = effective.identity !== undefined && this.identityMismatch(effective.identity);
		const fullReindex = force || chunkSizeChanged || dimensionChanged || identityChanged;
		const syncArgs =
			existsSync(cursorPath) && !fullReindex ? ["sync", "--format", "json"] : ["sync", "--full", "--format", "json"];
		const result = await this.spawn(syncArgs, spawnOpts);
		if (!result.ok) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: "sync-failed",
				diagnostic: {
					code: "embedder-unavailable",
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
		return {
			ok: true,
			synced: readSyncedCount(result.stdout),
			workspacePath: this.workspacePath,
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

	async query(text: string, topK: number, mode: MinSyncQueryMode = "vector"): Promise<readonly MinSyncQueryHit[]> {
		if (!existsSync(this.binaryPath)) return [];
		const configPath = minSyncConfigPath(this.workspacePath);
		const configuredDimension = configuredVectorDimension(this.workspacePath);
		const effective = mode === "bm25" ? { config: this.embedder ?? {} } : await this.effectiveEmbedder();
		if (
			effective.config.dimension !== undefined &&
			configuredDimension !== undefined &&
			effective.config.dimension !== configuredDimension
		) {
			const migration =
				configuredDimension === 768 && effective.config.dimension !== 768
					? ` ${MINSYNC_OLLAMA_MIGRATION_MESSAGE}`
					: "";
			throw new MinSyncQueryError(
				null,
				`configured embedder dimension ${effective.config.dimension} does not match indexed dimension ${configuredDimension}; reindex required.${migration}`,
			);
		}
		if (effective.identity !== undefined && this.identityMismatch(effective.identity)) {
			throw new MinSyncQueryError(
				null,
				"embedding identity does not match the indexed workspace; full reindex required",
			);
		}
		const shouldRewriteConfig = this.embedder !== undefined || effective.identity !== undefined;
		const originalConfig = shouldRewriteConfig ? readConfigSnapshot(configPath) : undefined;
		const configRewritten =
			shouldRewriteConfig && rewriteEmbedderConfig(this.workspacePath, effective.config) === true;
		try {
			if (effective.runtimeUnavailable)
				throw new MinSyncQueryError(
					null,
					effective.runtimeReason ??
						"Semantic embedder is unavailable; run autorag models prefetch (or models import).",
				);
			await ensureLocalEmbedder({ baseUrl: effective.config.baseUrl, timeoutMs: effective.config.timeoutMs });
			const result = await this.spawn(["query", "--format", "json", "--mode", mode, "-k", String(topK), text]);
			if (!result.ok) throw new MinSyncQueryError(result.code, result.stderr);
			return parseQueryHits(result.stdout);
		} finally {
			if (configRewritten && originalConfig !== undefined) writeFileSync(configPath, originalConfig);
		}
	}

	private async spawn(
		args: readonly string[],
		options: { readonly timeoutMs?: number } = {},
	): Promise<ReturnType<typeof spawnProcess> extends Promise<infer T> ? T : never> {
		return spawnProcess(this.binaryPath, args, this.workspacePath, options);
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

function readSyncedCount(stdout: string): number {
	const parsed = parseJson(stdout);
	if (!isRecord(parsed)) return 0;
	for (const key of ["files_processed", "synced"]) {
		const count = parsed[key];
		if (typeof count === "number" && Number.isFinite(count)) return count;
	}
	return 0;
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
	return candidates.filter(isMinSyncQueryHit);
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
