import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { configuredMaxChunkSize, minSyncConfigPath, rewriteEmbedderConfig } from "./embedder-config.ts";
import { spawnProcess } from "./process.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";

export interface MinSyncClientOptions {
	readonly binaryPath: string;
	readonly workspacePath: string;
	readonly embedder?: MinSyncEmbedderConfig;
	readonly maxChunkSize?: number;
}

export type MinSyncQueryMode = "vector" | "bm25" | "hybrid";

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;

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

	async sync(): Promise<MinSyncSyncResult> {
		if (!existsSync(this.binaryPath)) {
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "missing-binary" };
		}
		if (this.embedder?.apiKeyEnv) {
			const envName = this.embedder.apiKeyEnv;
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
		const spawnOpts = this.embedder?.timeoutMs !== undefined ? { timeoutMs: this.embedder.timeoutMs } : {};
		const initialized = existsSync(minSyncConfigPath(this.workspacePath));
		const cursorPath = join(this.workspacePath, ".minsync", "cursor.json");
		if (!initialized) {
			const initArgs = ["init", "--format", "json"];
			if (this.embedder?.id) {
				initArgs.push("--embedder", this.embedder.id);
			}
			const init = await this.spawn(initArgs, spawnOpts);
			if (!init.ok) {
				return {
					ok: false,
					synced: 0,
					workspacePath: this.workspacePath,
					reason: "init-failed",
				};
			}
		}
		const configuredChunkSize = configuredMaxChunkSize(this.workspacePath);
		const configPath = minSyncConfigPath(this.workspacePath);
		const shouldRewriteConfig = this.embedder !== undefined || this.maxChunkSize !== undefined;
		const originalConfig = shouldRewriteConfig ? readConfigSnapshot(configPath) : undefined;
		const configRewritten =
			shouldRewriteConfig &&
			rewriteEmbedderConfig(this.workspacePath, this.embedder ?? {}, { maxChunkSize: this.maxChunkSize });
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
			};
		}
		const checkFailure = readCheckFailure(check.stdout);
		if (checkFailure) {
			restoreConfig();
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: checkFailure };
		}
		const chunkSizeChanged = this.maxChunkSize !== undefined && configuredChunkSize !== this.maxChunkSize;
		const syncArgs =
			existsSync(cursorPath) && !chunkSizeChanged
				? ["sync", "--format", "json"]
				: ["sync", "--full", "--format", "json"];
		const result = await this.spawn(syncArgs, spawnOpts);
		if (!result.ok) {
			restoreConfig();
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: "sync-failed",
			};
		}
		if (!existsSync(cursorPath)) {
			restoreConfig();
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "not-ready: missing cursor" };
		}
		return { ok: true, synced: readSyncedCount(result.stdout), workspacePath: this.workspacePath };
	}

	async query(text: string, topK: number, mode: MinSyncQueryMode = "vector"): Promise<readonly MinSyncQueryHit[]> {
		if (!existsSync(this.binaryPath)) return [];
		const result = await this.spawn(["query", "--format", "json", "-k", String(topK), "--mode", mode, text]);
		if (!result.ok) return [];
		return parseQueryHits(result.stdout);
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

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null;
}
