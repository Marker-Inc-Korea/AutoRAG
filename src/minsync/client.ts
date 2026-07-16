import { existsSync } from "node:fs";
import { rewriteEmbedderConfig } from "./embedder-config.ts";
import { spawnProcess } from "./process.ts";
import type { MinSyncEmbedderConfig, MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";

export interface MinSyncClientOptions {
	readonly binaryPath: string;
	readonly workspacePath: string;
	readonly embedder?: MinSyncEmbedderConfig;
}

const API_KEY_ENV_PATTERN = /^[A-Za-z_][A-Za-z0-9_]*$/;
const SECRET_PATTERN = /sk-[A-Za-z0-9_-]+/g;

export class MinSyncClient {
	private readonly binaryPath: string;
	private readonly workspacePath: string;
	private readonly embedder: MinSyncEmbedderConfig | undefined;

	constructor(options: MinSyncClientOptions) {
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath;
		this.embedder = options.embedder;
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
		const initArgs = ["init", "--format", "json"];
		if (this.embedder?.id) {
			initArgs.push("--embedder", this.embedder.id);
		}
		const spawnOpts = this.embedder?.timeoutMs !== undefined ? { timeoutMs: this.embedder.timeoutMs } : {};
		const init = await spawnProcess(this.binaryPath, initArgs, this.workspacePath, spawnOpts);
		if (!init.ok) {
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: sanitizeReason(init.stderr || "init-failed"),
			};
		}
		if (this.embedder) {
			rewriteEmbedderConfig(this.workspacePath, this.embedder);
		}
		const result = await spawnProcess(this.binaryPath, ["sync", "--format", "json"], this.workspacePath, spawnOpts);
		if (!result.ok) {
			return {
				ok: false,
				synced: 0,
				workspacePath: this.workspacePath,
				reason: sanitizeReason(result.stderr || "sync-failed"),
			};
		}
		return { ok: true, synced: readSyncedCount(result.stdout), workspacePath: this.workspacePath };
	}

	async query(text: string, topK: number): Promise<readonly MinSyncQueryHit[]> {
		if (!existsSync(this.binaryPath)) return [];
		const result = await spawnProcess(
			this.binaryPath,
			["query", "--format", "json", "-k", String(topK), text],
			this.workspacePath,
		);
		if (!result.ok) return [];
		return parseQueryHits(result.stdout);
	}
}

function sanitizeReason(reason: string): string {
	return reason.replace(SECRET_PATTERN, "[redacted]");
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

function parseQueryHits(stdout: string): readonly MinSyncQueryHit[] {
	const parsed = parseJson(stdout);
	const candidates = isRecord(parsed) ? parsed.results : parsed;
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
