import { existsSync } from "node:fs";
import { spawnProcess } from "./process.ts";
import type { MinSyncQueryHit, MinSyncSyncResult } from "./types.ts";

export interface MinSyncClientOptions {
	readonly binaryPath: string;
	readonly workspacePath: string;
}

export class MinSyncClient {
	private readonly binaryPath: string;
	private readonly workspacePath: string;

	constructor(options: MinSyncClientOptions) {
		this.binaryPath = options.binaryPath;
		this.workspacePath = options.workspacePath;
	}

	async sync(): Promise<MinSyncSyncResult> {
		if (!existsSync(this.binaryPath)) {
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: "missing-binary" };
		}
		const init = await spawnProcess(this.binaryPath, ["init", "--format", "json"], this.workspacePath);
		if (!init.ok) {
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: init.stderr || "init-failed" };
		}
		const result = await spawnProcess(this.binaryPath, ["sync", "--format", "json"], this.workspacePath);
		if (!result.ok) {
			return { ok: false, synced: 0, workspacePath: this.workspacePath, reason: result.stderr || "sync-failed" };
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
