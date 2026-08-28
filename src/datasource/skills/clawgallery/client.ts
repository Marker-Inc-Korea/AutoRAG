import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { portableSpawnCommand } from "../../../process/portable-spawn.ts";
import type {
	ClawGalleryFailure,
	ClawGalleryHit,
	ClawGalleryIndexResult,
	ClawGalleryOptions,
	ClawGallerySearchMode,
	ClawGallerySearchOptions,
	ClawGallerySearchResult,
	ClawGalleryVdrResult,
} from "./types.ts";

const DEFAULT_BINARY = "clawgallery";
const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_BUFFER_BYTES = 4 * 1024 * 1024;
const SAFE_ENV = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP", "NO_COLOR"]);

export class ClawGalleryClient {
	private readonly options: ClawGalleryOptions;

	constructor(options: ClawGalleryOptions = {}) {
		this.options = options;
	}

	async bootstrap(signal?: AbortSignal): Promise<ClawGalleryIndexResult> {
		if (this.options.path !== undefined && !existsSync(this.options.path)) {
			return {
				ok: false,
				reason: "nonzero-exit",
				stdout: "",
				stderr: "configured ClawGallery folder does not exist",
				code: null,
			};
		}
		const args = ["bootstrap"];
		if (this.options.path !== undefined) args.push("--path", this.options.path);
		const result = await this.run(args, signal);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		const indexed =
			(parsed !== undefined ? numberValue(parsed.indexed ?? parsed.added ?? parsed.images) : undefined) ??
			parseSummaryCount(result.stdout, "ingested");
		if (indexed === undefined) return failure(result, "invalid-shape");
		return ok(
			{
				indexed,
				skipped: parsed ? (numberValue(parsed.skipped) ?? numberValue(parsed.unchanged) ?? 0) : 0,
				pruned: parsed ? (numberValue(parsed.pruned) ?? 0) : 0,
				...(parsed ? { metadata: parsed } : {}),
			},
			result,
		);
	}

	async syncVisual(signal?: AbortSignal): Promise<ClawGalleryVdrResult> {
		const args = ["vdr", "sync"];
		if (this.options.vdrBackend !== undefined) args.push("--backend", this.options.vdrBackend);
		const result = await this.run(args, signal);
		if (!result.ok) return failure(result);
		const parsed = parseObject(result.stdout);
		const processed =
			(parsed !== undefined ? numberValue(parsed.processed ?? parsed.indexed ?? parsed.images) : undefined) ??
			parseSummaryCount(result.stdout, "indexed");
		if (processed === undefined) return failure(result, "invalid-shape");
		return ok(
			{
				processed,
				skipped: parsed ? (numberValue(parsed.skipped) ?? numberValue(parsed.unchanged) ?? 0) : 0,
				failed: parsed ? (numberValue(parsed.failed) ?? 0) : 0,
				...(parsed ? { metadata: parsed } : {}),
			},
			result,
		);
	}

	async search(
		mode: ClawGallerySearchMode,
		query: string,
		options?: ClawGallerySearchOptions,
	): Promise<ClawGallerySearchResult> {
		const args = ["search", "--json", "--mode", mode];
		if (options?.topK !== undefined) args.push("--limit", String(options.topK));
		args.push(query);
		const result = await this.run(args, options?.signal);
		if (!result.ok) return failure(result);
		const parsed = parseJsonLines(result.stdout);
		const rows = Array.isArray(parsed)
			? parsed
			: parsed !== undefined && typeof parsed === "object"
				? ((parsed as Record<string, unknown>).results ??
					(parsed as Record<string, unknown>).hits ??
					(parsed as Record<string, unknown>).images)
				: undefined;
		if (!Array.isArray(rows)) return failure(result, "invalid-shape");
		const hits: ClawGalleryHit[] = [];
		for (const [index, value] of rows.entries()) {
			if (value === null || typeof value !== "object" || Array.isArray(value))
				return failure(result, "invalid-shape");
			const row = value as Record<string, unknown>;
			const imageId = stringValue(row.image_id ?? row.imageId ?? row.id) ?? stringValue(row.path);
			if (imageId === undefined) return failure(result, "invalid-shape");
			const content = stringValue(row.content ?? row.caption ?? row.title ?? row.path) ?? "";
			hits.push({
				imageId,
				content,
				score: numberValue(row.score) ?? 1 / (index + 1),
				...(stringValue(row.path ?? row.file) !== undefined ? { path: stringValue(row.path ?? row.file) } : {}),
				...(stringValue(row.title) !== undefined ? { title: stringValue(row.title) } : {}),
				...(stringValue(row.caption) !== undefined ? { caption: stringValue(row.caption) } : {}),
				metadata: row as Record<string, unknown>,
			});
		}
		return { ...ok({ hits }, result), hits };
	}

	private run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const env: NodeJS.ProcessEnv = {};
		for (const key of SAFE_ENV) if (process.env[key] !== undefined) env[key] = process.env[key];
		for (const [key, value] of Object.entries(this.options.env ?? {})) {
			if (value === undefined) delete env[key];
			else if (key === "CLAWGALLERY_PYTHON" || key.startsWith("CLAWGALLERY_")) env[key] = value;
		}
		if (this.options.configDir !== undefined) env.CLAWGALLERY_CONFIG_DIR = this.options.configDir;
		return spawnCli(this.options.binaryPath ?? DEFAULT_BINARY, args, env, this.options, signal);
	}
}

type ProcessResult = {
	readonly ok: boolean;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly reason?: ClawGalleryFailure["reason"];
};

function spawnCli(
	binary: string,
	args: readonly string[],
	env: NodeJS.ProcessEnv,
	options: ClawGalleryOptions,
	signal?: AbortSignal,
): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const portable = portableSpawnCommand(binary, args);
		const child = spawn(portable.command, [...portable.args], { env, stdio: ["ignore", "pipe", "pipe"] });
		let stdout = "";
		let stderr = "";
		let settled = false;
		let reason: ClawGalleryFailure["reason"] | undefined;
		const max = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
		const timeout = setTimeout(() => {
			reason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		const abort = (): void => {
			reason = "aborted";
			terminate(child);
		};
		if (signal?.aborted) abort();
		signal?.addEventListener("abort", abort, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout += chunk;
			if (Buffer.byteLength(stdout) > max) {
				reason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr += chunk;
			if (Buffer.byteLength(stderr) > max) {
				reason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error: NodeJS.ErrnoException) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abort);
			resolve({
				ok: false,
				stdout,
				stderr: error.code === "ENOENT" ? "clawgallery binary not found" : "clawgallery could not start",
				code: null,
				reason: error.code === "ENOENT" ? "binary-missing" : "spawn-error",
			});
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abort);
			resolve({
				ok: reason === undefined && code === 0,
				stdout,
				stderr: stderr.slice(0, max),
				code,
				reason: reason ?? (code === 0 ? undefined : "nonzero-exit"),
			});
		});
	});
}

function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}
function parseJson(value: string): Record<string, unknown> | readonly unknown[] | undefined {
	if (value.trim().length === 0) return [];
	try {
		const parsed: unknown = JSON.parse(value);
		return parsed !== null && typeof parsed === "object"
			? (parsed as Record<string, unknown> | readonly unknown[])
			: undefined;
	} catch {
		return undefined;
	}
}
function parseObject(value: string): Record<string, unknown> | undefined {
	const parsed = parseJson(value);
	return parsed !== undefined && !Array.isArray(parsed) ? (parsed as Record<string, unknown>) : undefined;
}
function parseJsonLines(value: string): Record<string, unknown> | readonly unknown[] | undefined {
	const lines = value
		.split("\n")
		.map((line) => line.trim())
		.filter((line) => line.length > 0);
	if (lines.length === 0) return [];
	const rows: unknown[] = [];
	for (const line of lines) {
		try {
			const parsed: unknown = JSON.parse(line);
			rows.push(parsed);
		} catch {
			return parseJson(value);
		}
	}
	return rows;
}
function parseSummaryCount(value: string, verb: string): number | undefined {
	const match = new RegExp(`(?:${verb})\\s+(\\d+)`, "i").exec(value);
	return match?.[1] === undefined ? undefined : Number.parseInt(match[1], 10);
}
function stringValue(value: unknown): string | undefined {
	return typeof value === "string" && value.length > 0 ? value : undefined;
}
function numberValue(value: unknown): number | undefined {
	return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
function ok<T>(data: T, result: ProcessResult): { ok: true; data: T; stdout: string; stderr: string; code: number } {
	return { ok: true, data, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
}
function failure(result: ProcessResult, reason?: ClawGalleryFailure["reason"]): ClawGalleryFailure {
	return {
		ok: false,
		reason: reason ?? result.reason ?? "nonzero-exit",
		stdout: result.stdout,
		stderr: result.stderr.includes("/")
			? "clawgallery command failed; path details suppressed"
			: result.stderr.trim().slice(0, 500),
		code: result.code,
	};
}
