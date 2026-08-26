import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { ManagedCliConfigManager, ManagedCliRegistry } from "../cli/managed-cli-config.ts";
import { createCrawlerManagedCliProvider } from "./crawler-managed-config.ts";
import type {
	CrawlerCliOptions,
	CrawlerFailure,
	CrawlerProfile,
	CrawlerSearchOptions,
	CrawlerSearchResult,
	CrawlerSyncResult,
} from "./crawler-types.ts";

const DEFAULT_TIMEOUT_MS = 60_000;
const DEFAULT_MAX_BUFFER_BYTES = 1_048_576;
const SAFE_ENV_KEYS = new Set(["HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "TMP", "TEMP", "NO_COLOR"]);

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: CrawlerFailure["reason"];
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

export class CrawlerCliClient {
	private readonly profile: CrawlerProfile;
	private readonly options: CrawlerCliOptions;
	private readonly managedCliConfigManager: ManagedCliConfigManager | undefined;

	constructor(profile: CrawlerProfile, options: CrawlerCliOptions = {}) {
		this.profile = profile;
		this.options = options;
		if (options.managedCliConfigManager) {
			this.managedCliConfigManager = options.managedCliConfigManager;
		} else if (options.workspacePath !== undefined) {
			const registry = new ManagedCliRegistry();
			registry.register(createCrawlerManagedCliProvider(profile.binaryName, options.binaryPath));
			this.managedCliConfigManager = new ManagedCliConfigManager({ workspace: options.workspacePath, registry });
		}
	}

	async sync(signal?: AbortSignal): Promise<CrawlerSyncResult> {
		const result = await this.run(this.profile.syncArgs(this.options), signal);
		if (!result.ok) return toFailure(this.profile.binaryName, result);
		const count = this.profile.parseSyncCount(result.stdout);
		if (count === undefined) return toFailure(this.profile.binaryName, result, "invalid-output");
		return { ok: true, count, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
	}

	async search(query: string, options: CrawlerSearchOptions = {}): Promise<CrawlerSearchResult> {
		const topK = options.topK ?? 20;
		const result = await this.run(this.profile.searchArgs(this.options, query, topK), options.signal);
		if (!result.ok) return toFailure(this.profile.binaryName, result);
		const hits = this.profile.parseHits(result.stdout);
		if (hits === undefined) return toFailure(this.profile.binaryName, result, "invalid-output");
		return { ok: true, hits, stdout: result.stdout, stderr: result.stderr, code: result.code ?? 0 };
	}

	private async run(args: readonly string[], signal?: AbortSignal): Promise<ProcessResult> {
		const env = controlledEnv(this.profile.allowedEnvPrefixes, this.options.env);
		env.CRAWLKIT_NO_UPDATE_CHECK = "1";
		let launch:
			| { readonly prefixArgs: readonly string[]; readonly cwd?: string; readonly env: Readonly<Record<string, string>> }
			| undefined;
		try {
			if (this.managedCliConfigManager) {
				launch = await this.managedCliConfigManager.materialize(this.profile.binaryName, {
					config: {
						...(this.options.databasePath === undefined ? {} : { databasePath: this.options.databasePath }),
						...(this.options.sourcePath === undefined ? {} : { sourcePath: this.options.sourcePath }),
					},
				});
			}
		} catch {
			return { ok: false, reason: "spawn-error", stdout: "", stderr: "", code: null };
		}
		return spawnCrawler(
			this.options.binaryPath ?? this.profile.binaryName,
			[...(launch?.prefixArgs ?? []), ...stripTransportArgs(args, launch !== undefined)],
			{ ...env, ...(launch?.env ?? {}) },
			signal,
			this.options,
			launch?.cwd,
		);
	}
}

function stripTransportArgs(args: readonly string[], managed: boolean): readonly string[] {
	if (!managed) return args;
	const result: string[] = [];
	for (let index = 0; index < args.length; index += 1) {
		if (args[index] === "--db" || args[index] === "--source") {
			index += 1;
			continue;
		}
		result.push(args[index] as string);
	}
	return result;
}

function spawnCrawler(
	command: string,
	args: readonly string[],
	env: NodeJS.ProcessEnv,
	signal: AbortSignal | undefined,
	options: CrawlerCliOptions,
	cwd?: string,
): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const portable = portableSpawnCommand(command, args);
		const child = spawn(portable.command, [...portable.args], {
			env,
			...(cwd === undefined ? {} : { cwd }),
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: CrawlerFailure["reason"] | undefined;
		const maxBuffer = options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
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
			resolve({
				ok: false,
				reason: error.code === "ENOENT" ? "binary-missing" : "spawn-error",
				stdout: stdout.text,
				stderr: "",
				code: null,
			});
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			signal?.removeEventListener("abort", abortHandler);
			resolve({
				ok: code === 0 && finalReason === undefined,
				...(finalReason !== undefined
					? { reason: finalReason }
					: code === 0
						? {}
						: { reason: "nonzero-exit" as const }),
				stdout: stdout.text,
				stderr: stderr.text,
				code,
			});
		});
	});
}

function controlledEnv(
	allowedPrefixes: readonly string[],
	configured: Readonly<Record<string, string | undefined>> | undefined,
): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = {};
	for (const [key, value] of Object.entries(process.env)) {
		if (value !== undefined && isAllowedEnvKey(key, allowedPrefixes)) env[key] = value;
	}
	for (const [key, value] of Object.entries(configured ?? {})) {
		if (value === undefined) delete env[key];
		else if (isAllowedEnvKey(key, allowedPrefixes)) env[key] = value;
	}
	return env;
}

function isAllowedEnvKey(key: string, allowedPrefixes: readonly string[]): boolean {
	return (
		SAFE_ENV_KEYS.has(key) || key.startsWith("CRAWLKIT_") || allowedPrefixes.some((prefix) => key.startsWith(prefix))
	);
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const nextBytes = state.bytes + Buffer.byteLength(chunk);
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remaining = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remaining), bytes: maxBytes, capped: true };
}

function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}

function toFailure(binaryName: string, result: ProcessResult, reason?: CrawlerFailure["reason"]): CrawlerFailure {
	const failed = result.stderr.length > 0 || result.stdout.length > 0;
	return {
		ok: false,
		reason: reason ?? result.reason ?? "nonzero-exit",
		stdout: "",
		stderr: failed ? `${binaryName} command failed; details suppressed for datasource privacy` : "",
		code: result.code,
	};
}
