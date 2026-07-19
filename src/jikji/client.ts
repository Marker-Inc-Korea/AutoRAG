import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { parseJikjiAnswerPack } from "./answer-pack.ts";
import { cachedJikjiBinaryPath, ensureJikjiBinary, lookupExecutableInPath } from "./installer.ts";
import type {
	JikjiFailureReason,
	JikjiFindOptions,
	JikjiFindResult,
	JikjiOptions,
	JikjiPrepareOptions,
	JikjiPrepareResult,
} from "./types.ts";

const DEFAULT_BINARY = "jikji";
const DEFAULT_TIMEOUT_MS = 10_000;
const DEFAULT_MAX_BUFFER_BYTES = 1_048_576;

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: JikjiFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
};

type BufferState = {
	readonly text: string;
	readonly bytes: number;
	readonly capped: boolean;
};

type SpawnJikjiRequest = {
	readonly command: string;
	readonly options: JikjiOptions;
	readonly args: readonly string[];
	readonly signal?: AbortSignal;
};

export class JikjiClient {
	private readonly options: JikjiOptions;
	private resolvedCommand: string | undefined;

	constructor(options: JikjiOptions = {}) {
		this.options = options;
	}

	/**
	 * Resolve the jikji command following the priority chain:
	 * 1. explicit binaryPath
	 * 2. PATH lookup for `jikji`
	 * 3. cached `<root>/.autorag/bin/jikji`
	 * 4. autoInstall (default true) via `cargo install jikji-cli` into the cache
	 * 5. bare `jikji` (spawn-error degrade preserves the previous behavior)
	 * The result is cached per client so a failed install is not retried.
	 */
	private async resolveCommand(): Promise<string> {
		if (this.options.binaryPath !== undefined) return commandFor(this.options.binaryPath);
		if (this.resolvedCommand !== undefined) return this.resolvedCommand;
		if (lookupExecutableInPath(DEFAULT_BINARY, process.env) !== undefined) {
			this.resolvedCommand = DEFAULT_BINARY;
			return this.resolvedCommand;
		}
		if (this.options.root !== undefined) {
			const cached = cachedJikjiBinaryPath(this.options.root);
			if (existsSync(cached)) {
				this.resolvedCommand = cached;
				return this.resolvedCommand;
			}
			if (this.options.autoInstall !== false) {
				const installed = await ensureJikjiBinary({ root: this.options.root });
				if (installed.ok) {
					this.resolvedCommand = installed.binaryPath;
					return this.resolvedCommand;
				}
			}
		}
		this.resolvedCommand = DEFAULT_BINARY;
		return this.resolvedCommand;
	}

	async prepare(root: string, options: JikjiPrepareOptions = {}): Promise<JikjiPrepareResult> {
		const result = await spawnJikji({
			command: await this.resolveCommand(),
			options: this.options,
			args: buildPrepareArgs(this.options, root),
			signal: options.signal,
		});
		if (!result.ok) {
			return {
				ok: false,
				reason: result.reason ?? "nonzero-exit",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code,
			};
		}
		return {
			ok: true,
			stdout: result.stdout,
			stderr: result.stderr,
			code: result.code ?? 0,
		};
	}

	async find(root: string, query: string, options: JikjiFindOptions = {}): Promise<JikjiFindResult> {
		const result = await spawnJikji({
			command: await this.resolveCommand(),
			options: this.options,
			args: buildFindArgs(root, query, options),
			signal: options.signal,
		});
		if (!result.ok) {
			return {
				ok: false,
				reason: result.reason ?? "nonzero-exit",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code,
			};
		}
		const answerPack = parseJikjiAnswerPack(result.stdout);
		if (answerPack === undefined) {
			return {
				ok: false,
				reason: "bad-answer-pack",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code ?? 0,
			};
		}
		return {
			ok: true,
			answerPack,
			stdout: result.stdout,
			stderr: result.stderr,
			code: result.code ?? 0,
		};
	}
}

/**
 * Shared spawn/buffer/abort/timeout helper used by both `prepare` and `find`.
 * Resolves a {@link ProcessResult} capturing stdout/stderr, exit code, and a
 * failure reason when the run is degraded (timeout, abort, buffer overflow,
 * spawn error, nonzero exit).
 */
function spawnJikji(request: SpawnJikjiRequest): Promise<ProcessResult> {
	return new Promise((resolve) => {
		const options = request.options;
		const child = spawn(request.command, request.args, {
			env: controlledEnv(options.env),
			stdio: ["ignore", "pipe", "pipe"],
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: JikjiFailureReason | undefined;
		const timeout = setTimeout(() => {
			finalReason = "timeout";
			terminate(child);
		}, options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminate(child);
		};
		if (request.signal?.aborted) abortHandler();
		request.signal?.addEventListener("abort", abortHandler, { once: true });
		child.stdout.setEncoding("utf8");
		child.stderr.setEncoding("utf8");
		child.stdout.on("data", (chunk: string) => {
			stdout = appendBounded(stdout, chunk, options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES);
			if (stdout.capped) {
				finalReason = "stdout-too-large";
				terminate(child);
			}
		});
		child.stderr.on("data", (chunk: string) => {
			stderr = appendBounded(stderr, chunk, options.maxBufferBytes ?? DEFAULT_MAX_BUFFER_BYTES);
			if (stderr.capped) {
				finalReason = "stderr-too-large";
				terminate(child);
			}
		});
		child.on("error", (error) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.signal?.removeEventListener("abort", abortHandler);
			resolve({ ok: false, reason: "spawn-error", stdout: stdout.text, stderr: error.message, code: null });
		});
		child.on("close", (code) => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			request.signal?.removeEventListener("abort", abortHandler);
			if (finalReason !== undefined) {
				resolve({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			resolve({ ok: code === 0, stdout: stdout.text, stderr: stderr.text, code });
		});
	});
}

function commandFor(binaryPath: string | undefined): string {
	return binaryPath === undefined || binaryPath === DEFAULT_BINARY ? DEFAULT_BINARY : binaryPath;
}

function buildPrepareArgs(options: JikjiOptions, root: string): readonly string[] {
	const args = ["prepare", root, "--json"];
	if (options.includeHidden === true) args.push("--include-hidden");
	if (options.includeSensitive === true) args.push("--include-sensitive");
	// SAFE DEFAULT: never rewrite AGENTS.md/CLAUDE.md/.cursorrules.
	// Emit --no-agent-rules unless the caller explicitly opts in via writeAgentRules.
	// (noAgentRules is kept for back-compat but no longer controls the flag.)
	if (options.writeAgentRules !== true) args.push("--no-agent-rules");
	if (options.enableMediaIndex === true) args.push("--enable-media-index");
	if (options.parseTimeout !== undefined) args.push("--parse-timeout", String(options.parseTimeout));
	if (options.maxHashBytes !== undefined) args.push("--max-hash-bytes", String(options.maxHashBytes));
	if (options.docTextMaxChars !== undefined) args.push("--doc-text-max-chars", String(options.docTextMaxChars));
	if (options.docTextChunkChars !== undefined) args.push("--doc-text-chunk-chars", String(options.docTextChunkChars));
	if (options.maxFiles !== undefined && options.maxFiles > 0) args.push("--max-files", String(options.maxFiles));
	if (options.enableMediaIndex === true && options.mediaIndexMaxMb !== undefined) {
		args.push("--media-index-max-mb", String(options.mediaIndexMaxMb));
	}
	for (const pattern of options.exclude ?? []) {
		args.push("--exclude", pattern);
	}
	return args;
}

function buildFindArgs(root: string, query: string, options: JikjiFindOptions): readonly string[] {
	const args = ["find", root, query, "--json"];
	if (options.topK !== undefined) args.push("--top-k", String(options.topK));
	if (options.first === true) args.push("--first");
	if (options.fresh === true) args.push("--fresh");
	if (options.autoPrepare === true) args.push("--auto-prepare");
	if (options.staleAfterSeconds !== undefined) {
		args.push("--stale-after-seconds", String(options.staleAfterSeconds));
	}
	return args;
}

function controlledEnv(configuredEnv: Readonly<Record<string, string | undefined>> | undefined): NodeJS.ProcessEnv {
	const env: NodeJS.ProcessEnv = { ...process.env };
	for (const [key, value] of Object.entries(configuredEnv ?? {})) {
		if (value !== undefined) env[key] = value;
		else delete env[key];
	}
	return env;
}

function terminate(child: ChildProcess): void {
	if (!child.killed) child.kill("SIGTERM");
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk);
	const nextBytes = state.bytes + chunkBytes;
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remainingBytes = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remainingBytes), bytes: maxBytes, capped: true };
}
