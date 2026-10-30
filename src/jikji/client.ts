import type { ChildProcess } from "node:child_process";
import { spawn } from "node:child_process";
import { existsSync, realpathSync } from "node:fs";
import { portableSpawnCommand } from "../process/portable-spawn.ts";
import { TREE_KILL_GRACE_MS, terminateProcessTree } from "../process/terminate-tree.ts";
import { parseJikjiAnswerPack } from "./answer-pack.ts";
import { cachedJikjiBinaryPath, ensureJikjiBinary, lookupExecutableInPath } from "./installer.ts";
import { countFiles, scaledPrepareTimeoutMs } from "./prepare-budget.ts";
import { SerialQueue } from "./serial-queue.ts";
import type {
	JikjiFailureReason,
	JikjiFindOptions,
	JikjiFindResult,
	JikjiOptions,
	JikjiPrepareOptions,
	JikjiPrepareResult,
} from "./types.ts";
import { isSupportedJikjiVersion, JIKJI_INSTALL_VERSION, JIKJI_MIN_VERSION, parseJikjiVersion } from "./version.ts";

const DEFAULT_BINARY = "jikji";
const DEFAULT_TIMEOUT_MS = 60_000;
/** `jikji prepare --parse-timeout` default (seconds): the per-file limit jikji enforces itself. */
const DEFAULT_PARSE_TIMEOUT_SECONDS = 5;
/** `jikji --version` answers instantly; a binary that does not within this is treated as unprobeable. */
const VERSION_PROBE_TIMEOUT_MS = 5_000;
const DEFAULT_MAX_BUFFER_BYTES = 1_048_576;

/** Result of a call whose signal aborted while it was still queued: nothing was spawned. */
const ABORTED_BEFORE_START: ProcessResult = { ok: false, reason: "aborted", stdout: "", stderr: "", code: null };

type ProcessResult = {
	readonly ok: boolean;
	readonly reason?: JikjiFailureReason;
	readonly stdout: string;
	readonly stderr: string;
	readonly code: number | null;
	readonly detectedVersion?: string;
};

/** A binary the client resolved: where it is and whether its `--version` clears the floor. */
type Resolved =
	| { readonly ok: true; readonly command: string }
	| { readonly ok: false; readonly detectedVersion: string; readonly message: string };

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
	readonly cwd?: string;
	/** Wall for this process; falls back to `options.timeoutMs`, then 60 s. */
	readonly timeoutMs?: number;
};

export class JikjiClient {
	private readonly options: JikjiOptions;
	private resolved: Resolved | undefined;
	private installInFlight: Promise<Resolved> | undefined;
	// Upstream jikji writes one central SQLite file (WAL + 30 s busy_timeout in
	// 0.2.0+, though long prepares can still hold the writer lock past 30 s).
	// Calls are serialized per kind: prepares queue behind each other, finds
	// queue behind each other, and a query never waits behind a minutes-long refresh.
	private readonly prepareQueue = new SerialQueue();
	private readonly findQueue = new SerialQueue();

	constructor(options: JikjiOptions = {}) {
		this.options = options;
	}

	/**
	 * Resolve the jikji command, refusing any binary below {@link JIKJI_MIN_VERSION}:
	 * 1. explicit binaryPath: used as given, never probed (the operator chose it).
	 * 2. PATH lookup for `jikji`, then cached `<root>/.autorag/bin/jikji`. The first one
	 *    whose `--version` clears the floor wins. A binary whose `--version` output is
	 *    not recognised is used as-is, so a custom build is never locked out.
	 * 3. Every candidate was below the floor (or none exists): `prepare` installs the
	 *    pinned {@link JIKJI_INSTALL_VERSION} into the cache with `cargo install --force`.
	 *    `find` (`install: false`) never installs; it reports the stale version instead.
	 * 4. Nothing resolvable and no root or autoInstall off: bare `jikji` (spawn-error
	 *    degrade preserves the previous behavior).
	 * A usable result is cached per client. A query-time miss is not cached, so a later
	 * refresh can still install.
	 */
	private async resolveCommand(options: { readonly install?: boolean } = {}): Promise<Resolved> {
		if (this.options.binaryPath !== undefined && this.options.binaryPath !== DEFAULT_BINARY) {
			return { ok: true, command: commandFor(this.options.binaryPath) };
		}
		if (this.resolved !== undefined) return this.resolved;
		const env = controlledEnv(this.options.env);
		const candidates: string[] = [];
		const pathCommand = lookupExecutableInPath(DEFAULT_BINARY, env);
		if (pathCommand !== undefined) candidates.push(pathCommand);
		const root = this.options.root;
		if (root !== undefined) {
			const cached = cachedJikjiBinaryPath(root);
			if (existsSync(cached) && cached !== pathCommand) candidates.push(cached);
		}
		let stale: Extract<Resolved, { ok: false }> | undefined;
		for (const candidate of candidates) {
			const verdict = await this.probe(candidate, env);
			if (verdict.ok) {
				this.resolved = verdict;
				return verdict;
			}
			stale ??= verdict;
		}
		if (root !== undefined && options.install !== false && this.options.autoInstall !== false) {
			// Share one cargo install across concurrent prepares — one client
			// serves multiple roots — instead of racing N compiles.
			if (this.installInFlight !== undefined) return this.installInFlight;
			const forced = candidates.includes(cachedJikjiBinaryPath(root));
			const install = ensureJikjiBinary({ root, env, version: JIKJI_INSTALL_VERSION, force: forced })
				.then((installed): Resolved => {
					if (installed.ok) return { ok: true, command: installed.binaryPath };
					return stale === undefined
						? { ok: true, command: DEFAULT_BINARY }
						: { ...stale, message: `${stale.message} ${installed.message}` };
				})
				.then((resolved) => {
					if (resolved.ok || stale !== undefined) this.resolved = resolved.ok ? resolved : undefined;
					return resolved;
				})
				.finally(() => {
					this.installInFlight = undefined;
				});
			this.installInFlight = install;
			return install;
		}
		if (stale !== undefined) return stale;
		// A query (`find`) with nothing resolvable: leave `resolved` unset so a later refresh can install.
		if (root !== undefined && options.install === false) return { ok: true, command: DEFAULT_BINARY };
		this.resolved = { ok: true, command: DEFAULT_BINARY };
		return this.resolved;
	}

	/** Run `<candidate> --version` and decide whether it clears the floor. */
	private async probe(candidate: string, env: NodeJS.ProcessEnv): Promise<Resolved> {
		const probed = await spawnJikji({
			command: candidate,
			args: ["--version"],
			options: { env, timeoutMs: VERSION_PROBE_TIMEOUT_MS, maxBufferBytes: 4_096 },
			timeoutMs: VERSION_PROBE_TIMEOUT_MS,
		});
		const version = probed.ok ? parseJikjiVersion(probed.stdout) : undefined;
		if (version === undefined || isSupportedJikjiVersion(version)) return { ok: true, command: candidate };
		return {
			ok: false,
			detectedVersion: version,
			message: `jikji ${version} at ${candidate} is older than the ${JIKJI_MIN_VERSION} AutoRAG requires.`,
		};
	}

	async prepare(root: string, options: JikjiPrepareOptions = {}): Promise<JikjiPrepareResult> {
		const canonicalRoot = canonicalJikjiRoot(root);
		const clientOptions: JikjiOptions =
			options.parseTimeout === undefined ? this.options : { ...this.options, parseTimeout: options.parseTimeout };
		const result = await this.prepareQueue.run(
			async () => {
				const resolved = await this.resolveCommand();
				if (!resolved.ok) return unsupportedVersion(resolved);
				return this.spawn({
					command: resolved.command,
					args: buildPrepareArgs(clientOptions, canonicalRoot),
					signal: options.signal,
					// Sized when the root's turn starts, so queue wait never eats the budget.
					timeoutMs:
						options.timeoutMs ?? (await this.prepareTimeoutMs(canonicalRoot, clientOptions, options.signal)),
				});
			},
			{ signal: options.signal, value: ABORTED_BEFORE_START },
		);
		if (!result.ok) {
			return {
				ok: false,
				reason: result.reason ?? "nonzero-exit",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code,
				...(result.detectedVersion === undefined ? {} : { detectedVersion: result.detectedVersion }),
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
		const canonicalRoot = canonicalJikjiRoot(root);
		const result = await this.findQueue.run(
			async () => {
				const resolved = await this.resolveCommand({ install: false });
				if (!resolved.ok) return unsupportedVersion(resolved);
				return this.spawn({
					command: resolved.command,
					args: buildFindArgs(canonicalRoot, query, options, this.options),
					signal: options.signal,
				});
			},
			{ signal: options.signal, value: ABORTED_BEFORE_START },
		);
		if (!result.ok) {
			return {
				ok: false,
				reason: result.reason ?? "nonzero-exit",
				stdout: result.stdout,
				stderr: result.stderr,
				code: result.code,
				...(result.detectedVersion === undefined ? {} : { detectedVersion: result.detectedVersion }),
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

	/**
	 * Hang guard for one root, never a fixed total wall: `timeoutMs` (base) plus a
	 * per-file slice for every file under the root, so a very large root is not
	 * cut by its size. The slice defaults to the per-file parse limit
	 * (`parseTimeout`, seconds) that jikji itself enforces by skipping the file,
	 * so only a run that outlives even an all-files-hit-the-limit budget is
	 * killed (reported as a degraded root). `prepareTimeoutMsPerFile` overrides
	 * the slice; `0` restores a flat `timeoutMs` wall. A per-call `timeoutMs`
	 * bypasses all of this.
	 */
	private async prepareTimeoutMs(
		root: string,
		options: JikjiOptions,
		signal: AbortSignal | undefined,
	): Promise<number | undefined> {
		const perFile = options.prepareTimeoutMsPerFile ?? (options.parseTimeout ?? DEFAULT_PARSE_TIMEOUT_SECONDS) * 1000;
		if (perFile <= 0) return undefined;
		const files = await countFiles(root, options.includeHidden === true, signal);
		return scaledPrepareTimeoutMs(options.timeoutMs ?? DEFAULT_TIMEOUT_MS, perFile, files);
	}

	private async spawn(request: Omit<SpawnJikjiRequest, "options">): Promise<ProcessResult> {
		return spawnJikji({
			...request,
			options: this.options,
		});
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
		const portable = portableSpawnCommand(request.command, request.args);
		const child = spawn(portable.command, [...portable.args], {
			env: controlledEnv(options.env),
			...(request.cwd === undefined ? {} : { cwd: request.cwd }),
			// POSIX: lead our own process group so terminateProcessTree can reap
			// descendants that inherited the pipes (#1578 hang class). Windows keeps
			// the parent/child chain that `taskkill /t` walks.
			detached: process.platform !== "win32",
			windowsHide: true,
			stdio: ["ignore", "pipe", "pipe"],
			// A signal-bearing run leads its own process group on POSIX so an abort reaches descendants too.
			...(request.signal !== undefined && process.platform !== "win32" ? { detached: true } : {}),
		});
		let stdout: BufferState = { text: "", bytes: 0, capped: false };
		let stderr: BufferState = { text: "", bytes: 0, capped: false };
		let settled = false;
		let finalReason: JikjiFailureReason | undefined;
		let pipeGuard: ReturnType<typeof setTimeout> | undefined;
		const settleWith = (result: ProcessResult): void => {
			if (settled) return;
			settled = true;
			clearTimeout(timeout);
			clearTimeout(pipeGuard);
			request.signal?.removeEventListener("abort", abortHandler);
			resolve(result);
		};
		const finish = (code: number | null): void => {
			if (finalReason !== undefined) {
				settleWith({ ok: false, reason: finalReason, stdout: stdout.text, stderr: stderr.text, code });
				return;
			}
			settleWith({ ok: code === 0, stdout: stdout.text, stderr: stderr.text, code });
		};
		const timeout = setTimeout(
			() => {
				finalReason = "timeout";
				terminate(child);
			},
			request.timeoutMs ?? options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
		);
		const abortHandler = (): void => {
			finalReason = "aborted";
			terminateProcessTree(child);
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
		child.on("exit", () => {
			if (settled) return;
			// The direct child can exit while a descendant still holds the pipes it
			// inherited, which keeps "close" from firing. Reap the group, then bound
			// the wait for the stream release.
			terminate(child);
			pipeGuard = setTimeout(() => finish(child.exitCode), TREE_KILL_GRACE_MS);
		});
		child.on("error", (error) => {
			settleWith({ ok: false, reason: "spawn-error", stdout: stdout.text, stderr: error.message, code: null });
		});
		child.on("close", (code) => finish(code));
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

function buildFindArgs(root: string, query: string, options: JikjiFindOptions, scope: JikjiOptions): readonly string[] {
	// `jikji find` otherwise spawns a detached `jikji prepare <root>` when the
	// index is changed or stale. That would index during a query and, because
	// the detached child omits `--no-agent-rules`, rewrite AGENTS.md /
	// CLAUDE.md / .cursorrules in the user's source root.
	// Not covered: upstream's synchronous reindex when a query returns no paths
	// has no CLI opt-out (0.2.x). Observed with 0.2.1: it does not rewrite
	// AGENTS.md / CLAUDE.md / .cursorrules, but it runs `prepare` with jikji's
	// built-in scope. The scope below is therefore forwarded so that reindex
	// indexes exactly what `refresh` did, not what `exclude` / `includeHidden` /
	// `includeSensitive` were set to keep out.
	const args = ["find", root, query, "--json", "--no-background-refresh"];
	if (options.topK !== undefined) args.push("--top-k", String(options.topK));
	if (options.first === true) args.push("--first");
	if (options.fresh === true) args.push("--fresh");
	if (options.autoPrepare === true) args.push("--auto-prepare");
	if (options.staleAfterSeconds !== undefined) {
		args.push("--stale-after-seconds", String(options.staleAfterSeconds));
	}
	if (scope.includeHidden === true) args.push("--include-hidden");
	if (scope.includeSensitive === true) args.push("--include-sensitive");
	if (scope.maxFiles !== undefined && scope.maxFiles > 0) args.push("--max-files", String(scope.maxFiles));
	for (const pattern of scope.exclude ?? []) args.push("--exclude", pattern);
	if (scope.maxHashBytes !== undefined) args.push("--max-hash-bytes", String(scope.maxHashBytes));
	if (scope.parseTimeout !== undefined) args.push("--parse-timeout", String(scope.parseTimeout));
	return args;
}

/** A resolved binary below the floor: nothing was spawned, and the message says which binary and why. */
function unsupportedVersion(resolved: Extract<Resolved, { ok: false }>): ProcessResult {
	return {
		ok: false,
		reason: "unsupported-version",
		stdout: "",
		stderr: resolved.message,
		code: null,
		detectedVersion: resolved.detectedVersion,
	};
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
	terminateProcessTree(child);
}

function appendBounded(state: BufferState, chunk: string, maxBytes: number): BufferState {
	const chunkBytes = Buffer.byteLength(chunk);
	const nextBytes = state.bytes + chunkBytes;
	if (nextBytes <= maxBytes) return { text: state.text + chunk, bytes: nextBytes, capped: false };
	const remainingBytes = Math.max(maxBytes - state.bytes, 0);
	return { text: state.text + chunk.slice(0, remainingBytes), bytes: maxBytes, capped: true };
}

/**
 * Jikji canonicalizes roots in `prepare` when recording them in SQLite, while
 * `find` queries by exact path string. Canonicalize existing paths so symlinked
 * roots (for example macOS `/var/folders/...`) match across both commands.
 */
function canonicalJikjiRoot(root: string): string {
	try {
		return realpathSync(root);
	} catch {
		return root;
	}
}
