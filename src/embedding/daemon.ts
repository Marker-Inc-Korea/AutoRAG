import { spawn } from "node:child_process";
import { accessSync, constants, readdirSync, readFileSync } from "node:fs";
import { basename, delimiter, join } from "node:path";
import { resolveUserHome } from "../config/home.ts";

/** The dense model an `inl serve` instance serves by default; callers pass another to look for that one. */
export const DEFAULT_DAEMON_MODEL = "Qwen/Qwen3-Embedding-0.6B";

/** The engine kinds an `inl serve` instance can speak, as reported by `GET /health`. */
export type DaemonKind = "dense" | "sparse" | "multimodal" | "late";

/** The model each kind serves when the caller names none (inl `engines.rs`). */
const DEFAULT_KIND_MODELS: Readonly<Record<DaemonKind, string>> = {
	dense: DEFAULT_DAEMON_MODEL,
	sparse: "nlpai-lab/KURE-Sparse-nano",
	multimodal: "google/embeddinggemma-2",
	late: "perplexity-ai/pplx-embed-v2-late-0.6b",
};

/**
 * Which executable starts the daemon. `inl` (im-not-llm) owns the weights and the state root; `minsync`
 * stays a compatible launcher for dense/sparse because it links the same `inl-server` and writes the same
 * discovery files.
 */
export type DaemonLauncher = "inl" | "minsync";

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);
const HEALTH_TIMEOUT_MS = 2_000;
const DEFAULT_READY_TIMEOUT_MS = 10 * 60 * 1_000;
const POLL_INTERVAL_MS = 150;
const STDERR_LIMIT = 4_000;
const STOP_TIMEOUT_MS = 15_000;

export interface InlDaemon {
	/** `http://127.0.0.1:<port>`; the OpenAI route lives under `/v1`, TEI at `/embed`. */
	readonly baseUrl: string;
	/** Engine kind the server reports (`dense`, `sparse`, `multimodal`, `late`). */
	readonly kind: DaemonKind;
	readonly model: string;
	readonly dimension: number;
	readonly instance: string;
	/** True when {@link ensureInlDaemon} spawned it. */
	readonly started: boolean;
	/** Which launcher spawned it; set only when {@link ensureInlDaemon} started it. */
	readonly launcher?: DaemonLauncher;
	readonly pid?: number;
}

export class InlDaemonError extends Error {
	constructor(message: string, options?: ErrorOptions) {
		super(message, options);
		this.name = "InlDaemonError";
	}
}

export interface ProbeOptions {
	/** User home; im-not-llm keeps its server state under `<home>/.cache/im-not-llm/serve`. */
	readonly home?: string;
	/** Engine kind to accept. Defaults to `"dense"`. */
	readonly kind?: DaemonKind;
	/** Model to accept; defaults to the kind's own default model. */
	readonly model?: string;
	readonly fetchImpl?: typeof fetch;
}

/**
 * `inl serve` publishes `host`/`port` (and `pid`) per model instance under
 * `~/.cache/im-not-llm/serve/<instance>/` (`%LOCALAPPDATA%\im-not-llm\serve` on Windows) and removes them on
 * a clean stop. A leftover file from a crashed server is harmless because every candidate must answer
 * `GET /health` as a loopback `status: ok` server of the wanted kind and model before it is used.
 */
export function inlServeStateRoot(home: string = resolveUserHome()): string {
	if (process.platform === "win32") {
		const local = process.env.LOCALAPPDATA ?? join(home, "AppData", "Local");
		return join(local, "im-not-llm", "serve");
	}
	return join(home, ".cache", "im-not-llm", "serve");
}

interface Health {
	readonly status?: string;
	readonly kind?: string;
	readonly model_id?: string;
	readonly dimension?: number;
	readonly instance?: string;
	readonly pid?: number;
}

async function probeStateDir(
	root: string,
	instance: string,
	wantedKind: DaemonKind,
	wantedModel: string,
	fetchImpl: typeof fetch,
): Promise<InlDaemon | undefined> {
	const dir = join(root, instance);
	const port = Number(readText(join(dir, "port")));
	const host = readText(join(dir, "host")) ?? "127.0.0.1";
	if (!Number.isInteger(port) || port <= 0 || !LOOPBACK_HOSTS.has(host)) return undefined;
	const baseUrl = `http://${host.includes(":") ? `[${host}]` : host}:${port}`;
	try {
		const response = await fetchImpl(`${baseUrl}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
		if (!response.ok) return undefined;
		const health = (await response.json()) as Health;
		if (health.status !== "ok") return undefined;
		// A missing kind means dense (older servers); a missing model_id can never match.
		if ((health.kind ?? "dense") !== wantedKind) return undefined;
		if (typeof health.model_id !== "string" || health.model_id.toLowerCase() !== wantedModel.toLowerCase())
			return undefined;
		if (typeof health.dimension !== "number") return undefined;
		return {
			baseUrl,
			kind: wantedKind,
			model: health.model_id,
			dimension: health.dimension,
			instance: health.instance ?? instance,
			started: false,
			...(typeof health.pid === "number" ? { pid: health.pid } : {}),
		};
	} catch {
		// Stale discovery files: nothing is listening there any more.
		return undefined;
	}
}

/**
 * The running `inl serve` server of the wanted kind and model, discovered by reading `port`/`host` under the
 * shared serve root and confirming the answer with `GET /health`. Matching is on the server's own `kind` and
 * `model_id` (case-insensitive), never on a directory name: instance keys encode device/dtype/attention, which
 * a consumer does not know.
 */
export async function probeInlDaemon(options: ProbeOptions = {}): Promise<InlDaemon | undefined> {
	const root = inlServeStateRoot(options.home);
	const wantedKind = options.kind ?? "dense";
	const wantedModel = options.model ?? DEFAULT_KIND_MODELS[wantedKind];
	const fetchImpl = options.fetchImpl ?? fetch;
	let instances: string[];
	try {
		instances = readdirSync(root);
	} catch {
		return undefined;
	}
	for (const instance of instances) {
		const found = await probeStateDir(root, instance, wantedKind, wantedModel, fetchImpl);
		if (found !== undefined) return found;
	}
	return undefined;
}

function readText(path: string): string | undefined {
	try {
		return readFileSync(path, "utf8").trim();
	} catch {
		return undefined;
	}
}

export interface EnsureDaemonOptions extends ProbeOptions {
	readonly env?: NodeJS.ProcessEnv;
	/** Working directory of the spawned server. */
	readonly cwd?: string;
	/** How long to wait for the server to publish and answer; the first start may download the model. */
	readonly readyTimeoutMs?: number;
	/** Idle seconds before the daemon exits and frees the model (`--idle-timeout`). */
	readonly idleTimeoutSeconds?: number;
	/**
	 * A launcher to use as given instead of resolving one from PATH. `kind` chooses the argument shape; when
	 * omitted it is inferred from the executable's base name. Used by embedded callers and tests.
	 */
	readonly launcher?: { readonly path: string; readonly kind?: DaemonLauncher };
	/**
	 * Resolves the `minsync` fallback launcher (it may install one), called only when `inl` is not on PATH.
	 * When absent, `minsync` is looked up on PATH.
	 */
	readonly resolveMinSyncBinary?: () => Promise<string>;
}

/** The first executable named `name` on the environment's PATH, if any. */
function lookupExecutableInPath(env: NodeJS.ProcessEnv, name: string): string | undefined {
	const pathEnv = env.PATH;
	if (typeof pathEnv !== "string" || pathEnv.length === 0) return undefined;
	const execName = process.platform === "win32" ? `${name}.exe` : name;
	for (const dir of pathEnv.split(delimiter)) {
		if (dir.length === 0) continue;
		const candidate = join(dir, execName);
		try {
			accessSync(candidate, constants.X_OK);
			return candidate;
		} catch {
			// not executable in this dir
		}
	}
	return undefined;
}

function inferLauncherKind(path: string): DaemonLauncher {
	const name = basename(path)
		.replace(/\.exe$/i, "")
		.toLowerCase();
	return name.includes("minsync") ? "minsync" : "inl";
}

/**
 * The launcher to start the daemon with: an explicit one, else `inl` when it is on PATH, else the `minsync`
 * fallback (resolved by the caller, or found on PATH). Throws {@link InlDaemonError} when neither exists.
 */
async function resolveLauncher(
	options: EnsureDaemonOptions,
	env: NodeJS.ProcessEnv,
): Promise<{ readonly path: string; readonly kind: DaemonLauncher }> {
	if (options.launcher !== undefined) {
		return { path: options.launcher.path, kind: options.launcher.kind ?? inferLauncherKind(options.launcher.path) };
	}
	const inl = lookupExecutableInPath(env, "inl");
	if (inl !== undefined) return { path: inl, kind: "inl" };
	if (options.resolveMinSyncBinary !== undefined) {
		const path = await options.resolveMinSyncBinary();
		return { path, kind: inferLauncherKind(path) };
	}
	const minsync = lookupExecutableInPath(env, "minsync");
	if (minsync !== undefined) return { path: minsync, kind: "minsync" };
	throw new InlDaemonError(
		"neither `inl` nor `minsync` is on PATH: install im-not-llm (`cargo install inl-server`) or MinSync, then retry",
	);
}

/** The argv that starts `kind`'s server, per launcher. */
function launcherArgs(kind: DaemonLauncher, options: EnsureDaemonOptions): readonly string[] {
	if (kind === "inl") {
		const args = ["serve"];
		if (options.idleTimeoutSeconds !== undefined) args.push("--idle-timeout", String(options.idleTimeoutSeconds));
		const engine = options.kind ?? "dense";
		if (engine === "sparse") args.push("--sparse");
		else if (engine === "multimodal") args.push("--multimodal");
		else if (engine === "late") args.push("--late");
		if (options.model !== undefined) args.push("--model", options.model);
		return args;
	}
	const args = ["serve", "--format", "json", "--quiet"];
	if (options.idleTimeoutSeconds !== undefined) args.push("--idle-timeout", String(options.idleTimeoutSeconds));
	return args;
}

/**
 * The running `inl serve` daemon of the wanted kind and model, starting one detached when none answers. im-not-llm
 * owns the model weights, the cache, and the idle shutdown. Concurrent starters are safe because both `inl serve`
 * and `minsync serve` adopt an already-running server of the same (kind, model) instead of loading a second copy.
 */
export async function ensureInlDaemon(options: EnsureDaemonOptions): Promise<InlDaemon> {
	const running = await probeInlDaemon(options);
	if (running !== undefined) return running;

	const env = options.env ?? process.env;
	const launcher = await resolveLauncher(options, env);
	const label = launcher.kind === "inl" ? "inl serve" : "minsync serve";
	const child = spawn(launcher.path, launcherArgs(launcher.kind, options), {
		env,
		...(options.cwd === undefined ? {} : { cwd: options.cwd }),
		detached: process.platform !== "win32",
		windowsHide: true,
		stdio: ["ignore", "ignore", "pipe"],
	});
	let stderr = "";
	child.stderr?.setEncoding("utf8");
	child.stderr?.on("data", (chunk: string) => {
		stderr = (stderr + chunk).slice(-STDERR_LIMIT);
	});
	let exit: { readonly code: number | null } | undefined;
	let spawnError: Error | undefined;
	child.on("error", (error) => {
		spawnError = error;
	});
	child.on("exit", (code) => {
		exit = { code };
	});

	const deadline = Date.now() + (options.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS);
	try {
		while (Date.now() < deadline) {
			if (spawnError !== undefined) {
				throw new InlDaemonError(`could not start \`${label}\`: ${spawnError.message}`, { cause: spawnError });
			}
			const found = await probeInlDaemon(options);
			if (found !== undefined) {
				child.stderr?.destroy();
				child.unref();
				return {
					...found,
					started: true,
					launcher: launcher.kind,
					...(child.pid === undefined ? {} : { pid: child.pid }),
				};
			}
			if (exit !== undefined) {
				// A launcher that found a running server for the same kind and model prints it and exits 0, and
				// that server answers the probe above. Exiting with nothing answering means nothing is serving.
				const answered = await probeInlDaemon(options);
				if (answered !== undefined) return { ...answered, started: false };
				throw new InlDaemonError(
					exit.code === 0
						? `\`${label}\` exited without a server being available: ${stderr.trim() || "(no output)"}`
						: `\`${label}\` exited with code ${exit.code ?? "unknown"} before it was ready: ${stderr.trim() || "(no output)"}`,
				);
			}
			await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
		}
		throw new InlDaemonError(
			`\`${label}\` was not ready within ${Math.round((options.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS) / 1000)}s: ${stderr.trim() || "(no output)"}`,
		);
	} catch (error) {
		if (exit === undefined) child.kill("SIGTERM");
		throw error;
	}
}

export interface StopDaemonOptions extends ProbeOptions {
	/** How long to wait for the server to remove its discovery files and stop answering. */
	readonly timeoutMs?: number;
}

/**
 * Stops the running `inl serve` daemon with SIGTERM, which it handles by removing its discovery files and
 * exiting. The pid comes from the server's own `GET /health` answer, so only a live loopback server of the
 * wanted kind and model is ever signalled. Returns the stopped daemon, or undefined when none ran.
 */
export async function stopInlDaemon(options: StopDaemonOptions = {}): Promise<InlDaemon | undefined> {
	const running = await probeInlDaemon(options);
	if (running === undefined) return undefined;
	if (running.pid === undefined) {
		throw new InlDaemonError("the running daemon did not report its pid, so it cannot be stopped from here");
	}
	process.kill(running.pid, "SIGTERM");
	const deadline = Date.now() + (options.timeoutMs ?? STOP_TIMEOUT_MS);
	while (Date.now() < deadline) {
		if ((await probeInlDaemon(options)) === undefined) return running;
		await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
	}
	throw new InlDaemonError(
		`the daemon (pid ${running.pid}) did not stop within ${(options.timeoutMs ?? STOP_TIMEOUT_MS) / 1000}s`,
	);
}
