import { spawn } from "node:child_process";
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { resolveUserHome } from "../config/home.ts";

/** The model `minsync serve` serves by default; callers pass another to look for that one. */
export const DEFAULT_DAEMON_MODEL = "Qwen/Qwen3-Embedding-0.6B";
const LOOPBACK_HOSTS = new Set(["127.0.0.1", "::1", "localhost"]);
const HEALTH_TIMEOUT_MS = 2_000;
const DEFAULT_READY_TIMEOUT_MS = 10 * 60 * 1_000;
const POLL_INTERVAL_MS = 150;
const STDERR_LIMIT = 4_000;

export interface MinSyncDaemon {
	/** `http://127.0.0.1:<port>`; the OpenAI route lives under `/v1`, TEI at `/embed`. */
	readonly baseUrl: string;
	readonly model: string;
	readonly dimension: number;
	readonly instance: string;
	/** True when {@link ensureMinSyncDaemon} spawned it. */
	readonly started: boolean;
	readonly pid?: number;
}

export class MinSyncDaemonError extends Error {
	constructor(message: string, options?: ErrorOptions) {
		super(message, options);
		this.name = "MinSyncDaemonError";
	}
}

export interface ProbeOptions {
	/** User home; MinSync keeps its server state under `<home>/.cache/minsync/serve`. */
	readonly home?: string;
	/** Only accept a server of this model. Defaults to {@link DEFAULT_DAEMON_MODEL}. */
	readonly model?: string;
	readonly fetchImpl?: typeof fetch;
}

/**
 * MinSync publishes `host`/`port` (and `pid`) per model instance under
 * `~/.cache/minsync/serve/<instance>/` and removes them on a clean stop. A leftover file from a crashed
 * server is harmless because every candidate must answer `GET /health` as a loopback `status: ok` server of
 * the wanted model before it is used.
 */
export function minSyncServeStateRoot(home: string = resolveUserHome()): string {
	if (process.platform === "win32") {
		const local = process.env.LOCALAPPDATA ?? join(home, "AppData", "Local");
		return join(local, "minsync", "serve");
	}
	return join(home, ".cache", "minsync", "serve");
}

interface Health {
	readonly status?: string;
	readonly model_id?: string;
	readonly dimension?: number;
	readonly instance?: string;
	readonly pid?: number;
}

export async function probeMinSyncDaemon(options: ProbeOptions = {}): Promise<MinSyncDaemon | undefined> {
	const root = minSyncServeStateRoot(options.home);
	const wanted = options.model ?? DEFAULT_DAEMON_MODEL;
	const fetchImpl = options.fetchImpl ?? fetch;
	let instances: string[];
	try {
		instances = readdirSync(root);
	} catch {
		return undefined;
	}
	for (const instance of instances) {
		const dir = join(root, instance);
		const port = Number(readText(join(dir, "port")));
		const host = readText(join(dir, "host")) ?? "127.0.0.1";
		if (!Number.isInteger(port) || port <= 0 || !LOOPBACK_HOSTS.has(host)) continue;
		const baseUrl = `http://${host.includes(":") ? `[${host}]` : host}:${port}`;
		try {
			const response = await fetchImpl(`${baseUrl}/health`, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
			if (!response.ok) continue;
			const health = (await response.json()) as Health;
			if (health.status !== "ok" || health.model_id?.toLowerCase() !== wanted.toLowerCase()) continue;
			if (typeof health.dimension !== "number") continue;
			return {
				baseUrl,
				model: health.model_id,
				dimension: health.dimension,
				instance: health.instance ?? instance,
				started: false,
				...(typeof health.pid === "number" ? { pid: health.pid } : {}),
			};
		} catch {
			// Stale discovery files: nothing is listening there any more.
		}
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
	/** The `minsync` executable. Required only when no daemon is running yet. */
	readonly binaryPath: string;
	readonly env?: NodeJS.ProcessEnv;
	/** Working directory of the spawned server. `minsync serve` reads `.minsync/config.toml` from it. */
	readonly cwd?: string;
	/** How long to wait for the server to publish and answer; the first start may download the model. */
	readonly readyTimeoutMs?: number;
	/** Idle seconds before the daemon exits and frees the model (`minsync serve --idle-timeout`). */
	readonly idleTimeoutSeconds?: number;
}

/**
 * The running `minsync serve` daemon, starting one detached when none answers. The daemon is MinSync's:
 * it owns the model weights, the cache, and its own idle shutdown. Concurrent starters are safe because
 * `minsync serve` adopts an already-running server of the same model instead of loading a second copy.
 */
export async function ensureMinSyncDaemon(options: EnsureDaemonOptions): Promise<MinSyncDaemon> {
	const running = await probeMinSyncDaemon(options);
	if (running !== undefined) return running;

	const args = ["serve", "--format", "json", "--quiet"];
	if (options.idleTimeoutSeconds !== undefined) args.push("--idle-timeout", String(options.idleTimeoutSeconds));
	const child = spawn(options.binaryPath, args, {
		env: options.env ?? process.env,
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
				throw new MinSyncDaemonError(`could not start \`minsync serve\`: ${spawnError.message}`, {
					cause: spawnError,
				});
			}
			const found = await probeMinSyncDaemon(options);
			if (found !== undefined) {
				child.stderr?.destroy();
				child.unref();
				return { ...found, started: true, ...(child.pid === undefined ? {} : { pid: child.pid }) };
			}
			if (exit !== undefined) {
				// A `minsync serve` that found a running server for the same model prints it and exits 0, and that
				// server answers the probe above. Exiting with nothing answering means nothing is serving.
				const answered = await probeMinSyncDaemon(options);
				if (answered !== undefined) return { ...answered, started: false };
				throw new MinSyncDaemonError(
					exit.code === 0
						? `\`minsync serve\` exited without a server being available: ${stderr.trim() || "(no output)"}`
						: `\`minsync serve\` exited with code ${exit.code ?? "unknown"} before it was ready: ${stderr.trim() || "(no output)"}`,
				);
			}
			await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
		}
		throw new MinSyncDaemonError(
			`\`minsync serve\` was not ready within ${Math.round((options.readyTimeoutMs ?? DEFAULT_READY_TIMEOUT_MS) / 1000)}s: ${stderr.trim() || "(no output)"}`,
		);
	} catch (error) {
		if (exit === undefined) child.kill("SIGTERM");
		throw error;
	}
}
