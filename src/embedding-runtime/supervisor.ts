import { type ChildProcess, spawn as defaultSpawn, type SpawnOptions } from "node:child_process";
import { appendFile, mkdir, open, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import { createServer } from "node:net";
import { platform } from "node:os";
import { basename, join } from "node:path";
import { resolveAutoRAGHome } from "../config/home.ts";
import { cacheDirectory } from "./cache.ts";
import { resolveProfile } from "./manifest.ts";
import type { BackendKind, ProfileId } from "./types.ts";

export type SupervisorState = "stopped" | "starting" | "ready" | "failed" | "stopping";

export interface SupervisorStatus {
	readonly state: SupervisorState;
	readonly pid?: number;
	readonly port?: number;
	readonly backend: BackendKind;
	readonly model: string;
	readonly uptimeMs?: number;
	readonly lastError?: string;
}

export class SupervisorError extends Error {
	readonly code: "lock-conflict" | "readiness-timeout" | "spawn" | "exited" | "shutdown" | "missing-executable";
	constructor(code: SupervisorError["code"], message: string, options: { readonly cause?: unknown } = {}) {
		super(message, { cause: options.cause });
		this.name = "SupervisorError";
		this.code = code;
	}
}

type SpawnLike = (command: string, args: readonly string[], options: SpawnOptions) => ChildProcess;
type KillLike = (pid: number, signal?: NodeJS.Signals | number) => void;

export interface SupervisorOptions {
	readonly cacheRoot?: string;
	readonly profileId?: ProfileId;
	readonly backend?: BackendKind;
	readonly executablePath?: string;
	readonly modelPath?: string;
	readonly readinessTimeoutMs?: number;
	readonly readinessIntervalMs?: number;
	readonly logMaxBytes?: number;
	readonly fetch?: typeof globalThis.fetch;
	readonly spawn?: SpawnLike;
	readonly kill?: KillLike;
	readonly platform?: NodeJS.Platform;
}

const DEFAULT_TIMEOUT = 15_000;
const DEFAULT_INTERVAL = 100;
const DEFAULT_LOG_BYTES = 256 * 1024;

export class EmbeddingRuntimeSupervisor {
	private readonly root: string;
	private readonly backend: BackendKind;
	private readonly executablePath: string;
	private readonly modelPath: string;
	private readonly timeoutMs: number;
	private readonly intervalMs: number;
	private readonly logMaxBytes: number;
	private readonly spawnProcess: SpawnLike;
	private readonly killProcess: KillLike;
	private readonly fetchHealth: typeof globalThis.fetch;
	private readonly osPlatform: NodeJS.Platform;
	private readonly defaultExecutable: boolean;
	private child?: ChildProcess;
	private startedAt?: number;
	private currentPort?: number;
	private state: SupervisorState = "stopped";
	private lastError?: string;
	private restartCount = 0;
	private shutdownRequested = false;
	private stopping?: Promise<void>;

	constructor(options: SupervisorOptions = {}) {
		const profile = resolveProfile(options.profileId ?? "qwen3-embedding-0.6b");
		this.root = options.cacheRoot ?? resolveAutoRAGHome();
		this.backend = options.backend ?? profile.backend;
		this.osPlatform = options.platform ?? platform();
		this.defaultExecutable = options.executablePath === undefined;
		this.executablePath =
			options.executablePath ??
			(this.osPlatform === "win32"
				? join(cacheDirectory("runtime", this.root), "llama-server.exe")
				: join(cacheDirectory("runtime", this.root), "bin", "llama-server"));
		this.modelPath = options.modelPath ?? join(cacheDirectory("models", this.root), basename(profile.model));
		this.timeoutMs = options.readinessTimeoutMs ?? DEFAULT_TIMEOUT;
		this.intervalMs = options.readinessIntervalMs ?? DEFAULT_INTERVAL;
		this.logMaxBytes = options.logMaxBytes ?? DEFAULT_LOG_BYTES;
		this.spawnProcess = options.spawn ?? ((command, args, spawnOptions) => defaultSpawn(command, args, spawnOptions));
		this.killProcess =
			options.kill ??
			((pid, signal) => {
				if (this.osPlatform === "win32" && signal === "SIGKILL") {
					defaultSpawn("taskkill", ["/PID", String(pid), "/T", "/F"], { stdio: "ignore", windowsHide: true });
					return;
				}
				process.kill(pid, signal);
			});
		this.fetchHealth = options.fetch ?? globalThis.fetch;
	}

	async ensureRunning(): Promise<SupervisorStatus> {
		if (this.state === "ready" && this.child && this.currentPort) return this.status();
		if (this.stopping) await this.stopping;
		await this.acquireLock();
		this.shutdownRequested = false;
		this.restartCount = 0;
		this.state = "starting";
		this.lastError = undefined;
		try {
			while (true) {
				await this.startChild();
				try {
					await this.waitUntilReady();
					this.state = "ready";
					return this.status();
				} catch (error) {
					if (
						error instanceof SupervisorError &&
						error.code === "exited" &&
						this.restartCount < 1 &&
						!this.shutdownRequested
					) {
						this.restartCount += 1;
						continue;
					}
					throw error;
				}
			}
		} catch (error) {
			this.lastError = sanitizeError(error);
			if (this.child) await this.stopChild();
			this.state = "failed";
			await this.cleanupState();
			if (error instanceof SupervisorError) throw error;
			throw new SupervisorError("spawn", this.lastError, { cause: error });
		}
	}

	status(): SupervisorStatus {
		return {
			state: this.state,
			pid: this.child?.pid,
			port: this.currentPort,
			backend: this.backend,
			model: basename(this.modelPath),
			uptimeMs: this.startedAt ? Math.max(0, Date.now() - this.startedAt) : undefined,
			lastError: this.lastError,
		};
	}

	async shutdown(): Promise<void> {
		if (this.stopping) return this.stopping;
		if (!this.child) {
			this.state = "stopped";
			await this.cleanupState();
			return;
		}
		this.stopping = this.stopChild();
		try {
			await this.stopping;
		} finally {
			this.stopping = undefined;
		}
	}

	private async acquireLock(): Promise<void> {
		await mkdir(this.root, { recursive: true });
		const lockPath = join(this.root, "embedding-runtime.lock");
		try {
			const handle = await open(lockPath, "wx");
			await handle.writeFile(`${process.pid}\n`);
			await handle.close();
		} catch (error) {
			if ((error as NodeJS.ErrnoException).code !== "EEXIST") throw error;
			const pid = await readPid(join(this.root, "embedding-runtime.pid"));
			if (pid && isAlive(pid))
				throw new SupervisorError("lock-conflict", "Another embedding runtime is already running.");
			await rm(lockPath, { force: true });
			await rm(join(this.root, "embedding-runtime.pid"), { force: true });
			const handle = await open(lockPath, "wx");
			await handle.writeFile(`${process.pid}\n`);
			await handle.close();
		}
	}

	private async resolveExecutable(): Promise<string> {
		if (!this.defaultExecutable) return this.executablePath;
		try {
			if (await stat(this.executablePath)) return this.executablePath;
		} catch {
			// Search atomically extracted runtime trees.
		}
		const runtimeRoot = cacheDirectory("runtime", this.root);
		try {
			for (const entry of await readdir(runtimeRoot, { withFileTypes: true })) {
				if (!entry.isDirectory() || !entry.name.endsWith(".extracted")) continue;
				const candidate = join(
					runtimeRoot,
					entry.name,
					this.osPlatform === "win32" ? "llama-server.exe" : "bin",
					...(this.osPlatform === "win32" ? [] : ["llama-server"]),
				);
				try {
					await stat(candidate);
					return candidate;
				} catch {
					/* continue */
				}
			}
		} catch {
			/* handled below */
		}
		return this.executablePath;
	}

	private async startChild(): Promise<void> {
		const executablePath = await this.resolveExecutable();
		try {
			await stat(executablePath);
		} catch (error) {
			throw new SupervisorError(
				"missing-executable",
				`Embedding runtime executable is missing. Run autorag models prefetch.`,
				{ cause: error },
			);
		}
		this.currentPort = await reservePort();
		const args = [
			"--embeddings",
			"--model",
			this.modelPath,
			"--host",
			"127.0.0.1",
			"--port",
			String(this.currentPort),
		];
		const logPath = join(this.root, "embedding-runtime.log");
		await mkdir(this.root, { recursive: true });
		const spawnOptions: SpawnOptions = { stdio: ["ignore", "pipe", "pipe"], detached: this.osPlatform !== "win32" };
		let child: ChildProcess;
		try {
			child = this.spawnProcess(executablePath, args, spawnOptions);
		} catch (error) {
			throw new SupervisorError("spawn", "Unable to start embedding runtime.", { cause: error });
		}
		this.child = child;
		this.startedAt = Date.now();
		await writeFile(join(this.root, "embedding-runtime.pid"), `${child.pid ?? ""}\n`);
		const consume = (chunk: Buffer) => void this.appendLog(logPath, chunk.toString());
		child.stdout?.on("data", consume);
		child.stderr?.on("data", consume);
		child.once("error", (error) => {
			this.lastError = sanitizeError(error);
		});
		child.once("exit", () => {
			if (!this.shutdownRequested && this.state === "ready") void this.recoverUnexpectedExit();
		});
	}

	private async recoverUnexpectedExit(): Promise<void> {
		if (this.restartCount >= 1) {
			this.state = "failed";
			this.lastError = "embedding runtime exited unexpectedly after restart";
			await this.cleanupState();
			return;
		}
		this.restartCount += 1;
		this.state = "starting";
		try {
			await this.startChild();
			await this.waitUntilReady();
			this.state = "ready";
		} catch (error) {
			this.lastError = sanitizeError(error);
			this.state = "failed";
			if (this.child) await this.stopChild();
		}
	}

	private async waitUntilReady(): Promise<void> {
		const deadline = Date.now() + this.timeoutMs;
		while (Date.now() < deadline) {
			if (this.shutdownRequested) throw new SupervisorError("shutdown", "Embedding runtime startup was cancelled.");
			if (!this.child || this.child.exitCode !== null || this.child.killed)
				throw new SupervisorError("exited", "Embedding runtime exited before becoming ready.");
			try {
				const response = await this.fetchHealth(`http://127.0.0.1:${this.currentPort}/health`);
				if (response.ok) return;
			} catch {
				// The child may still be binding its loopback port.
			}
			await new Promise<void>((resolve) => setTimeout(resolve, this.intervalMs));
		}
		const log = await this.readLog();
		throw new SupervisorError(
			"readiness-timeout",
			`Embedding runtime did not become ready: ${log || "health check timed out"}`,
		);
	}

	private async stopChild(): Promise<void> {
		this.shutdownRequested = true;
		this.state = "stopping";
		const child = this.child;
		if (!child) return;
		const exited = new Promise<void>((resolve) => {
			if (child.exitCode !== null || child.killed) return resolve();
			child.once("exit", () => resolve());
		});
		try {
			if (this.osPlatform === "win32") this.killProcess(child.pid ?? 0, "SIGTERM");
			else if (child.pid) this.killProcess(-child.pid, "SIGTERM");
		} catch {
			// A concurrently exiting child is already stopped.
		}
		await Promise.race([exited, delay(1_000)]);
		if (child.exitCode === null && !child.killed && child.pid) {
			try {
				if (this.osPlatform === "win32") this.killProcess(child.pid, "SIGKILL");
				else this.killProcess(-child.pid, "SIGKILL");
			} catch {
				// The process may have exited between the check and kill.
			}
		}
		await Promise.race([exited, delay(500)]);
		this.child = undefined;
		this.currentPort = undefined;
		this.startedAt = undefined;
		this.state = "stopped";
		await this.cleanupState();
	}

	private async cleanupState(): Promise<void> {
		await rm(join(this.root, "embedding-runtime.pid"), { force: true });
		await rm(join(this.root, "embedding-runtime.lock"), { force: true });
	}

	private async appendLog(path: string, text: string): Promise<void> {
		try {
			await appendFile(path, text);
			const info = await stat(path);
			if (info.size > this.logMaxBytes) {
				const content = await readFile(path);
				await writeFile(path, content.subarray(-this.logMaxBytes));
			}
		} catch {
			// Logging must not alter process supervision.
		}
	}

	private async readLog(): Promise<string> {
		try {
			return sanitizeLog(await readFile(join(this.root, "embedding-runtime.log"), "utf8"));
		} catch {
			return "";
		}
	}
}

async function reservePort(): Promise<number> {
	const server = createServer();
	await new Promise<void>((resolve, reject) => {
		server.once("error", reject);
		server.listen(0, "127.0.0.1", () => resolve());
	});
	const address = server.address();
	await new Promise<void>((resolve) => server.close(() => resolve()));
	if (!address || typeof address === "string")
		throw new SupervisorError("spawn", "Unable to allocate an internal port.");
	return address.port;
}

function isAlive(pid: number): boolean {
	try {
		process.kill(pid, 0);
		return true;
	} catch {
		return false;
	}
}

async function readPid(path: string): Promise<number | undefined> {
	try {
		const value = Number.parseInt((await readFile(path, "utf8")).trim(), 10);
		return Number.isInteger(value) && value > 0 ? value : undefined;
	} catch {
		return undefined;
	}
}

function sanitizeLog(value: string): string {
	return value
		.replaceAll(/(?:token|api[_-]?key|password|secret)\s*[=:]\s*[^\s,;]+/gi, "$1=[redacted]")
		.replaceAll(/[\\/][^\s]+/g, "[path]")
		.slice(-2_000);
}
function sanitizeError(error: unknown): string {
	return sanitizeLog(error instanceof Error ? error.message : String(error));
}
function delay(ms: number): Promise<void> {
	return new Promise((resolve) => setTimeout(resolve, ms));
}
