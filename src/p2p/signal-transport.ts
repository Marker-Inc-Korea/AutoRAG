import { spawn } from "node:child_process";

/**
 * Thin adapter over the signal-cli subprocess (GPLv3).
 *
 * signal-cli is never linked; it is spawned as an external process and driven
 * through its JSON-RPC HTTP daemon (`signal-cli daemon --http`). Registration
 * and verification are one-shot CLI invocations.
 */

const DEFAULT_BINARY = "signal-cli";
const DEFAULT_HOST = "127.0.0.1";
const DEFAULT_PORT = 8080;
const RPC_PATH = "/api/v1/rpc";
const EVENTS_PATH = "/api/v1/events";
const DAEMON_READY_TIMEOUT_MS = 10_000;
const DAEMON_READY_POLL_MS = 100;

export interface SpawnedProcess {
	readonly pid: number;
	kill(): void;
	onExit(): Promise<number>;
	stderrText(): string;
}

export type SpawnProcess = (args: readonly string[]) => SpawnedProcess;

export interface SignalIncomingMessage {
	readonly source: string;
	readonly sourceUuid?: string;
	readonly message: string;
	readonly timestamp: number;
}

export interface SignalTransport {
	readonly account: string;
	sendMessage(recipient: string, message: string): Promise<void>;
	/** Subscribe to incoming data messages; returns an unsubscribe function. */
	onMessage(handler: (message: SignalIncomingMessage) => void): () => void;
	close(): Promise<void>;
}

export interface StartSignalDaemonOptions {
	readonly account: string;
	readonly binary?: string;
	readonly dataDir?: string;
	readonly host?: string;
	readonly port?: number;
	readonly spawnProcess?: SpawnProcess;
	readonly readyTimeoutMs?: number;
}

export interface RegisterAccountOptions {
	readonly number: string;
	readonly binary?: string;
	readonly dataDir?: string;
	readonly captcha?: string;
	readonly voice?: boolean;
	readonly spawnProcess?: SpawnProcess;
}

export interface VerifyAccountOptions {
	readonly number: string;
	readonly code: string;
	readonly binary?: string;
	readonly dataDir?: string;
	readonly pin?: string;
	readonly spawnProcess?: SpawnProcess;
}

class SignalCliError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "SignalCliError";
	}
}

function defaultSpawn(binary: string): SpawnProcess {
	return (args) => {
		const child = spawn(binary, [...args], { stdio: ["ignore", "ignore", "pipe"] });
		let stderr = "";
		child.stderr?.on("data", (chunk: Buffer) => {
			stderr += chunk.toString("utf8");
		});
		return {
			pid: child.pid ?? -1,
			kill: () => {
				child.kill("SIGTERM");
			},
			onExit: () =>
				new Promise<number>((resolve) => {
					child.on("close", (code) => resolve(code ?? -1));
				}),
			stderrText: () => stderr,
		};
	};
}

function baseArgs(options: { readonly account?: string; readonly dataDir?: string }): string[] {
	const args: string[] = [];
	if (options.dataDir !== undefined) args.push("--data-dir", options.dataDir);
	if (options.account !== undefined) args.push("-a", options.account);
	return args;
}

async function runOneShot(
	binary: string,
	args: readonly string[],
	spawnProcess: SpawnProcess | undefined,
): Promise<void> {
	const spawnFn = spawnProcess ?? defaultSpawn(binary);
	const proc = spawnFn(args);
	const code = await proc.onExit();
	if (code !== 0) {
		const detail = proc.stderrText().trim();
		throw new SignalCliError(
			`signal-cli ${args.join(" ")} exited with code ${code}${detail.length > 0 ? `: ${detail}` : ""}`,
		);
	}
}

/**
 * Register a new Signal account with a phone number. Sends an SMS (or voice
 * call when `voice` is set) with a verification code. A signalcaptcha:// token
 * may be required when Signal rate-limits the number.
 */
export async function registerAccount(options: RegisterAccountOptions): Promise<void> {
	const args = [
		...baseArgs({ account: options.number, dataDir: options.dataDir }),
		"register",
		...(options.voice === true ? ["--voice"] : []),
		...(options.captcha !== undefined ? ["--captcha", options.captcha] : []),
	];
	await runOneShot(options.binary ?? DEFAULT_BINARY, args, options.spawnProcess);
}

/** Complete registration with the code delivered by SMS/voice. */
export async function verifyAccount(options: VerifyAccountOptions): Promise<void> {
	const args = [
		...baseArgs({ account: options.number, dataDir: options.dataDir }),
		"verify",
		options.code,
		...(options.pin !== undefined ? ["--pin", options.pin] : []),
	];
	await runOneShot(options.binary ?? DEFAULT_BINARY, args, options.spawnProcess);
}

interface JsonRpcResponse {
	readonly jsonrpc: string;
	readonly id: number;
	readonly result?: unknown;
	readonly error?: { readonly code: number; readonly message: string };
}

class SignalDaemonTransport implements SignalTransport {
	readonly account: string;
	private readonly rpcUrl: string;
	private readonly eventsUrl: string;
	private readonly proc: SpawnedProcess;
	private readonly handlers: ((message: SignalIncomingMessage) => void)[] = [];
	private nextId = 1;
	private eventsAbort: AbortController | undefined;
	private closed = false;

	constructor(account: string, host: string, port: number, proc: SpawnedProcess) {
		this.account = account;
		this.rpcUrl = `http://${host}:${port}${RPC_PATH}`;
		this.eventsUrl = `http://${host}:${port}${EVENTS_PATH}`;
		this.proc = proc;
	}

	async sendMessage(recipient: string, message: string): Promise<void> {
		await this.rpc("send", { recipient: [recipient], message });
	}

	onMessage(handler: (message: SignalIncomingMessage) => void): () => void {
		this.handlers.push(handler);
		return () => {
			const index = this.handlers.indexOf(handler);
			if (index !== -1) this.handlers.splice(index, 1);
		};
	}

	async close(): Promise<void> {
		if (this.closed) return;
		this.closed = true;
		this.eventsAbort?.abort();
		this.proc.kill();
		await this.proc.onExit().catch(() => -1);
	}

	startEvents(): void {
		this.eventsAbort = new AbortController();
		void this.consumeEvents(this.eventsAbort.signal);
	}

	private async rpc(method: string, params: unknown): Promise<unknown> {
		const id = this.nextId++;
		const response = await fetch(this.rpcUrl, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ jsonrpc: "2.0", id, method, params }),
		});
		if (!response.ok) {
			throw new SignalCliError(`signal-cli RPC ${method} failed with HTTP ${response.status}`);
		}
		const body = (await response.json()) as JsonRpcResponse;
		if (body.error !== undefined) {
			throw new SignalCliError(`signal-cli RPC ${method} error ${body.error.code}: ${body.error.message}`);
		}
		return body.result;
	}

	private async consumeEvents(signal: AbortSignal): Promise<void> {
		while (!this.closed) {
			try {
				const response = await fetch(this.eventsUrl, {
					headers: { accept: "text/event-stream" },
					signal,
				});
				if (!response.ok || response.body === null) {
					throw new SignalCliError(`signal-cli events stream failed with HTTP ${response.status}`);
				}
				await this.readEventStream(response.body, signal);
			} catch (error) {
				if (this.closed || signal.aborted) return;
				// Back off briefly before reconnecting a dropped event stream.
				await new Promise((resolve) => setTimeout(resolve, 250));
				if (error instanceof SignalCliError) continue;
			}
		}
	}

	private async readEventStream(body: ReadableStream<Uint8Array>, signal: AbortSignal): Promise<void> {
		const reader = body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";
		try {
			for (;;) {
				const { done, value } = await reader.read();
				if (done) return;
				buffer += decoder.decode(value, { stream: true });
				let boundary = buffer.indexOf("\n\n");
				while (boundary !== -1) {
					const rawEvent = buffer.slice(0, boundary);
					buffer = buffer.slice(boundary + 2);
					this.dispatchEvent(rawEvent);
					boundary = buffer.indexOf("\n\n");
				}
			}
		} finally {
			reader.releaseLock();
			if (!signal.aborted) body.cancel().catch(() => {});
		}
	}

	private dispatchEvent(rawEvent: string): void {
		const dataLines = rawEvent
			.split("\n")
			.filter((line) => line.startsWith("data:"))
			.map((line) => line.slice(5).trimStart());
		if (dataLines.length === 0) return;
		let parsed: unknown;
		try {
			parsed = JSON.parse(dataLines.join("\n"));
		} catch {
			return;
		}
		const message = extractIncomingMessage(parsed);
		if (message === undefined) return;
		for (const handler of this.handlers) {
			try {
				handler(message);
			} catch {
				// A subscriber must not break the event stream.
			}
		}
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function extractIncomingMessage(event: unknown): SignalIncomingMessage | undefined {
	if (!isRecord(event)) return undefined;
	const params = event.params;
	if (!isRecord(params)) return undefined;
	const envelope = params.envelope;
	if (!isRecord(envelope)) return undefined;
	const dataMessage = envelope.dataMessage;
	if (!isRecord(dataMessage)) return undefined;
	if (typeof dataMessage.message !== "string") return undefined;
	const source = typeof envelope.sourceNumber === "string" ? envelope.sourceNumber : envelope.source;
	if (typeof source !== "string") return undefined;
	const timestamp = typeof dataMessage.timestamp === "number" ? dataMessage.timestamp : 0;
	const sourceUuid = typeof envelope.sourceUuid === "string" ? envelope.sourceUuid : undefined;
	return { source, ...(sourceUuid === undefined ? {} : { sourceUuid }), message: dataMessage.message, timestamp };
}

async function waitForDaemon(url: string, timeoutMs: number): Promise<void> {
	const deadline = Date.now() + timeoutMs;
	for (;;) {
		try {
			const response = await fetch(url, {
				method: "POST",
				headers: { "content-type": "application/json" },
				body: JSON.stringify({ jsonrpc: "2.0", id: 0, method: "version", params: {} }),
			});
			if (response.ok) return;
		} catch {
			// not up yet
		}
		if (Date.now() >= deadline) {
			throw new SignalCliError(`signal-cli daemon did not become ready within ${timeoutMs}ms`);
		}
		await new Promise((resolve) => setTimeout(resolve, DAEMON_READY_POLL_MS));
	}
}

/**
 * Spawn `signal-cli daemon --http` for one account and return a transport
 * connected to its JSON-RPC endpoint and SSE event stream.
 */
export async function startSignalDaemon(options: StartSignalDaemonOptions): Promise<SignalTransport> {
	const host = options.host ?? DEFAULT_HOST;
	const port = options.port ?? DEFAULT_PORT;
	const spawnFn = options.spawnProcess ?? defaultSpawn(options.binary ?? DEFAULT_BINARY);
	const args = [
		...baseArgs({ account: options.account, dataDir: options.dataDir }),
		"daemon",
		"--http",
		`${host}:${port}`,
		"--no-receive-stdout",
	];
	const proc = spawnFn(args);
	const transport = new SignalDaemonTransport(options.account, host, port, proc);
	try {
		await waitForDaemon(`http://${host}:${port}${RPC_PATH}`, options.readyTimeoutMs ?? DAEMON_READY_TIMEOUT_MS);
	} catch (error) {
		proc.kill();
		throw error;
	}
	transport.startEvents();
	return transport;
}
