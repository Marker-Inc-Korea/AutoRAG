import { spawn } from "node:child_process";
import { request } from "node:http";

const LOOPBACK_HOSTS = new Set(["127.0.0.1", "localhost", "::1"]);
const OLLAMA_PORT = 11434;
const DEFAULT_TIMEOUT_MS = 10_000;
const PROBE_INTERVAL_MS = 100;

export class LocalEmbedderError extends Error {
	readonly baseUrl: string;

	constructor(message: string, baseUrl: string) {
		super(message);
		this.name = "LocalEmbedderError";
		this.baseUrl = baseUrl;
	}
}

export interface LocalEmbedderPreflightOptions {
	readonly baseUrl?: string;
	readonly timeoutMs?: number;
	readonly probe?: () => Promise<boolean>;
	readonly start?: () => Promise<void>;
}

/**
 * Ensure a directly configured Ollama OpenAI-compatible endpoint is available.
 * TEI endpoints remain caller-managed because AutoRAG cannot assume an adapter
 * executable is installed in a published package.
 */
export async function ensureLocalEmbedder(options: LocalEmbedderPreflightOptions): Promise<void> {
	const baseUrl = options.baseUrl;
	if (baseUrl === undefined || !isDirectOllamaUrl(baseUrl)) return;

	const probe = options.probe ?? (() => probeEndpoint(baseUrl));
	if (await probe()) return;

	await (options.start ?? startOllama)();
	const deadline = Date.now() + (options.timeoutMs ?? DEFAULT_TIMEOUT_MS);
	while (Date.now() < deadline) {
		if (await probe()) return;
		await new Promise((resolve) => setTimeout(resolve, PROBE_INTERVAL_MS));
	}
	throw new LocalEmbedderError(`Ollama did not become ready at ${baseUrl}`, baseUrl);
}

function isDirectOllamaUrl(value: string): boolean {
	try {
		const url = new URL(value);
		return url.protocol === "http:" && LOOPBACK_HOSTS.has(url.hostname) && url.port === String(OLLAMA_PORT);
	} catch {
		return false;
	}
}

function probeEndpoint(baseUrl: string): Promise<boolean> {
	const url = new URL("/api/tags", baseUrl);
	return new Promise((resolve) => {
		const req = request(
			{
				hostname: url.hostname,
				port: url.port,
				path: url.pathname,
				method: "GET",
				timeout: 1_000,
			},
			(response) => {
				response.resume();
				resolve(response.statusCode !== undefined && response.statusCode >= 200 && response.statusCode < 500);
			},
		);
		req.on("error", () => resolve(false));
		req.on("timeout", () => {
			req.destroy();
			resolve(false);
		});
		req.end();
	});
}

function startOllama(): Promise<void> {
	return new Promise((resolve, reject) => {
		const child = spawn("ollama", ["serve"], {
			detached: true,
			stdio: "ignore",
		});
		let settled = false;
		child.once("error", (error) => {
			if (settled) return;
			settled = true;
			reject(error);
		});
		child.unref();
		settled = true;
		resolve();
	});
}
