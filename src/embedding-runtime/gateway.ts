import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import type { RuntimeProfile } from "./types.ts";

export type GatewayErrorCode =
	| "non-loopback-host"
	| "non-loopback-upstream"
	| "bad-request"
	| "batch-limit"
	| "upstream"
	| "upstream-violation"
	| "timeout";

export class GatewayError extends Error {
	readonly code: GatewayErrorCode;
	readonly status: number;

	constructor(
		code: GatewayErrorCode,
		message: string,
		status = statusFor(code),
		options: { readonly cause?: unknown } = {},
	) {
		super(message, { cause: options.cause });
		this.name = "GatewayError";
		this.code = code;
		this.status = status;
	}
}

export interface StartEmbeddingGatewayOptions {
	readonly profile: RuntimeProfile;
	readonly host?: string;
	readonly port?: number;
	readonly upstreamUrl: string;
	readonly fetch?: typeof fetch;
	readonly maxInputs?: number;
	readonly maxCharacters?: number;
	readonly timeoutMs?: number;
}

export interface EmbeddingGateway {
	readonly host: string;
	readonly port: number;
	readonly url: string;
	close(): Promise<void>;
}

const DEFAULT_MAX_INPUTS = 128;
const DEFAULT_MAX_CHARACTERS = 100_000;
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_BODY_BYTES = 2_000_000;

export function isLoopbackHost(host: string): boolean {
	const normalized = host
		.trim()
		.toLowerCase()
		.replace(/^\[|\]$/g, "");
	return normalized === "127.0.0.1" || normalized === "localhost" || normalized === "::1";
}

export async function startEmbeddingGateway(options: StartEmbeddingGatewayOptions): Promise<EmbeddingGateway> {
	const host = options.host ?? "127.0.0.1";
	if (!isLoopbackHost(host)) throw new GatewayError("non-loopback-host", "Gateway host must be loopback.");
	const upstream = parseLoopbackUrl(options.upstreamUrl);
	const fetchImpl = options.fetch ?? fetch;
	const server = createServer((req, res) => {
		void handleRequest(req, res, options, upstream, fetchImpl);
	});
	try {
		await listen(server, options.port ?? 0, host);
		const address = server.address();
		if (address === null || typeof address === "string")
			throw new GatewayError("upstream", "Gateway failed to bind.");
		const displayedHost = host.includes(":") ? `[${host}]` : host;
		return {
			host,
			port: address.port,
			url: `http://${displayedHost}:${address.port}`,
			close: () => close(server),
		};
	} catch (error) {
		server.close();
		throw error;
	}
}

function parseLoopbackUrl(raw: string): URL {
	let url: URL;
	try {
		url = new URL(raw);
	} catch (error) {
		throw new GatewayError("non-loopback-upstream", "Upstream URL must be a valid URL.", 400, { cause: error });
	}
	if (url.protocol !== "http:" && url.protocol !== "https:")
		throw new GatewayError("non-loopback-upstream", "Upstream URL must use HTTP(S).");
	if (!isLoopbackHost(url.hostname)) throw new GatewayError("non-loopback-upstream", "Upstream URL must be loopback.");
	return url;
}

async function handleRequest(
	req: IncomingMessage,
	res: ServerResponse,
	options: StartEmbeddingGatewayOptions,
	upstream: URL,
	fetchImpl: typeof fetch,
): Promise<void> {
	try {
		if (req.method === "GET" && req.url === "/healthz") {
			send(res, 200, {
				status: "ok",
				backend: options.profile.backend,
				model: options.profile.model,
				dimension: options.profile.dimension,
				runtimeBuild: options.profile.runtimeBuild,
				profileId: options.profile.profileId,
			});
			return;
		}
		if (req.method !== "POST" || !["/embed", "/v1/embeddings", "/api/embeddings"].includes(req.url ?? "")) {
			send(res, 404, { error: "Not found", code: "not-found" });
			return;
		}
		const body = await readJson(req);
		let inputs: string[];
		if (req.url === "/api/embeddings") {
			if (!isRecord(body) || typeof body.prompt !== "string")
				throw new GatewayError("bad-request", "prompt must be a string.");
			inputs = [body.prompt];
		} else {
			const candidate =
				req.url === "/embed" ? (isRecord(body) ? body.inputs : undefined) : isRecord(body) ? body.input : undefined;
			if (!Array.isArray(candidate))
				throw new GatewayError("bad-request", "inputs/input must be an array of strings.");
			inputs = candidate as string[];
		}
		validateInputs(inputs, options);
		const rows = await requestEmbeddings(
			inputs,
			options.profile,
			upstream,
			fetchImpl,
			options.timeoutMs ?? DEFAULT_TIMEOUT_MS,
		);
		if (req.url === "/embed") send(res, 200, rows);
		else if (req.url === "/api/embeddings") send(res, 200, { embedding: rows[0] });
		else
			send(res, 200, {
				data: rows.map((embedding, index) => ({ object: "embedding", embedding, index })),
				model: options.profile.model,
				object: "list",
			});
	} catch (error) {
		const failure =
			error instanceof GatewayError
				? error
				: new GatewayError("upstream", "Embedding upstream request failed.", 502, { cause: error });
		send(res, failure.status, { error: failure.message, code: failure.code });
	}
}

async function requestEmbeddings(
	inputs: readonly string[],
	profile: RuntimeProfile,
	upstream: URL,
	fetchImpl: typeof fetch,
	timeoutMs: number,
): Promise<number[][]> {
	const endpoint = new URL("/v1/embeddings", upstream);
	const signal = AbortSignal.timeout(timeoutMs);
	let response: Response;
	try {
		response = await fetchImpl(endpoint, {
			method: "POST",
			headers: { "content-type": "application/json" },
			body: JSON.stringify({ model: profile.model, input: inputs }),
			signal,
		});
	} catch (error) {
		if (signal.aborted) throw new GatewayError("timeout", `Embedding upstream timed out after ${timeoutMs}ms.`);
		throw new GatewayError("upstream", "Embedding upstream request failed.", 502, { cause: error });
	}
	if (!response.ok) throw new GatewayError("upstream", `Embedding upstream returned HTTP ${response.status}.`, 502);
	let payload: unknown;
	try {
		payload = await response.json();
	} catch (error) {
		throw new GatewayError("upstream-violation", "Embedding upstream returned invalid JSON.", 502, { cause: error });
	}
	if (!isRecord(payload) || !Array.isArray(payload.data) || payload.data.length !== inputs.length)
		throw new GatewayError("upstream-violation", "Embedding upstream returned the wrong number of rows.", 502);
	const indexed: Array<{ index: number; embedding: number[] }> = [];
	for (const item of payload.data) {
		if (!isRecord(item) || !Number.isInteger(item.index) || !Array.isArray(item.embedding))
			throw new GatewayError("upstream-violation", "Embedding upstream returned an invalid row.", 502);
		const index = item.index as number;
		if (index < 0 || index >= inputs.length || indexed.some((row) => row.index === index))
			throw new GatewayError("upstream-violation", "Embedding upstream returned invalid indices.", 502);
		const embedding = item.embedding as number[];
		if (
			embedding.length !== profile.dimension ||
			embedding.some((value) => typeof value !== "number" || !Number.isFinite(value))
		)
			throw new GatewayError("upstream-violation", "Embedding upstream returned an invalid embedding.", 502);
		indexed.push({ index, embedding });
	}
	indexed.sort((a, b) => a.index - b.index);
	return indexed.map((row) => row.embedding);
}

function validateInputs(inputs: readonly string[], options: StartEmbeddingGatewayOptions): void {
	if (inputs.length === 0 || inputs.some((input) => typeof input !== "string" || input.length === 0))
		throw new GatewayError("bad-request", "Inputs must be a non-empty array of non-empty strings.");
	if (
		inputs.length > (options.maxInputs ?? DEFAULT_MAX_INPUTS) ||
		inputs.reduce((total, input) => total + input.length, 0) > (options.maxCharacters ?? DEFAULT_MAX_CHARACTERS)
	)
		throw new GatewayError("batch-limit", "Embedding batch exceeds the configured limit.", 413);
}

async function readJson(req: IncomingMessage): Promise<unknown> {
	const chunks: Buffer[] = [];
	let size = 0;
	for await (const chunk of req) {
		size += chunk.length;
		if (size > MAX_BODY_BYTES) throw new GatewayError("batch-limit", "Request body is too large.", 413);
		chunks.push(Buffer.from(chunk));
	}
	try {
		return JSON.parse(Buffer.concat(chunks).toString("utf8")) as unknown;
	} catch (error) {
		throw new GatewayError("bad-request", "Request body must be valid JSON.", 400, { cause: error });
	}
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}
function statusFor(code: GatewayErrorCode): number {
	if (code === "bad-request" || code === "non-loopback-upstream") return 400;
	if (code === "batch-limit") return 413;
	if (code === "timeout") return 504;
	if (code === "non-loopback-host") return 400;
	return 502;
}
function send(res: ServerResponse, status: number, payload: unknown): void {
	res.statusCode = status;
	res.setHeader("content-type", "application/json");
	res.end(JSON.stringify(payload));
}
function listen(server: Server, port: number, host: string): Promise<void> {
	return new Promise((resolve, reject) => {
		const onError = (error: Error) => {
			server.off("error", onError);
			reject(error);
		};
		server.once("error", onError);
		server.listen(port, host, () => {
			server.off("error", onError);
			resolve();
		});
	});
}
function close(server: Server): Promise<void> {
	return new Promise((resolve, reject) => server.close((error) => (error ? reject(error) : resolve())));
}
