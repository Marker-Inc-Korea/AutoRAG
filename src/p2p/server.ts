import { createHash } from "node:crypto";
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { Value } from "typebox/value";
import type { SearchDocumentsResponse } from "../agent/search-documents.ts";
import { planSourceRoots, type SourceRoot } from "../filesystem/source-paths.ts";
import { parsedMirrorRoot } from "../mirror/paths.ts";
import type { RetrievalOptions } from "../retrieval/types.ts";
import { buildPeerResponse as defaultBuildPeerResponse } from "./egress-gate.ts";
import { resolveFileShare } from "./file-sharing.ts";
import { loadPeerRegistry, type PeerRecord, type PeerRegistry, sha256Body, verifyRequest } from "./identity.ts";
import { classifyInjection, type InjectionClassifierModel } from "./injection-classifier.ts";
import { screenInboundQuery } from "./injection-gate.ts";
import { type PolicyQuotas, type PolicyResolution, PolicyStore } from "./policy.ts";
import type { PolicyResolver } from "./policy-filter.ts";
import { QueryQueue, QueryQueueFullError } from "./query-queue.ts";
import {
	type PeerQueryRequest,
	PeerQueryRequestSchema,
	type PeerQueryResponse,
	PeerQueryResponseSchema,
} from "./wire.ts";

const DEFAULT_HOST = "0.0.0.0";
const DEFAULT_PORT = 9470;
const DEFAULT_QUEUE_DEPTH = 8;
const DEFAULT_MAX_BODY_BYTES = 65_536;
const DEFAULT_SEARCH_TIMEOUT_MS = 120_000;
const FINGERPRINT_PATTERN = /^[a-f0-9]{64}$/i;
const TIMESTAMP_PATTERN = /^(?:0|[1-9][0-9]*)$/;

const PRIVATE_POLICY: PolicyResolution = {
	tier: "private",
	allowed: false,
	shareBytes: false,
	redact: true,
};

export interface P2pSearchAgent {
	/** The server must only invoke an agent constructed with remote enforcement. */
	/** Must be true; startP2pServer validates this before serving requests. */
	readonly remoteSession: boolean;
	searchDocuments(query: string, options?: RetrievalOptions): Promise<SearchDocumentsResponse>;
}

export interface P2pServerConfig {
	readonly host?: string;
	readonly port?: number;
	readonly maxBodyBytes?: number;
	readonly maxFileBytes?: number;
	readonly queueDepth?: number;
	readonly injectionClassifier?: boolean;
	readonly piiNer?: boolean;
	readonly searchTimeoutMs?: number;
	readonly pseudonymize?: boolean;
	readonly workspaceRoots?: readonly string[];
	readonly quotas?: Partial<PolicyQuotas>;
}

export interface PeerResponseBuilderArgs {
	readonly response: SearchDocumentsResponse;
	readonly observedSources: ReadonlySet<string>;
	readonly resolvePolicy: PolicyResolver;
	readonly peerFingerprint: string;
	readonly workspaceRoots: readonly string[];
	readonly pseudonymize: boolean;
}

export type PeerResponseBuilder = (args: PeerResponseBuilderArgs) => PeerQueryResponse | Promise<PeerQueryResponse>;

export interface P2pRequestLog {
	readonly event: "request" | "request_rejected";
	readonly method: string;
	readonly path: string;
	readonly bodySha256: string;
	readonly querySha256?: string;
	readonly peerFingerprint?: string;
	readonly code?: string;
}

export interface StartP2pServerOptions {
	readonly agent: P2pSearchAgent;
	readonly peers?: PeerRegistry;
	/** Alias accepted for callers that name the registry explicitly. */
	readonly peerRegistry?: PeerRegistry;
	readonly config?: P2pServerConfig;
	/** Top-level aliases keep the listener convenient for CLI/bootstrap callers. */
	readonly host?: string;
	readonly port?: number;
	readonly maxBodyBytes?: number;
	readonly maxFileBytes?: number;
	readonly queueDepth?: number;
	readonly injectionClassifier?: boolean;
	readonly piiNer?: boolean;
	readonly searchTimeoutMs?: number;
	readonly pseudonymize?: boolean;
	readonly workspaceRoots?: readonly string[];
	readonly workspacePath?: string;
	readonly quotas?: Partial<PolicyQuotas>;
	readonly policyStore?: {
		readonly quotas?: PolicyQuotas;
		resolvePolicy(source: string, peerFingerprint?: string): PolicyResolution;
	};
	readonly resolvePolicy?: PolicyResolver;
	readonly injectionClassifierModel?: InjectionClassifierModel;
	readonly buildPeerResponse?: PeerResponseBuilder;
	readonly logger?: (entry: P2pRequestLog) => void;
	/** Test/integration seam for observing queue admission without polling. */
	readonly onQueueEnqueued?: (pendingCount: number) => void;
}

export interface P2pServer {
	readonly url: string;
	readonly origin: string;
	readonly host: string;
	readonly port: number;
	readonly queue: QueryQueue<SearchDocumentsResponse>;
	close(): Promise<void>;
}

interface ParsedHeaders {
	readonly fingerprint: string;
	readonly timestamp: number;
	readonly signature: string;
}

interface ReadBodyResult {
	readonly ok: true;
	readonly body: Buffer;
}

interface ReadBodyFailure {
	readonly ok: false;
	readonly reason: "too-large" | "aborted";
}

type TokenBucket = {
	tokens: number;
	lastRefillMs: number;
};

export async function startP2pServer(options: StartP2pServerOptions): Promise<P2pServer> {
	if (!options.agent || typeof options.agent.searchDocuments !== "function") {
		throw new TypeError("A public agent.searchDocuments implementation is required.");
	}
	if (options.agent.remoteSession !== true) {
		throw new TypeError("P2P server requires an agent constructed with remoteSession: true.");
	}
	const buildPeerResponse = options.buildPeerResponse ?? defaultBuildPeerResponse;

	const config = options.config ?? {};
	const host = options.host ?? config.host ?? DEFAULT_HOST;
	const port = options.port ?? config.port ?? DEFAULT_PORT;
	const injectionClassifier = options.injectionClassifier ?? config.injectionClassifier ?? true;
	if (injectionClassifier && typeof options.injectionClassifierModel !== "function") {
		throw new TypeError("P2P injection classifier is enabled but no classifier model was provided.");
	}
	const policyStore =
		options.policyStore ?? (options.workspacePath ? new PolicyStore(options.workspacePath) : undefined);
	const quotas = resolveQuotas(options.quotas ?? config.quotas, policyStore?.quotas);
	const maxBodyBytes = options.maxBodyBytes ?? config.maxBodyBytes ?? quotas.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
	const maxFileBytes = options.maxFileBytes ?? config.maxFileBytes ?? quotas.maxFileBytes;
	const queue = new QueryQueue<SearchDocumentsResponse>(
		options.queueDepth ?? config.queueDepth ?? DEFAULT_QUEUE_DEPTH,
	);
	const peerRegistry =
		options.peers ?? options.peerRegistry ?? (options.workspacePath ? loadPeerRegistry(options.workspacePath) : {});
	const resolvePolicy =
		options.resolvePolicy ?? (policyStore ? policyStore.resolvePolicy.bind(policyStore) : () => PRIVATE_POLICY);
	const effectiveConfig: P2pServerConfig = {
		...config,
		injectionClassifier,
		piiNer: options.piiNer ?? config.piiNer,
		searchTimeoutMs: options.searchTimeoutMs ?? config.searchTimeoutMs,
		pseudonymize: options.pseudonymize ?? config.pseudonymize,
		workspaceRoots:
			options.workspaceRoots ?? config.workspaceRoots ?? (options.workspacePath ? [options.workspacePath] : []),
	};
	const sourceRoots = planSourceRoots(effectiveConfig.workspaceRoots ?? []);
	const parsedRoot = options.workspacePath !== undefined ? parsedMirrorRoot(options.workspacePath) : "";
	const logger = options.logger ?? ((entry: P2pRequestLog) => console.info(JSON.stringify(entry)));
	const buckets = new Map<string, TokenBucket>();
	const httpServer = createServer((request, response) => {
		void handleRequest(request, response, {
			agent: options.agent,
			buildPeerResponse,
			config: effectiveConfig,
			maxBodyBytes,
			maxFileBytes,
			peerRegistry,
			resolvePolicy,
			injectionClassifierModel: options.injectionClassifierModel,
			logger,
			buckets,
			queue,
			quotas,
			sourceRoots,
			parsedMirrorRoot: parsedRoot,
			onQueueEnqueued: options.onQueueEnqueued,
		});
	});

	await listen(httpServer, port, host);
	const address = httpServer.address();
	if (address === null || typeof address === "string") {
		httpServer.close();
		throw new Error("P2P server failed to bind a TCP port.");
	}
	const boundHost = formatHost(host);
	const origin = `http://${boundHost}:${address.port}`;
	return {
		url: origin,
		origin,
		host,
		port: address.port,
		queue,
		close: async () => {
			queue.close();
			await closeServer(httpServer);
		},
	};
}

export const startP2PServer = startP2pServer;
export const createP2pServer = startP2pServer;
export const createP2PServer = startP2pServer;

async function handleRequest(
	req: IncomingMessage,
	res: ServerResponse,
	ctx: {
		readonly agent: P2pSearchAgent;
		readonly buildPeerResponse: PeerResponseBuilder;
		readonly config: P2pServerConfig;
		readonly maxBodyBytes: number;
		readonly maxFileBytes: number;
		readonly peerRegistry: PeerRegistry;
		readonly resolvePolicy: PolicyResolver;
		readonly injectionClassifierModel: InjectionClassifierModel | undefined;
		readonly logger: (entry: P2pRequestLog) => void;
		readonly buckets: Map<string, TokenBucket>;
		readonly queue: QueryQueue<SearchDocumentsResponse>;
		readonly quotas: PolicyQuotas;
		readonly sourceRoots: readonly SourceRoot[];
		readonly parsedMirrorRoot: string;
		readonly onQueueEnqueued: ((pendingCount: number) => void) | undefined;
	},
): Promise<void> {
	const parsedUrl = new URL(req.url ?? "/", "http://p2p.invalid");
	const bodyResult = await readBody(req, ctx.maxBodyBytes);
	const body = bodyResult.ok ? bodyResult.body : Buffer.alloc(0);
	const bodySha256 = sha256Body(body);
	const headers = parseSignedHeaders(req);
	const baseLog: Omit<P2pRequestLog, "event"> = {
		method: req.method ?? "",
		path: parsedUrl.pathname,
		bodySha256,
		...(headers?.fingerprint === undefined ? {} : { peerFingerprint: headers.fingerprint }),
	};

	if (!bodyResult.ok) {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "internal-error" });
		sendRefusal(res, 413, "internal-error", "Request body exceeds the configured limit.");
		return;
	}

	const peer = headers === undefined ? undefined : findPeer(ctx.peerRegistry, headers.fingerprint);
	if (peer === undefined || headers === undefined) {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "auth-error" });
		sendRefusal(res, 401, "auth-error", "Peer authentication failed.");
		return;
	}
	const verified = verifyRequest(peer, headers.timestamp, bodySha256, headers.signature);
	if (!verified) {
		const replay = isReplay(peer, headers.timestamp);
		const code = replay ? "replay-rejected" : "auth-error";
		log(ctx.logger, { ...baseLog, event: "request_rejected", code });
		sendRefusal(
			res,
			replay ? 409 : 401,
			code,
			replay ? "The signed request was rejected as a replay." : "Peer authentication failed.",
		);
		return;
	}
	let parsedBody: unknown;
	if (body.length > 0) {
		try {
			parsedBody = JSON.parse(body.toString("utf8")) as unknown;
		} catch {
			log(ctx.logger, { ...baseLog, event: "request_rejected", code: "internal-error" });
			sendRefusal(res, 400, "internal-error", "Request body must be valid JSON.");
			return;
		}
	}
	const queryHash = queryHashFromBody(parsedBody);
	const requestLog: P2pRequestLog = {
		...baseLog,
		event: "request",
		...(queryHash === undefined ? {} : { querySha256: queryHash }),
	};
	log(ctx.logger, requestLog);

	if (req.method === "GET" && parsedUrl.pathname === "/v1/file") {
		const file = resolveFileShare(parsedUrl.searchParams.get("source") ?? "", headers.fingerprint, {
			resolvePolicy: ctx.resolvePolicy,
			workspaceRoots: ctx.sourceRoots,
			parsedMirrorRoot: ctx.parsedMirrorRoot,
			maxFileBytes: ctx.maxFileBytes,
		});
		sendJson(res, file.status === "ok" ? 200 : 403, file);
		return;
	}
	if (req.method !== "POST" || parsedUrl.pathname !== "/v1/query") {
		sendRefusal(res, 404, "internal-error", "Not found.");
		return;
	}
	if (!consumeToken(ctx.buckets, headers.fingerprint, ctx.quotas)) {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "rate-limited" });
		sendRefusal(res, 429, "rate-limited", "Peer query quota exceeded.");
		return;
	}

	const request = parseQueryRequest(parsedBody);
	if (request === undefined) {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "internal-error" });
		sendRefusal(res, 400, "internal-error", "Request body is not a valid peer query.");
		return;
	}
	const screened = screenInboundQuery(request.query);
	if (!screened.ok) {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "injection-detected" });
		sendRefusal(res, 400, "injection-detected", "The query was rejected by the inbound safety gate.");
		return;
	}
	if (ctx.config.injectionClassifier !== false) {
		const classification = await classifyInjection(
			ctx.injectionClassifierModel ??
				(() => {
					throw new Error("P2P injection classifier is not configured.");
				}),
			screened.canonicalQuery,
		);
		if (classification.injection) {
			log(ctx.logger, { ...baseLog, event: "request_rejected", code: "injection-detected" });
			sendRefusal(res, 400, "injection-detected", "The query was rejected by the inbound safety gate.");
			return;
		}
	}

	const observedSources = new Set<string>();
	let searchResponse: SearchDocumentsResponse;
	try {
		const queued = ctx.queue.enqueue(() =>
			ctx.agent.searchDocuments(screened.canonicalQuery, {
				topK: request.topK,
				scope: request.scope,
				peerFingerprint: headers.fingerprint,
				resolvePolicy: ctx.resolvePolicy,
				observedSources,
				searchTimeoutMs: ctx.config.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
			} as RetrievalOptions),
		);
		ctx.onQueueEnqueued?.(ctx.queue.pendingCount);
		searchResponse = await queued;
	} catch (error) {
		if (error instanceof QueryQueueFullError) {
			log(ctx.logger, { ...baseLog, event: "request_rejected", code: "queue-full" });
			sendRefusal(res, 503, "queue-full", "The peer query queue is full.");
			return;
		}
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "internal-error" });
		sendRefusal(res, 500, "internal-error", "The peer query could not be completed.");
		return;
	}

	try {
		const peerResponse = await ctx.buildPeerResponse({
			response: searchResponse,
			observedSources,
			resolvePolicy: ctx.resolvePolicy,
			peerFingerprint: headers.fingerprint,
			workspaceRoots: ctx.config.workspaceRoots ?? [],
			pseudonymize: ctx.config.pseudonymize ?? ctx.config.piiNer ?? false,
		});
		if (!Value.Check(PeerQueryResponseSchema, peerResponse))
			throw new Error("Egress response failed wire validation.");
		sendJson(res, 200, peerResponse);
	} catch {
		log(ctx.logger, { ...baseLog, event: "request_rejected", code: "internal-error" });
		sendRefusal(res, 500, "internal-error", "The peer response could not be built.");
	}
}

function resolveQuotas(configured: Partial<PolicyQuotas> | undefined, policy: PolicyQuotas | undefined): PolicyQuotas {
	return {
		queriesPerHour: configured?.queriesPerHour ?? policy?.queriesPerHour ?? 10,
		burst: configured?.burst ?? policy?.burst ?? 3,
		maxBodyBytes: configured?.maxBodyBytes ?? policy?.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES,
		maxFileBytes: configured?.maxFileBytes ?? policy?.maxFileBytes ?? 26_214_400,
	};
}

function parseSignedHeaders(req: IncomingMessage): ParsedHeaders | undefined {
	const fingerprint = singleHeader(req.headers["x-peer-fingerprint"]);
	const timestampText = singleHeader(req.headers["x-peer-timestamp"]);
	const signature = singleHeader(req.headers["x-peer-signature"]);
	if (
		fingerprint === undefined ||
		!FINGERPRINT_PATTERN.test(fingerprint) ||
		timestampText === undefined ||
		!TIMESTAMP_PATTERN.test(timestampText) ||
		signature === undefined ||
		signature.length === 0
	) {
		return undefined;
	}
	const timestamp = Number(timestampText);
	if (!Number.isSafeInteger(timestamp)) return undefined;
	return { fingerprint: fingerprint.toLowerCase(), timestamp, signature };
}

function singleHeader(value: string | string[] | undefined): string | undefined {
	if (typeof value === "string") return value;
	if (Array.isArray(value) && value.length === 1) return value[0];
	return undefined;
}

function findPeer(registry: PeerRegistry, fingerprint: string): PeerRecord | undefined {
	for (const peer of Object.values(registry)) {
		if (peer.fingerprint.toLowerCase() === fingerprint.toLowerCase()) return peer;
	}
	return undefined;
}

function isReplay(peer: PeerRecord, timestamp: number): boolean {
	return peer.highWaterTimestamp !== undefined && timestamp <= peer.highWaterTimestamp;
}

function parseQueryRequest(value: unknown): PeerQueryRequest | undefined {
	return Value.Check(PeerQueryRequestSchema, value) ? (value as PeerQueryRequest) : undefined;
}

function queryHashFromBody(value: unknown): string | undefined {
	if (!isRecord(value) || typeof value.query !== "string") return undefined;
	return createHash("sha256").update(value.query).digest("hex");
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function consumeToken(buckets: Map<string, TokenBucket>, fingerprint: string, quotas: PolicyQuotas): boolean {
	const now = Date.now();
	const refillPerMs = quotas.queriesPerHour / (60 * 60 * 1000);
	const bucket = buckets.get(fingerprint);
	if (bucket === undefined) {
		buckets.set(fingerprint, { tokens: Math.max(0, quotas.burst - 1), lastRefillMs: now });
		return quotas.burst >= 1;
	}
	bucket.tokens = Math.min(quotas.burst, bucket.tokens + (now - bucket.lastRefillMs) * refillPerMs);
	bucket.lastRefillMs = now;
	if (bucket.tokens < 1) return false;
	bucket.tokens -= 1;
	return true;
}

function log(logger: (entry: P2pRequestLog) => void, entry: P2pRequestLog): void {
	try {
		logger(entry);
	} catch {
		// Logging must not change peer request behavior.
	}
}

async function readBody(req: IncomingMessage, maxBytes: number): Promise<ReadBodyResult | ReadBodyFailure> {
	if (!Number.isSafeInteger(maxBytes) || maxBytes < 1) return { ok: false, reason: "too-large" };
	return new Promise((resolve) => {
		const chunks: Buffer[] = [];
		let length = 0;
		let settled = false;
		const finish = (result: ReadBodyResult | ReadBodyFailure): void => {
			if (settled) return;
			settled = true;
			resolve(result);
		};
		req.on("data", (chunk: Buffer | string) => {
			if (settled) return;
			const bytes = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
			length += bytes.length;
			if (length > maxBytes) {
				finish({ ok: false, reason: "too-large" });
				req.resume();
				return;
			}
			chunks.push(bytes);
		});
		req.on("end", () => finish({ ok: true, body: Buffer.concat(chunks) }));
		req.on("aborted", () => finish({ ok: false, reason: "aborted" }));
		req.on("error", () => finish({ ok: false, reason: "aborted" }));
	});
}

function sendRefusal(res: ServerResponse, status: number, code: string, message: string): void {
	sendJson(res, status, {
		v: 1,
		status: "rejected",
		answer: "",
		results: [],
		files: [],
		diagnostics: [{ code, message }],
	});
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
	if (res.headersSent) return;
	const encoded = JSON.stringify(body);
	res.statusCode = status;
	res.setHeader("content-type", "application/json; charset=utf-8");
	res.setHeader("content-length", Buffer.byteLength(encoded));
	res.end(encoded);
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

function closeServer(server: Server): Promise<void> {
	return new Promise((resolve, reject) => {
		server.close((error) => {
			if (error && (error as NodeJS.ErrnoException).code !== "ERR_SERVER_NOT_RUNNING") reject(error);
			else resolve();
		});
	});
}

function formatHost(host: string): string {
	return host.includes(":") && !host.startsWith("[") ? `[${host}]` : host;
}
