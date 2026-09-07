import { randomUUID } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type Static, Type } from "typebox";
import { Value } from "typebox/value";
import type { SearchDocumentsResponse } from "../agent/search-documents.ts";
import type { RetrievalOptions } from "../retrieval/types.ts";
import { buildPeerResponse as defaultBuildPeerResponse } from "./egress-gate.ts";
import { classifyInjection, type InjectionClassifierModel } from "./injection-classifier.ts";
import { screenInboundQuery } from "./injection-gate.ts";
import { type PolicyQuotas, type PolicyResolution, PolicyStore } from "./policy.ts";
import type { PolicyResolver } from "./policy-filter.ts";
import type { SignalIncomingMessage, SignalTransport } from "./signal-transport.ts";
import {
	type PeerQueryRequest,
	PeerQueryRequestSchema,
	type PeerQueryResponse,
	PeerQueryResponseSchema,
} from "./wire.ts";

/**
 * Peer query server carried over Signal messages instead of the retired
 * node:http transport. All security gates (L0/L1 injection, policy quotas,
 * deterministic egress) are transport-independent and run exactly as before;
 * only the carrier changed.
 */

const DEFAULT_QUEUE_DEPTH = 8;
const DEFAULT_SEARCH_TIMEOUT_MS = 120_000;
const DEFAULT_QUERY_TIMEOUT_MS = 120_000;
/**
 * Conservative cap for one Signal text message carrying a wire envelope.
 * Signal's inline body limit is 2,048 UTF-8 bytes; the adapter chunks the
 * envelope below that and refuses oversize payloads rather than truncating.
 * File bytes move as Signal attachments in milestone 3.
 */
export const MAX_SIGNAL_WIRE_BYTES = 1_800;

/**
 * Maximum UTF-8 bytes per Signal text chunk. Signal's inline body limit is
 * 2,048 bytes (SignalServiceMessageLimits.MAX_INLINE_BODY_SIZE_BYTES); this
 * leaves headroom for multi-byte runes never splitting mid-chunk.
 */
export const SIGNAL_TEXT_CHUNK_BYTES = 1_900;

/** Split a wire payload into Signal-safe UTF-8 chunks, never splitting a rune. */
export function chunkSignalText(text: string, maxBytes = SIGNAL_TEXT_CHUNK_BYTES): string[] {
	const chunks: string[] = [];
	let current = "";
	let currentBytes = 0;
	for (const char of text) {
		const charBytes = Buffer.byteLength(char, "utf8");
		if (currentBytes + charBytes > maxBytes && current.length > 0) {
			chunks.push(current);
			current = "";
			currentBytes = 0;
		}
		current += char;
		currentBytes += charBytes;
	}
	if (current.length > 0) chunks.push(current);
	return chunks;
}

const PRIVATE_POLICY: PolicyResolution = {
	tier: "private",
	allowed: false,
	shareBytes: false,
	redact: true,
};

// ---------------------------------------------------------------------------
// Wire envelope: wire.ts shapes carried as Signal message payloads
// ---------------------------------------------------------------------------

export const SignalWireEnvelopeSchema = Type.Object({
	v: Type.Literal(1, { description: "Envelope version" }),
	kind: Type.Union([Type.Literal("query"), Type.Literal("response")], { description: "Payload kind" }),
	id: Type.String({ minLength: 1, maxLength: 128, description: "Correlation id" }),
	payload: Type.Unknown({ description: "PeerQueryRequest or PeerQueryResponse" }),
});

export type SignalWireEnvelope = Static<typeof SignalWireEnvelopeSchema>;

// ---------------------------------------------------------------------------
// Peer registry: Signal phone-number/UUID keyed by local alias
// ---------------------------------------------------------------------------

export interface SignalPeerRecord {
	/** E.164 phone number or Signal UUID (ACI) of the peer. */
	readonly signalId: string;
	readonly addedAt: string;
}

export type SignalPeerRegistry = Record<string, SignalPeerRecord>;

const SIGNAL_PEERS_FILENAME = join(".autorag", "p2p", "signal-peers.json");

/** Load the Signal peer registry; a missing file means no trusted peers. */
export function loadSignalPeerRegistry(workspacePath: string): SignalPeerRegistry {
	const path = join(workspacePath, SIGNAL_PEERS_FILENAME);
	if (!existsSync(path)) return {};
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
		const registry: SignalPeerRegistry = {};
		for (const [alias, record] of Object.entries(parsed as Record<string, unknown>)) {
			if (
				typeof record === "object" &&
				record !== null &&
				typeof (record as Record<string, unknown>).signalId === "string" &&
				typeof (record as Record<string, unknown>).addedAt === "string"
			) {
				registry[alias] = record as SignalPeerRecord;
			}
		}
		return registry;
	} catch {
		return {};
	}
}

// ---------------------------------------------------------------------------
// Agent + response-builder contracts (moved from the retired http server)
// ---------------------------------------------------------------------------

export interface P2pSearchAgent {
	/** Must be true; startSignalPeerServer validates this before serving. */
	readonly remoteSession: boolean;
	searchDocuments(query: string, options?: RetrievalOptions): Promise<SearchDocumentsResponse>;
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

export interface SignalPeerLog {
	readonly event: "request" | "request_rejected";
	readonly peerSignalId?: string;
	readonly code?: string;
}

export interface StartSignalPeerServerOptions {
	readonly transport: SignalTransport;
	readonly agent: P2pSearchAgent;
	readonly peers?: SignalPeerRegistry;
	readonly workspacePath?: string;
	readonly workspaceRoots?: readonly string[];
	readonly policyStore?: {
		readonly quotas?: PolicyQuotas;
		resolvePolicy(source: string, peerFingerprint?: string): PolicyResolution;
	};
	readonly resolvePolicy?: PolicyResolver;
	readonly quotas?: Partial<PolicyQuotas>;
	readonly queueDepth?: number;
	readonly injectionClassifier?: boolean;
	readonly injectionClassifierModel?: InjectionClassifierModel;
	readonly buildPeerResponse?: PeerResponseBuilder;
	readonly searchTimeoutMs?: number;
	readonly pseudonymize?: boolean;
	readonly logger?: (entry: SignalPeerLog) => void;
}

export interface SignalPeerServer {
	close(): Promise<void>;
}

type TokenBucket = {
	tokens: number;
	lastRefillMs: number;
};

function consumeToken(buckets: Map<string, TokenBucket>, peerId: string, quotas: PolicyQuotas): boolean {
	const now = Date.now();
	const refillPerMs = quotas.queriesPerHour / (60 * 60 * 1000);
	const bucket = buckets.get(peerId);
	if (bucket === undefined) {
		buckets.set(peerId, { tokens: Math.max(0, quotas.burst - 1), lastRefillMs: now });
		return quotas.burst >= 1;
	}
	bucket.tokens = Math.min(quotas.burst, bucket.tokens + (now - bucket.lastRefillMs) * refillPerMs);
	bucket.lastRefillMs = now;
	if (bucket.tokens < 1) return false;
	bucket.tokens -= 1;
	return true;
}

function resolveQuotas(configured: Partial<PolicyQuotas> | undefined, policy: PolicyQuotas | undefined): PolicyQuotas {
	return {
		queriesPerHour: configured?.queriesPerHour ?? policy?.queriesPerHour ?? 10,
		burst: configured?.burst ?? policy?.burst ?? 3,
		maxBodyBytes: configured?.maxBodyBytes ?? policy?.maxBodyBytes ?? 65_536,
		maxFileBytes: configured?.maxFileBytes ?? policy?.maxFileBytes ?? 26_214_400,
	};
}

function rejection(id: string, code: string, message: string): string {
	const payload: PeerQueryResponse = {
		v: 1,
		status: "rejected",
		answer: "",
		results: [],
		files: [],
		diagnostics: [{ code, message }],
	};
	return JSON.stringify({ v: 1, kind: "response", id, payload });
}

function findPeer(registry: SignalPeerRegistry, message: SignalIncomingMessage): string | undefined {
	const candidates = [message.sourceUuid, message.source].filter(
		(candidate): candidate is string => typeof candidate === "string",
	);
	for (const record of Object.values(registry)) {
		if (candidates.includes(record.signalId)) return record.signalId;
	}
	return undefined;
}

/**
 * Bounded single-flight serializer: peer queries run one at a time to
 * preserve the AutoRAGAgent.searchDocuments() single-flight invariant, with a
 * typed queue-full refusal when the backlog is full.
 */
class QuerySerializer {
	private chain: Promise<unknown> = Promise.resolve();
	private waiting = 0;
	private readonly depth: number;

	constructor(depth: number) {
		this.depth = depth;
	}

	get pendingCount(): number {
		return this.waiting;
	}

	enqueue<T>(task: () => Promise<T>): Promise<T> {
		if (this.waiting >= this.depth) return Promise.reject(new QueueFullError());
		this.waiting += 1;
		const result = this.chain.then(task);
		this.chain = result.catch(() => {});
		this.chain.finally(() => {
			this.waiting -= 1;
		});
		return result;
	}
}

export class QueueFullError extends Error {
	constructor(message = "The peer query queue is full.") {
		super(message);
		this.name = "QueueFullError";
	}
}

export async function startSignalPeerServer(options: StartSignalPeerServerOptions): Promise<SignalPeerServer> {
	if (!options.agent || typeof options.agent.searchDocuments !== "function") {
		throw new TypeError("A public agent.searchDocuments implementation is required.");
	}
	if (options.agent.remoteSession !== true) {
		throw new TypeError("Signal peer server requires an agent constructed with remoteSession: true.");
	}
	const injectionClassifier = options.injectionClassifier ?? true;
	if (injectionClassifier && typeof options.injectionClassifierModel !== "function") {
		throw new TypeError("P2P injection classifier is enabled but no classifier model was provided.");
	}
	const buildPeerResponse = options.buildPeerResponse ?? defaultBuildPeerResponse;
	const policyStore =
		options.policyStore ?? (options.workspacePath ? new PolicyStore(options.workspacePath) : undefined);
	const quotas = resolveQuotas(options.quotas, policyStore?.quotas);
	const resolvePolicy =
		options.resolvePolicy ?? (policyStore ? policyStore.resolvePolicy.bind(policyStore) : () => PRIVATE_POLICY);
	const peers = options.peers ?? (options.workspacePath ? loadSignalPeerRegistry(options.workspacePath) : {});
	const workspaceRoots = options.workspaceRoots ?? (options.workspacePath ? [options.workspacePath] : []);
	const logger = options.logger ?? ((entry: SignalPeerLog) => console.info(JSON.stringify(entry)));
	const buckets = new Map<string, TokenBucket>();
	const serializer = new QuerySerializer(options.queueDepth ?? DEFAULT_QUEUE_DEPTH);
	const transport = options.transport;

	const handle = async (incoming: SignalIncomingMessage): Promise<void> => {
		let envelope: SignalWireEnvelope | undefined;
		try {
			const parsed = JSON.parse(incoming.message) as unknown;
			if (Value.Check(SignalWireEnvelopeSchema, parsed)) envelope = parsed;
		} catch {
			envelope = undefined;
		}
		if (envelope === undefined) {
			await transport.sendMessage(
				incoming.source,
				rejection("unknown", "internal-error", "Message is not a valid peer wire envelope."),
			);
			log(logger, { event: "request_rejected", code: "internal-error" });
			return;
		}
		if (envelope.kind !== "query") return; // responses are consumed by querySignalPeer

		const peerSignalId = findPeer(peers, incoming);
		if (peerSignalId === undefined) {
			await transport.sendMessage(
				incoming.source,
				rejection(envelope.id, "auth-error", "Peer authentication failed."),
			);
			log(logger, { event: "request_rejected", code: "auth-error" });
			return;
		}
		if (!consumeToken(buckets, peerSignalId, quotas)) {
			await transport.sendMessage(
				incoming.source,
				rejection(envelope.id, "rate-limited", "Peer query quota exceeded."),
			);
			log(logger, { event: "request_rejected", peerSignalId, code: "rate-limited" });
			return;
		}
		if (!Value.Check(PeerQueryRequestSchema, envelope.payload)) {
			await transport.sendMessage(
				incoming.source,
				rejection(envelope.id, "internal-error", "Request payload is not a valid peer query."),
			);
			log(logger, { event: "request_rejected", peerSignalId, code: "internal-error" });
			return;
		}
		const request = envelope.payload as PeerQueryRequest;
		const screened = screenInboundQuery(request.query);
		if (!screened.ok) {
			await transport.sendMessage(
				incoming.source,
				rejection(envelope.id, "injection-detected", "The query was rejected by the inbound safety gate."),
			);
			log(logger, { event: "request_rejected", peerSignalId, code: "injection-detected" });
			return;
		}
		if (injectionClassifier) {
			const classification = await classifyInjection(options.injectionClassifierModel!, screened.canonicalQuery);
			if (classification.injection) {
				await transport.sendMessage(
					incoming.source,
					rejection(envelope.id, "injection-detected", "The query was rejected by the inbound safety gate."),
				);
				log(logger, { event: "request_rejected", peerSignalId, code: "injection-detected" });
				return;
			}
		}
		log(logger, { event: "request", peerSignalId });

		const observedSources = new Set<string>();
		let searchResponse: SearchDocumentsResponse;
		try {
			searchResponse = await serializer.enqueue(() =>
				options.agent.searchDocuments(screened.canonicalQuery, {
					topK: request.topK,
					scope: request.scope,
					peerFingerprint: peerSignalId,
					resolvePolicy,
					observedSources,
					searchTimeoutMs: options.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
				} as RetrievalOptions),
			);
		} catch (error) {
			const code = error instanceof QueueFullError ? "queue-full" : "internal-error";
			await transport.sendMessage(
				incoming.source,
				rejection(
					envelope.id,
					code,
					error instanceof QueueFullError
						? "The peer query queue is full."
						: "The peer query could not be completed.",
				),
			);
			log(logger, { event: "request_rejected", peerSignalId, code });
			return;
		}

		try {
			const peerResponse = await buildPeerResponse({
				response: searchResponse,
				observedSources,
				resolvePolicy,
				peerFingerprint: peerSignalId,
				workspaceRoots,
				pseudonymize: options.pseudonymize ?? false,
			});
			if (!Value.Check(PeerQueryResponseSchema, peerResponse)) {
				throw new Error("Egress response failed wire validation.");
			}
			const message = JSON.stringify({ v: 1, kind: "response", id: envelope.id, payload: peerResponse });
			if (Buffer.byteLength(message) > MAX_SIGNAL_WIRE_BYTES) {
				await transport.sendMessage(
					incoming.source,
					rejection(envelope.id, "internal-error", "The peer response exceeds the Signal message size limit."),
				);
				log(logger, { event: "request_rejected", peerSignalId, code: "internal-error" });
				return;
			}
			await transport.sendMessage(incoming.source, message);
		} catch {
			await transport.sendMessage(
				incoming.source,
				rejection(envelope.id, "internal-error", "The peer response could not be built."),
			);
			log(logger, { event: "request_rejected", peerSignalId, code: "internal-error" });
		}
	};

	transport.onMessage((incoming) => {
		void handle(incoming);
	});

	return {
		close: async () => {},
	};
}

function log(logger: (entry: SignalPeerLog) => void, entry: SignalPeerLog): void {
	try {
		logger(entry);
	} catch {
		// Logging must not change peer request behavior.
	}
}

export interface QuerySignalPeerOptions {
	readonly timeoutMs?: number;
}

/**
 * Client side of the Signal peer wire: send a PeerQueryRequest and await the
 * correlated PeerQueryResponse.
 */
export async function querySignalPeer(
	transport: SignalTransport,
	peerSignalId: string,
	request: PeerQueryRequest,
	options: QuerySignalPeerOptions = {},
): Promise<PeerQueryResponse> {
	const id = randomUUID();
	const timeoutMs = options.timeoutMs ?? DEFAULT_QUERY_TIMEOUT_MS;
	return new Promise<PeerQueryResponse>((resolve, reject) => {
		const handler = (incoming: SignalIncomingMessage): void => {
			let envelope: SignalWireEnvelope | undefined;
			try {
				const parsed = JSON.parse(incoming.message) as unknown;
				if (Value.Check(SignalWireEnvelopeSchema, parsed)) envelope = parsed;
			} catch {
				return;
			}
			if (envelope === undefined || envelope.kind !== "response" || envelope.id !== id) return;
			clearTimeout(timer);
			unsubscribe();
			if (Value.Check(PeerQueryResponseSchema, envelope.payload)) {
				resolve(envelope.payload as PeerQueryResponse);
			} else {
				reject(new Error("Signal peer response failed wire validation."));
			}
		};
		const unsubscribe = transport.onMessage(handler);
		const timer = setTimeout(() => {
			unsubscribe();
			reject(new Error(`Signal peer query timed out after ${timeoutMs}ms.`));
		}, timeoutMs);
		void transport
			.sendMessage(peerSignalId, JSON.stringify({ v: 1, kind: "query", id, payload: request }))
			.catch((error: unknown) => {
				clearTimeout(timer);
				unsubscribe();
				reject(error instanceof Error ? error : new Error(String(error)));
			});
	});
}
