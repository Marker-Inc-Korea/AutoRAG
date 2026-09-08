import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { type Static, Type } from "typebox";
import { Value } from "typebox/value";
import type { SearchDocumentsResponse } from "../agent/search-documents.ts";
import type { RetrievalOptions } from "../retrieval/types.ts";
import {
	ApprovalAbortedError,
	ApprovalTimeoutError,
	savePendingPeerRequest,
	waitForPeerRequestDecision,
} from "./approval-store.ts";
import { buildPeerResponse as defaultBuildPeerResponse } from "./egress-gate.ts";
import { classifyInjection, type InjectionClassifierModel } from "./injection-classifier.ts";
import { screenInboundQuery } from "./injection-gate.ts";
import { type PolicyQuotas, type PolicyResolution, PolicyStore } from "./policy.ts";
import type { PolicyResolver } from "./policy-filter.ts";
import type { SimplexIncomingMessage, SimplexTransport } from "./simplex-transport.ts";
import {
	type PeerQueryRequest,
	PeerQueryRequestSchema,
	type PeerQueryResponse,
	PeerQueryResponseSchema,
} from "./wire.ts";

/**
 * Peer query server carried over SimpleX Chat messages. All security gates
 * (L0/L1 injection, policy quotas, deterministic egress) are transport-
 * independent and run exactly as before; only the carrier changed.
 */

const DEFAULT_QUEUE_DEPTH = 8;
const DEFAULT_SEARCH_TIMEOUT_MS = 120_000;
const DEFAULT_QUERY_TIMEOUT_MS = 120_000;
/**
 * SimpleX message bodies have no documented byte cap, but the wire envelope
 * stays bounded so a malformed or hostile payload cannot exhaust the peer.
 */
export const MAX_SIMPLEX_WIRE_BYTES = 262_144;

const PRIVATE_POLICY: PolicyResolution = {
	tier: "private",
	allowed: false,
	shareBytes: false,
	redact: true,
};

// ---------------------------------------------------------------------------
// Wire envelope: wire.ts shapes carried as SimpleX text messages
// ---------------------------------------------------------------------------

export const SimplexWireEnvelopeSchema = Type.Object({
	v: Type.Literal(1, { description: "Envelope version" }),
	kind: Type.Union([Type.Literal("query"), Type.Literal("response")], { description: "Payload kind" }),
	id: Type.String({ minLength: 1, maxLength: 128, description: "Correlation id" }),
	payload: Type.Unknown({ description: "PeerQueryRequest or PeerQueryResponse" }),
});

export type SimplexWireEnvelope = Static<typeof SimplexWireEnvelopeSchema>;

// ---------------------------------------------------------------------------
// Peer registry: SimpleX contactId keyed by local alias
// ---------------------------------------------------------------------------

export interface SimplexPeerRecord {
	/** SimpleX contact id of the peer (stable per profile). */
	readonly contactId: number;
	readonly addedAt: string;
}

export type SimplexPeerRegistry = Record<string, SimplexPeerRecord>;

const PEERS_DIR = join(".autorag", "p2p");
const PEERS_FILENAME = "simplex-peers.json";

/** Load the SimpleX peer registry; a missing file means no trusted peers. */
export function loadSimplexPeerRegistry(workspacePath: string): SimplexPeerRegistry {
	const path = join(workspacePath, PEERS_DIR, PEERS_FILENAME);
	if (!existsSync(path)) return {};
	try {
		const parsed = JSON.parse(readFileSync(path, "utf8")) as unknown;
		if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return {};
		const registry: SimplexPeerRegistry = {};
		for (const [alias, record] of Object.entries(parsed as Record<string, unknown>)) {
			if (
				typeof record === "object" &&
				record !== null &&
				typeof (record as Record<string, unknown>).contactId === "number" &&
				typeof (record as Record<string, unknown>).addedAt === "string"
			) {
				registry[alias] = record as SimplexPeerRecord;
			}
		}
		return registry;
	} catch {
		return {};
	}
}

/** Persist the SimpleX peer registry. */
export function saveSimplexPeerRegistry(workspacePath: string, registry: SimplexPeerRegistry): void {
	const dir = join(workspacePath, PEERS_DIR);
	mkdirSync(dir, { recursive: true });
	writeFileSync(join(dir, PEERS_FILENAME), JSON.stringify(registry, null, 2), { mode: 0o600 });
}

// ---------------------------------------------------------------------------
// Agent + response-builder contracts
// ---------------------------------------------------------------------------

export interface P2pSearchAgent {
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

export interface SimplexPeerLog {
	readonly event: "request" | "request_rejected";
	readonly peerContactId?: number;
	readonly code?: string;
}

export interface StartSimplexPeerServerOptions {
	readonly transport: SimplexTransport;
	readonly agent: P2pSearchAgent;
	readonly peers?: SimplexPeerRegistry;
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
	readonly logger?: (entry: SimplexPeerLog) => void;
}

export interface SimplexPeerServer {
	close(): Promise<void>;
}

type TokenBucket = {
	tokens: number;
	lastRefillMs: number;
};

function consumeToken(buckets: Map<number, TokenBucket>, contactId: number, quotas: PolicyQuotas): boolean {
	const now = Date.now();
	const refillPerMs = quotas.queriesPerHour / (60 * 60 * 1000);
	const bucket = buckets.get(contactId);
	if (bucket === undefined) {
		buckets.set(contactId, { tokens: Math.max(0, quotas.burst - 1), lastRefillMs: now });
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
	return JSON.stringify({ v: 1, kind: "response", id, payload } satisfies SimplexWireEnvelope);
}

function findPeer(registry: SimplexPeerRegistry, message: SimplexIncomingMessage): number | undefined {
	for (const record of Object.values(registry)) {
		if (record.contactId === message.contactId) return record.contactId;
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
		void this.chain.finally(() => {
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

export async function startSimplexPeerServer(options: StartSimplexPeerServerOptions): Promise<SimplexPeerServer> {
	if (!options.agent || typeof options.agent.searchDocuments !== "function") {
		throw new TypeError("A public agent.searchDocuments implementation is required.");
	}
	if (options.agent.remoteSession !== true) {
		throw new TypeError("SimpleX peer server requires an agent constructed with remoteSession: true.");
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
	const peers = options.peers ?? (options.workspacePath ? loadSimplexPeerRegistry(options.workspacePath) : {});
	const workspaceRoots = options.workspaceRoots ?? (options.workspacePath ? [options.workspacePath] : []);
	const logger = options.logger ?? ((entry: SimplexPeerLog) => console.info(JSON.stringify(entry)));
	const buckets = new Map<number, TokenBucket>();
	const serializer = new QuerySerializer(options.queueDepth ?? DEFAULT_QUEUE_DEPTH);
	const transport = options.transport;
	const abort = new AbortController();

	const handle = async (incoming: SimplexIncomingMessage): Promise<void> => {
		let envelope: SimplexWireEnvelope | undefined;
		try {
			const parsed = JSON.parse(incoming.text) as unknown;
			if (Value.Check(SimplexWireEnvelopeSchema, parsed)) envelope = parsed;
		} catch {
			envelope = undefined;
		}
		if (envelope === undefined) {
			await transport.sendMessage(
				incoming.contactId,
				rejection("unknown", "internal-error", "Message is not a valid peer wire envelope."),
			);
			log(logger, { event: "request_rejected", code: "internal-error" });
			return;
		}
		if (envelope.kind !== "query") return; // responses are consumed by querySimplexPeer

		const peerContactId = findPeer(peers, incoming);
		if (peerContactId === undefined) {
			await transport.sendMessage(
				incoming.contactId,
				rejection(envelope.id, "auth-error", "Peer authentication failed."),
			);
			log(logger, { event: "request_rejected", code: "auth-error" });
			return;
		}
		if (!consumeToken(buckets, peerContactId, quotas)) {
			await transport.sendMessage(
				incoming.contactId,
				rejection(envelope.id, "rate-limited", "Peer query quota exceeded."),
			);
			log(logger, { event: "request_rejected", peerContactId, code: "rate-limited" });
			return;
		}
		if (!Value.Check(PeerQueryRequestSchema, envelope.payload)) {
			await transport.sendMessage(
				incoming.contactId,
				rejection(envelope.id, "internal-error", "Request payload is not a valid peer query."),
			);
			log(logger, { event: "request_rejected", peerContactId, code: "internal-error" });
			return;
		}
		const request = envelope.payload as PeerQueryRequest;
		const screened = screenInboundQuery(request.query);
		if (!screened.ok) {
			await transport.sendMessage(
				incoming.contactId,
				rejection(envelope.id, "injection-detected", "The query was rejected by the inbound safety gate."),
			);
			log(logger, { event: "request_rejected", peerContactId, code: "injection-detected" });
			return;
		}
		if (injectionClassifier) {
			const classification = await classifyInjection(options.injectionClassifierModel!, screened.canonicalQuery);
			if (classification.injection) {
				await transport.sendMessage(
					incoming.contactId,
					rejection(envelope.id, "injection-detected", "The query was rejected by the inbound safety gate."),
				);
				log(logger, { event: "request_rejected", peerContactId, code: "injection-detected" });
				return;
			}
		}
		log(logger, { event: "request", peerContactId });

		const observedSources = new Set<string>();
		let searchResponse: SearchDocumentsResponse;
		try {
			searchResponse = await serializer.enqueue(() =>
				options.agent.searchDocuments(screened.canonicalQuery, {
					topK: request.topK,
					scope: request.scope,
					peerFingerprint: String(peerContactId),
					resolvePolicy,
					observedSources,
					searchTimeoutMs: options.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
				} as RetrievalOptions),
			);
		} catch (error) {
			const code = error instanceof QueueFullError ? "queue-full" : "internal-error";
			const detail = error instanceof Error ? error.message.slice(0, 300) : String(error).slice(0, 300);
			await transport.sendMessage(
				incoming.contactId,
				rejection(
					envelope.id,
					code,
					error instanceof QueueFullError
						? "The peer query queue is full."
						: `The peer query could not be completed: ${detail}`,
				),
			);
			log(logger, { event: "request_rejected", peerContactId, code });
			return;
		}

		try {
			const peerResponse = await buildPeerResponse({
				response: searchResponse,
				observedSources,
				resolvePolicy,
				peerFingerprint: String(peerContactId),
				workspaceRoots,
				pseudonymize: options.pseudonymize ?? false,
			});
			if (!Value.Check(PeerQueryResponseSchema, peerResponse)) {
				throw new Error("Egress response failed wire validation.");
			}
			const message = JSON.stringify({ v: 1, kind: "response", id: envelope.id, payload: peerResponse });
			if (Buffer.byteLength(message) > MAX_SIMPLEX_WIRE_BYTES) {
				await transport.sendMessage(
					incoming.contactId,
					rejection(envelope.id, "internal-error", "The peer response exceeds the SimpleX message size limit."),
				);
				log(logger, { event: "request_rejected", peerContactId, code: "internal-error" });
				return;
			}
			if (peerResponse.status !== "ok") {
				await transport.sendMessage(incoming.contactId, message);
				return;
			}
			if (options.workspacePath === undefined) {
				await transport.sendMessage(
					incoming.contactId,
					rejection(envelope.id, "internal-error", "Peer query approval requires a workspace path."),
				);
				return;
			}
			savePendingPeerRequest(options.workspacePath, {
				id: envelope.id,
				contactId: incoming.contactId,
				query: screened.canonicalQuery,
				createdAt: new Date().toISOString(),
				sources: [...observedSources],
				payload: peerResponse,
			});
			try {
				const decision = await waitForPeerRequestDecision(options.workspacePath, envelope.id, {
					timeoutMs: options.searchTimeoutMs ?? DEFAULT_SEARCH_TIMEOUT_MS,
					abort: abort.signal,
				});
				if (decision.decision === "approve") {
					await transport.sendMessage(incoming.contactId, message);
				} else {
					await transport.sendMessage(
						incoming.contactId,
						rejection(envelope.id, "policy-denied", "The operator declined to share this response."),
					);
				}
			} catch (error) {
				if (error instanceof ApprovalAbortedError) return;
				const reason =
					error instanceof ApprovalTimeoutError
						? "The operator did not approve this response in time."
						: "Peer query approval failed.";
				await transport.sendMessage(incoming.contactId, rejection(envelope.id, "policy-denied", reason));
			}
		} catch {
			await transport.sendMessage(
				incoming.contactId,
				rejection(envelope.id, "internal-error", "The peer response could not be built."),
			);
			log(logger, { event: "request_rejected", peerContactId, code: "internal-error" });
		}
	};

	transport.onMessage((incoming) => {
		void handle(incoming);
	});

	return {
		close: async () => {
			abort.abort();
		},
	};
}

function log(logger: (entry: SimplexPeerLog) => void, entry: SimplexPeerLog): void {
	try {
		logger(entry);
	} catch {
		// Logging must not change peer request behavior.
	}
}

export interface QuerySimplexPeerOptions {
	readonly timeoutMs?: number;
}

/**
 * Client side of the SimpleX peer wire: send a PeerQueryRequest and await the
 * correlated PeerQueryResponse.
 */
export async function querySimplexPeer(
	transport: SimplexTransport,
	contactId: number,
	request: PeerQueryRequest,
	options: QuerySimplexPeerOptions = {},
): Promise<PeerQueryResponse> {
	const id = randomUUID();
	const timeoutMs = options.timeoutMs ?? DEFAULT_QUERY_TIMEOUT_MS;
	return new Promise<PeerQueryResponse>((resolve, reject) => {
		const handler = (incoming: SimplexIncomingMessage): void => {
			let envelope: SimplexWireEnvelope | undefined;
			try {
				const parsed = JSON.parse(incoming.text) as unknown;
				if (Value.Check(SimplexWireEnvelopeSchema, parsed)) envelope = parsed;
			} catch {
				return;
			}
			if (envelope === undefined || envelope.kind !== "response" || envelope.id !== id) return;
			clearTimeout(timer);
			unsubscribe();
			if (Value.Check(PeerQueryResponseSchema, envelope.payload)) {
				resolve(envelope.payload as PeerQueryResponse);
			} else {
				reject(new Error("SimpleX peer response failed wire validation."));
			}
		};
		const unsubscribe = transport.onMessage(handler);
		const timer = setTimeout(() => {
			unsubscribe();
			reject(new Error(`SimpleX peer query timed out after ${timeoutMs}ms.`));
		}, timeoutMs);
		void transport
			.sendMessage(contactId, JSON.stringify({ v: 1, kind: "query", id, payload: request }))
			.catch((error: unknown) => {
				clearTimeout(timer);
				unsubscribe();
				reject(error instanceof Error ? error : new Error(String(error)));
			});
	});
}
