import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, unlinkSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { Value } from "typebox/value";
import type { SimplexIncomingMessage, SimplexTransport } from "./simplex-transport.ts";
import {
	type PeerQueryRequest,
	PeerQueryRequestSchema,
	type PeerQueryResponse,
	PeerQueryResponseSchema,
} from "./wire.ts";

export const DEFAULT_SIMPLEX_QUERY_FAST_TIMEOUT_MS = 60_000;
export const SIMPLEX_QUERY_TTL_MS = 21 * 24 * 60 * 60 * 1000;

export type SimplexQueryStateStatus = "pending" | "completed" | "expired";

export interface SimplexQueryState {
	readonly id: string;
	readonly contactId: number;
	readonly request: PeerQueryRequest;
	readonly sessionId?: string;
	readonly createdAt: string;
	readonly expiresAt: string;
	readonly status: SimplexQueryStateStatus;
	readonly response?: PeerQueryResponse;
	readonly completedAt?: string;
	readonly expiredAt?: string;
	readonly diagnostic?: string;
}

const QUERY_DIR = join(".autorag", "p2p", "queries");
const PENDING_DIR = "pending";
const COMPLETED_DIR = "completed";
const EXPIRED_DIR = "expired";

function statePath(workspacePath: string, directory: string, id: string): string {
	return join(workspacePath, QUERY_DIR, directory, `${id}.json`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function atomicWrite(path: string, value: unknown): void {
	mkdirSync(dirname(path), { recursive: true });
	const tempPath = `${path}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
	writeFileSync(tempPath, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
	renameSync(tempPath, path);
}

function parseState(value: unknown): SimplexQueryState | undefined {
	if (!isRecord(value)) return undefined;
	if (typeof value.id !== "string" || value.id.length === 0) return undefined;
	if (typeof value.contactId !== "number" || !Number.isSafeInteger(value.contactId)) return undefined;
	if (!Value.Check(PeerQueryRequestSchema, value.request)) return undefined;
	if (value.sessionId !== undefined && typeof value.sessionId !== "string") return undefined;
	if (typeof value.createdAt !== "string" || typeof value.expiresAt !== "string") return undefined;
	if (value.status !== "pending" && value.status !== "completed" && value.status !== "expired") return undefined;
	if (value.response !== undefined && !Value.Check(PeerQueryResponseSchema, value.response)) return undefined;
	return value as unknown as SimplexQueryState;
}

function readState(path: string): SimplexQueryState | undefined {
	if (!existsSync(path)) return undefined;
	try {
		return parseState(JSON.parse(readFileSync(path, "utf8")) as unknown);
	} catch (error) {
		if (error instanceof SyntaxError) return undefined;
		throw error;
	}
}

function moveState(workspacePath: string, from: string, to: string, state: SimplexQueryState): boolean {
	const source = statePath(workspacePath, from, state.id);
	const destination = statePath(workspacePath, to, state.id);
	const claim = `${source}.claim`;
	try {
		renameSync(source, claim);
	} catch (error) {
		if (isNodeError(error) && error.code === "ENOENT") return false;
		throw error;
	}
	atomicWrite(destination, state);
	unlinkSync(claim);
	return true;
}

function recoverClaims(workspacePath: string): void {
	const directory = join(workspacePath, QUERY_DIR, PENDING_DIR);
	if (!existsSync(directory)) return;
	for (const name of readdirSync(directory)) {
		if (!name.endsWith(".json.claim")) continue;
		const claim = join(directory, name);
		const restored = join(directory, name.slice(0, -".claim".length));
		if (!existsSync(restored)) renameSync(claim, restored);
	}
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
	return error instanceof Error && "code" in error;
}

export function saveSimplexQueryState(workspacePath: string, state: SimplexQueryState): void {
	atomicWrite(statePath(workspacePath, PENDING_DIR, state.id), state);
}

export function loadSimplexQueryState(workspacePath: string, id: string): SimplexQueryState | undefined {
	for (const [directory, status] of [
		[COMPLETED_DIR, "completed"],
		[EXPIRED_DIR, "expired"],
		[PENDING_DIR, "pending"],
	] as const) {
		const state = readState(statePath(workspacePath, directory, id));
		if (state?.status === status) return state;
	}
	return undefined;
}

export function listPendingSimplexQueryStates(workspacePath: string): readonly SimplexQueryState[] {
	const directory = join(workspacePath, QUERY_DIR, PENDING_DIR);
	if (!existsSync(directory)) return [];
	return readdirSync(directory)
		.filter((name) => name.endsWith(".json"))
		.map((name) => readState(join(directory, name)))
		.filter((state): state is SimplexQueryState => state?.status === "pending")
		.filter((state) => loadSimplexQueryState(workspacePath, state.id)?.status === "pending")
		.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}

export function completeSimplexQueryState(
	workspacePath: string,
	id: string,
	response: PeerQueryResponse,
	completedAt: string,
): SimplexQueryState | undefined {
	const state = loadSimplexQueryState(workspacePath, id);
	if (state?.status !== "pending") return undefined;
	const completed: SimplexQueryState = { ...state, status: "completed", response, completedAt };
	return moveState(workspacePath, PENDING_DIR, COMPLETED_DIR, completed) ? completed : undefined;
}

export function expireSimplexQueryState(
	workspacePath: string,
	id: string,
	expiredAt: string,
	diagnostic: string,
): SimplexQueryState | undefined {
	const state = loadSimplexQueryState(workspacePath, id);
	if (state?.status !== "pending") return undefined;
	const expired: SimplexQueryState = { ...state, status: "expired", expiredAt, diagnostic };
	return moveState(workspacePath, PENDING_DIR, EXPIRED_DIR, expired) ? expired : undefined;
}

export function removeSimplexQueryState(workspacePath: string, id: string): void {
	for (const directory of [PENDING_DIR, COMPLETED_DIR, EXPIRED_DIR]) {
		const path = statePath(workspacePath, directory, id);
		if (existsSync(path)) unlinkSync(path);
	}
}

export type SimplexQueryResult =
	| { readonly status: "completed"; readonly id: string; readonly response: PeerQueryResponse }
	| { readonly status: "pending"; readonly id: string; readonly expiresAt: string };

export interface SimplexQueryClientOptions {
	readonly fastTimeoutMs?: number;
	readonly now?: () => Date;
	readonly onResponse?: (state: SimplexQueryState, response: PeerQueryResponse) => void;
	readonly onExpired?: (state: SimplexQueryState) => void;
}

function responseOf(message: SimplexIncomingMessage): { id: string; response: PeerQueryResponse } | undefined {
	let parsed: unknown;
	try {
		parsed = JSON.parse(message.text) as unknown;
	} catch (error) {
		if (error instanceof SyntaxError) return undefined;
		throw error;
	}
	if (!isRecord(parsed) || parsed.kind !== "response" || typeof parsed.id !== "string") return undefined;
	if (!Value.Check(PeerQueryResponseSchema, parsed.payload)) return undefined;
	return { id: parsed.id, response: parsed.payload as PeerQueryResponse };
}

export interface SimplexQueryClient {
	send(contactId: number, request: PeerQueryRequest, sessionId?: string): Promise<SimplexQueryResult>;
	close(): void;
}

export function createSimplexQueryClient(
	transport: SimplexTransport,
	workspacePath: string,
	options: SimplexQueryClientOptions = {},
): SimplexQueryClient {
	const now = options.now ?? (() => new Date());
	const fastTimeoutMs = options.fastTimeoutMs ?? DEFAULT_SIMPLEX_QUERY_FAST_TIMEOUT_MS;
	const waiters = new Map<string, (result: SimplexQueryResult) => void>();
	const timers = new Map<string, ReturnType<typeof setTimeout>>();

	const expire = (state: SimplexQueryState): void => {
		const current = loadSimplexQueryState(workspacePath, state.id);
		if (current?.status !== "pending") return;
		const expired = expireSimplexQueryState(
			workspacePath,
			state.id,
			now().toISOString(),
			`SimpleX peer query expired after ${SIMPLEX_QUERY_TTL_MS}ms.`,
		);
		if (expired !== undefined) void options.onExpired?.(expired);
	};

	const scheduleExpiry = (state: SimplexQueryState): void => {
		const delay = Math.max(0, Date.parse(state.expiresAt) - now().getTime());
		const timer = setTimeout(() => expire(state), delay);
		timer.unref?.();
		timers.set(state.id, timer);
	};

	const onMessage = (message: SimplexIncomingMessage): void => {
		const incoming = responseOf(message);
		if (incoming === undefined) return;
		const state = loadSimplexQueryState(workspacePath, incoming.id);
		if (state?.status !== "pending") return;
		const completed = completeSimplexQueryState(workspacePath, incoming.id, incoming.response, now().toISOString());
		if (completed === undefined) return;
		const timer = timers.get(incoming.id);
		if (timer !== undefined) clearTimeout(timer);
		timers.delete(incoming.id);
		const waiter = waiters.get(incoming.id);
		if (waiter !== undefined) {
			waiters.delete(incoming.id);
			waiter({ status: "completed", id: incoming.id, response: incoming.response });
			return;
		}
		void options.onResponse?.(completed, incoming.response);
	};

	const unsubscribe = transport.onMessage(onMessage);
	recoverClaims(workspacePath);
	for (const state of listPendingSimplexQueryStates(workspacePath)) {
		if (Date.parse(state.expiresAt) <= now().getTime()) expire(state);
		else scheduleExpiry(state);
	}

	return {
		async send(contactId, request, sessionId) {
			const createdAt = now();
			const id = randomUUID();
			const state: SimplexQueryState = {
				id,
				contactId,
				request,
				...(sessionId !== undefined ? { sessionId } : {}),
				createdAt: createdAt.toISOString(),
				expiresAt: new Date(createdAt.getTime() + SIMPLEX_QUERY_TTL_MS).toISOString(),
				status: "pending",
			};
			saveSimplexQueryState(workspacePath, state);
			scheduleExpiry(state);
			const result = new Promise<SimplexQueryResult>((resolve) => {
				const timer = setTimeout(() => {
					waiters.delete(id);
					resolve({ status: "pending", id, expiresAt: state.expiresAt });
				}, fastTimeoutMs);
				waiters.set(id, (result) => {
					clearTimeout(timer);
					resolve(result);
				});
			});
			try {
				await transport.sendMessage(contactId, JSON.stringify({ v: 1, kind: "query", id, payload: request }));
			} catch (error) {
				const current = loadSimplexQueryState(workspacePath, id);
				if (current?.status === "pending") {
					expireSimplexQueryState(
						workspacePath,
						id,
						now().toISOString(),
						`SimpleX peer query could not be sent: ${error instanceof Error ? error.message : String(error)}`,
					);
				}
				throw error;
			}
			const completed = loadSimplexQueryState(workspacePath, id);
			if (completed?.status === "completed" && completed.response !== undefined) {
				const waiter = waiters.get(id);
				if (waiter !== undefined) {
					waiters.delete(id);
					waiter({ status: "completed", id, response: completed.response });
				}
			}
			return result;
		},
		close() {
			unsubscribe();
			for (const timer of timers.values()) clearTimeout(timer);
			timers.clear();
			waiters.clear();
		},
	};
}
