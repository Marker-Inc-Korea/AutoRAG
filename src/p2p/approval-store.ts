import {
	existsSync,
	mkdirSync,
	readdirSync,
	readFileSync,
	renameSync,
	unlinkSync,
	watch,
	writeFileSync,
} from "node:fs";
import { dirname, join } from "node:path";
import type { PeerQueryResponse } from "./wire.ts";

export type PeerRequestDecisionKind = "approve" | "deny";

export interface PendingPeerRequest {
	readonly id: string;
	readonly contactId: number;
	readonly query: string;
	readonly createdAt: string;
	readonly sources: readonly string[];
	readonly payload: PeerQueryResponse;
}

export interface PeerRequestDecision {
	readonly id: string;
	readonly decision: PeerRequestDecisionKind;
	readonly decidedAt: string;
}

export class ApprovalTimeoutError extends Error {
	constructor(id: string) {
		super(`Peer request ${id} was not approved in time.`);
		this.name = "ApprovalTimeoutError";
	}
}

export class ApprovalAbortedError extends Error {
	constructor(id: string) {
		super(`Peer request ${id} was aborted before an operator decision.`);
		this.name = "ApprovalAbortedError";
	}
}

const P2P_DIR = join(".autorag", "p2p");

function requestsDir(workspacePath: string): string {
	return join(workspacePath, P2P_DIR, "requests");
}

function decisionsDir(workspacePath: string): string {
	return join(workspacePath, P2P_DIR, "decisions");
}

function requestPath(workspacePath: string, id: string): string {
	return join(requestsDir(workspacePath), `${id}.json`);
}

function decisionPath(workspacePath: string, id: string): string {
	return join(decisionsDir(workspacePath), `${id}.json`);
}

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

function atomicWrite(path: string, value: unknown): void {
	mkdirSync(dirname(path), { recursive: true });
	const tempPath = `${path}.${process.pid}.tmp`;
	writeFileSync(tempPath, `${JSON.stringify(value, null, 2)}\n`, { mode: 0o600 });
	renameSync(tempPath, path);
}

function parsePending(value: unknown): PendingPeerRequest | undefined {
	if (!isRecord(value)) return undefined;
	if (typeof value.id !== "string" || value.id.length === 0) return undefined;
	if (typeof value.contactId !== "number" || !Number.isSafeInteger(value.contactId)) return undefined;
	if (typeof value.query !== "string") return undefined;
	if (typeof value.createdAt !== "string") return undefined;
	if (!Array.isArray(value.sources) || !value.sources.every((source) => typeof source === "string")) return undefined;
	if (!isRecord(value.payload)) return undefined;
	return value as unknown as PendingPeerRequest;
}

function parseDecision(value: unknown): PeerRequestDecision | undefined {
	if (!isRecord(value)) return undefined;
	if (typeof value.id !== "string" || value.id.length === 0) return undefined;
	if (value.decision !== "approve" && value.decision !== "deny") return undefined;
	if (typeof value.decidedAt !== "string") return undefined;
	return { id: value.id, decision: value.decision, decidedAt: value.decidedAt };
}

export function savePendingPeerRequest(workspacePath: string, request: PendingPeerRequest): void {
	atomicWrite(requestPath(workspacePath, request.id), request);
}

export function loadPendingPeerRequest(workspacePath: string, id: string): PendingPeerRequest | undefined {
	const path = requestPath(workspacePath, id);
	if (!existsSync(path)) return undefined;
	try {
		return parsePending(JSON.parse(readFileSync(path, "utf8")) as unknown);
	} catch {
		return undefined;
	}
}

export function listPendingPeerRequests(workspacePath: string): PendingPeerRequest[] {
	const dir = requestsDir(workspacePath);
	if (!existsSync(dir)) return [];
	const requests: PendingPeerRequest[] = [];
	for (const name of readdirSync(dir)) {
		if (!name.endsWith(".json")) continue;
		const parsed = loadPendingPeerRequest(workspacePath, name.slice(0, -".json".length));
		if (parsed === undefined) continue;
		if (loadPeerRequestDecision(workspacePath, parsed.id) !== undefined) continue;
		requests.push(parsed);
	}
	return requests.sort((left, right) => left.createdAt.localeCompare(right.createdAt));
}

export function writePeerRequestDecision(
	workspacePath: string,
	id: string,
	decision: PeerRequestDecisionKind,
): PeerRequestDecision {
	const record: PeerRequestDecision = { id, decision, decidedAt: new Date().toISOString() };
	atomicWrite(decisionPath(workspacePath, id), record);
	return record;
}

export function loadPeerRequestDecision(workspacePath: string, id: string): PeerRequestDecision | undefined {
	const path = decisionPath(workspacePath, id);
	if (!existsSync(path)) return undefined;
	try {
		return parseDecision(JSON.parse(readFileSync(path, "utf8")) as unknown);
	} catch {
		return undefined;
	}
}

export function removePeerRequest(workspacePath: string, id: string): void {
	for (const path of [requestPath(workspacePath, id), decisionPath(workspacePath, id)]) {
		if (!existsSync(path)) continue;
		unlinkSync(path);
	}
}

export function waitForPeerRequestDecision(
	workspacePath: string,
	id: string,
	options: { readonly timeoutMs?: number; readonly abort?: AbortSignal } = {},
): Promise<PeerRequestDecision> {
	const timeoutMs = options.timeoutMs ?? 120_000;
	const existing = loadPeerRequestDecision(workspacePath, id);
	if (existing !== undefined) return Promise.resolve(existing);
	mkdirSync(decisionsDir(workspacePath), { recursive: true });
	return new Promise((resolve, reject) => {
		let settled = false;
		const finish = (action: () => void): void => {
			if (settled) return;
			settled = true;
			watcher.close();
			clearTimeout(timer);
			options.abort?.removeEventListener("abort", onAbort);
			action();
		};
		const tryRead = (): void => {
			const decision = loadPeerRequestDecision(workspacePath, id);
			if (decision !== undefined) finish(() => resolve(decision));
		};
		const watcher = watch(decisionsDir(workspacePath), tryRead);
		const timer = setTimeout(() => finish(() => reject(new ApprovalTimeoutError(id))), timeoutMs);
		const onAbort = (): void => finish(() => reject(new ApprovalAbortedError(id)));
		if (options.abort?.aborted) {
			onAbort();
			return;
		}
		options.abort?.addEventListener("abort", onAbort, { once: true });
		tryRead();
	});
}
