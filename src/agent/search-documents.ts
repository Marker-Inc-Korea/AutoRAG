import type { RetrievalMemory } from "../memory/memory.ts";
import type { CuratedResult, RetrievalResult } from "../retrieval/types.ts";

export type SearchDocumentWarning = "empty-query";

export interface SearchDocumentEvidence {
	readonly excerpt: string;
	readonly lineNumber?: number;
}

export interface SearchDocumentResult {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly evidence: readonly SearchDocumentEvidence[];
	readonly confidence: number;
	readonly feedbackId: string;
}

export interface SearchDocumentsResponse {
	readonly sessionId: string;
	readonly query: string;
	readonly results: readonly SearchDocumentResult[];
	readonly answer: string;
	readonly searched: number;
	readonly warnings: readonly SearchDocumentWarning[];
}

type SearchSessions = Map<string, { query: string; registry: Map<number, CuratedResult> }>;
type ReadonlySearchSessions = ReadonlyMap<string, { query: string; registry: ReadonlyMap<number, CuratedResult> }>;

function evidenceFrom(result: RetrievalResult): SearchDocumentEvidence {
	const lineNumber = result.metadata.lineNumber;
	if (typeof lineNumber === "number") {
		return { excerpt: result.content, lineNumber };
	}
	return { excerpt: result.content };
}

function confidenceFrom(score: number): number {
	if (!Number.isFinite(score)) return 0;
	return Math.max(0, Math.min(1, score));
}

export function createEmptySearchDocumentsResponse(
	sessionId: string,
	query: string,
	sessions: SearchSessions,
): SearchDocumentsResponse {
	sessions.set(sessionId, { query, registry: new Map() });
	return {
		sessionId,
		query,
		results: [],
		answer: "",
		searched: 0,
		warnings: ["empty-query"],
	};
}

export function createSearchDocumentsResponse(
	sessionId: string,
	query: string,
	retrievalResults: readonly RetrievalResult[],
): SearchDocumentsResponse {
	const results = retrievalResults.map((result, index) => {
		const number = index + 1;
		return {
			number,
			title: result.content,
			summary: result.content,
			evidence: [evidenceFrom(result)],
			confidence: confidenceFrom(result.score),
			feedbackId: `${sessionId}:${number}`,
		};
	});

	return {
		sessionId,
		query,
		results,
		answer: results.map((result) => `[${result.number}] ${result.summary}`).join("\n"),
		searched: retrievalResults.length,
		warnings: [],
	};
}

export function recordSearchDocumentsSession(
	sessionId: string,
	query: string,
	retrievalResults: readonly RetrievalResult[],
	sessions: SearchSessions,
	memory: RetrievalMemory,
): SearchDocumentsResponse {
	const registry = new Map<number, CuratedResult>();
	for (let index = 0; index < retrievalResults.length; index += 1) {
		const result = retrievalResults[index];
		const method = typeof result.metadata.method === "string" ? result.metadata.method : "unknown";
		registry.set(index + 1, { index: index + 1, content: result.content, source: result.source, method });
		const memoryEntry = memory.append({ query, method, outcome: "pending", metadata: { resultCount: 1 } });
		memory.registerAttempt({
			id: memoryEntry.id,
			query,
			method,
			sources: [result.source],
			timestamp: memoryEntry.timestamp,
		});
	}
	sessions.set(sessionId, { query, registry });
	memory.save();
	return createSearchDocumentsResponse(sessionId, query, retrievalResults);
}

export function recordNumberedFeedback(
	sessions: ReadonlySearchSessions,
	memory: RetrievalMemory,
	sessionId: string,
	usefulNumbers: readonly number[],
	notUsefulNumbers: readonly number[],
): void {
	const session = sessions.get(sessionId);
	if (!session) return;
	const feedback = [];
	for (const n of usefulNumbers) {
		const entry = session.registry.get(n);
		if (entry) feedback.push({ source: entry.source, useful: true });
	}
	for (const n of notUsefulNumbers) {
		const entry = session.registry.get(n);
		if (entry) feedback.push({ source: entry.source, useful: false });
	}
	if (feedback.length === 0) return;
	memory.recordResultFeedback(feedback);
	memory.save();
}
