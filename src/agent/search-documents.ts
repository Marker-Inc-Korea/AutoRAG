import type { RetrievalMemory } from "../memory/memory.ts";
import type { CuratedResult } from "../retrieval/types.ts";
import type { AutoRAGResultsDetails } from "./emit-results-tool.ts";

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

function confidenceFrom(score: number): number {
	if (!Number.isFinite(score)) return 0;
	return Math.max(0, Math.min(1, score));
}

function normalizeWarnings(warnings: readonly string[]): SearchDocumentWarning[] {
	return warnings.filter((warning): warning is SearchDocumentWarning => warning === "empty-query");
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

export function recordStructuredResultsSession(
	sessionId: string,
	query: string,
	details: AutoRAGResultsDetails,
	sessions: SearchSessions,
	memory: RetrievalMemory,
): SearchDocumentsResponse {
	const resultNumbers = details.results.map((result) => result.number).sort((a, b) => a - b);
	const mappingNumbers = details.mapping.map((entry) => entry.number).sort((a, b) => a - b);
	const oneToOne =
		resultNumbers.length === mappingNumbers.length &&
		resultNumbers.every((number, index) => number === mappingNumbers[index]);
	if (!oneToOne) {
		throw new Error("emit_autorag_results: result numbers and mapping numbers must be one-to-one");
	}

	const registry = new Map<number, CuratedResult>();
	for (const entry of details.mapping) {
		registry.set(entry.number, {
			index: entry.number,
			content: entry.content,
			source: entry.source,
			method: entry.method,
		});
		const memoryEntry = memory.append({
			query,
			method: entry.method,
			outcome: "pending",
			metadata: { resultCount: 1 },
		});
		memory.registerAttempt({
			id: memoryEntry.id,
			query,
			method: entry.method,
			sources: [entry.source],
			timestamp: memoryEntry.timestamp,
		});
	}
	sessions.set(sessionId, { query, registry });
	memory.save();

	const results: SearchDocumentResult[] = details.results.map((result) => ({
		number: result.number,
		title: result.title,
		summary: result.summary,
		evidence: result.evidence.map((evidence) =>
			evidence.lineNumber !== undefined
				? { excerpt: evidence.excerpt, lineNumber: evidence.lineNumber }
				: { excerpt: evidence.excerpt },
		),
		confidence: confidenceFrom(result.confidence),
		feedbackId: `${sessionId}:${result.number}`,
	}));

	return {
		sessionId,
		query,
		results,
		answer: details.answer,
		searched: details.results.length,
		warnings: normalizeWarnings(details.warnings),
	};
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
