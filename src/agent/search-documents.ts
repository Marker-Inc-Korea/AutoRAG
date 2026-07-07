import { normalizeSessionEvidenceRef, type RetrievalMemory, type SessionEvidenceRef } from "../memory/memory.ts";
import type { CuratedResult } from "../retrieval/types.ts";
import type { AutoRAGMappingEntry, AutoRAGResultsDetails } from "./emit-results-tool.ts";
import { createPathSanitizer, type PathSanitizer } from "./public-output-sanitizer.ts";

export type SearchDocumentWarning = "empty-query";

export type SearchDocumentDiagnosticSeverity = "info" | "warning" | "error";

/**
 * Stable, path-opaque diagnostic codes surfaced through the public search
 * response. The union is intentionally broad so later degraded-mode wiring
 * (#21/#22) does not require another source-breaking type change; codes not yet
 * emitted are still valid members.
 */
export type SearchDocumentDiagnosticCode =
	| "empty-query"
	| "unknown-warning"
	| "caller-tool-dropped"
	| "path-redacted"
	| "bm25-unavailable"
	| "bm25-degraded-fallback"
	| "minsync-unavailable"
	| "parser-skipped"
	| "parser-failed"
	| "unsupported-file"
	| "stale-index"
	| "deleted-mirror"
	| "retrieval-method-failed"
	| "jikji-unavailable"
	| "jikji-prepare-failed"
	| "refresh-failed"
	| "watch-failed"
	| "watch-limited";

export interface SearchDocumentDiagnostic {
	readonly code: SearchDocumentDiagnosticCode;
	readonly severity: SearchDocumentDiagnosticSeverity;
	readonly message: string;
	/** Component label (e.g. "sanitizer", "bm25") or opaque virtual path — never a real filesystem path. */
	readonly source?: string;
}

/** Redaction context for the public-output sanitizer, supplied by the agent. */
export interface RedactionContext {
	readonly workspaceRoot: string;
	readonly searchPaths: readonly string[];
}

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
	/**
	 * Structured degraded-mode diagnostics. Typed optional for the compatibility
	 * window, but ALWAYS populated at runtime (defaults to an empty array).
	 */
	readonly diagnostics?: readonly SearchDocumentDiagnostic[];
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
		diagnostics: [],
	};
}

function normalizeEntryEvidenceRefs(entry: AutoRAGMappingEntry): SessionEvidenceRef[] {
	const rawRefs =
		entry.evidenceRefs.length > 0
			? entry.evidenceRefs
			: [{ method: entry.method, source: entry.source, content: entry.content }];
	return rawRefs.map((ref) => {
		if (ref.excerpt === undefined && ref.content === undefined) {
			throw new Error("emit_autorag_results: every evidenceRef must include excerpt or content");
		}
		return normalizeSessionEvidenceRef({
			method: ref.method,
			source: ref.source,
			...(ref.excerpt !== undefined ? { excerpt: ref.excerpt } : {}),
			...(ref.content !== undefined ? { content: ref.content } : {}),
			...(ref.retrievalResultId !== undefined ? { retrievalResultId: ref.retrievalResultId } : {}),
			...(ref.chunkIndex !== undefined ? { chunkIndex: ref.chunkIndex } : {}),
			...(ref.lineNumber !== undefined ? { lineNumber: ref.lineNumber } : {}),
			...(ref.stableEvidenceId !== undefined ? { stableEvidenceId: ref.stableEvidenceId } : {}),
		});
	});
}

export function recordStructuredResultsSession(
	sessionId: string,
	query: string,
	details: AutoRAGResultsDetails,
	sessions: SearchSessions,
	memory: RetrievalMemory,
	redactionContext?: RedactionContext,
	componentDiagnostics: readonly SearchDocumentDiagnostic[] = [],
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
	const memoryResults = [];
	for (const entry of details.mapping) {
		const evidenceRefs = normalizeEntryEvidenceRefs(entry);
		registry.set(entry.number, {
			index: entry.number,
			content: entry.content,
			source: entry.source,
			method: entry.method,
			evidenceRefs,
		});
		const emittedResult = details.results.find((result) => result.number === entry.number);
		memoryResults.push({
			number: entry.number,
			title: emittedResult?.title ?? `Result ${entry.number}`,
			summary: emittedResult?.summary ?? entry.content,
			content: entry.content,
			method: entry.method,
			source: entry.source,
			evidenceRefs,
		});
	}
	memory.recordCuratedResultsSession({ sessionId, query, results: memoryResults });
	sessions.set(sessionId, { query, registry });
	memory.save();

	const sanitizer = buildSanitizer(details, redactionContext);
	let redacted = false;
	const scrub = (text: string): string => {
		const outcome = sanitizer.sanitize(text);
		if (outcome.redacted) redacted = true;
		return outcome.text;
	};

	const results: SearchDocumentResult[] = details.results.map((result) => ({
		number: result.number,
		title: scrub(result.title),
		summary: scrub(result.summary),
		evidence: result.evidence.map((evidence) =>
			evidence.lineNumber !== undefined
				? { excerpt: scrub(evidence.excerpt), lineNumber: evidence.lineNumber }
				: { excerpt: scrub(evidence.excerpt) },
		),
		confidence: confidenceFrom(result.confidence),
		feedbackId: `${sessionId}:${result.number}`,
	}));
	const answer = scrub(details.answer);

	const diagnostics: SearchDocumentDiagnostic[] = [];
	if (redacted) {
		diagnostics.push({
			code: "path-redacted",
			severity: "warning",
			message: "Removed internal source paths from the public answer/results before returning them.",
			source: "sanitizer",
		});
	}
	// Never silently drop unknown emitted warnings — route them to diagnostics.
	for (const warning of details.warnings) {
		if (warning === "empty-query") continue;
		diagnostics.push({
			code: "unknown-warning",
			severity: "info",
			message: `Unrecognized warning from the search agent: ${scrub(warning)}`,
			source: "agent",
		});
	}
	diagnostics.push(...componentDiagnostics);

	return {
		sessionId,
		query,
		results,
		answer,
		searched: details.results.length,
		warnings: normalizeWarnings(details.warnings),
		diagnostics,
	};
}

function buildSanitizer(details: AutoRAGResultsDetails, redactionContext?: RedactionContext): PathSanitizer {
	if (redactionContext === undefined) {
		return { sanitize: (text) => ({ text, redacted: false }) };
	}
	const sources = new Set<string>();
	for (const entry of details.mapping) {
		if (entry.source) sources.add(entry.source);
		for (const ref of entry.evidenceRefs) {
			if (ref.source) sources.add(ref.source);
		}
	}
	return createPathSanitizer({
		workspaceRoot: redactionContext.workspaceRoot,
		searchPaths: redactionContext.searchPaths,
		sources,
	});
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
		if (session.registry.has(n)) feedback.push({ number: n, useful: true });
	}
	for (const n of notUsefulNumbers) {
		if (session.registry.has(n)) feedback.push({ number: n, useful: false });
	}
	if (feedback.length === 0) return;
	if (memory.recordNumberedFeedback({ sessionId, query: session.query, feedback })) {
		memory.save();
	}
}
