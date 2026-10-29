import { normalizeSessionEvidenceRef, type RetrievalMemory, type SessionEvidenceRef } from "../memory/memory.ts";
import type { RetrievalResult } from "../retrieval/types.ts";
import { assertResultsMappingOneToOne, formatCitationList, stripUnresolvedCitations } from "./citations.ts";
import type { AutoRAGMappingEntry, AutoRAGResultsDetails } from "./results.ts";

export type SearchDocumentWarning = "empty-query";

export type SearchDocumentDiagnosticSeverity = "info" | "warning" | "error";

/**
 * Stable diagnostic codes surfaced through the public search
 * response. The union is intentionally broad so later degraded-mode wiring
 * (#21/#22) does not require another source-breaking type change; codes not yet
 * emitted are still valid members.
 */
export type SearchDocumentDiagnosticCode =
	| "empty-query"
	| "no-verified-results"
	| "unknown-warning"
	| "caller-tool-dropped"
	| "minsync-staging-excluded"
	| "minsync-file-skipped"
	| "minsync-sync-failed"
	| "minsync-lock-held"
	| "embedder-unavailable"
	| "embedding-identity-mismatch"
	| "lexical-tokenizer-unavailable"
	| "parser-skipped"
	| "parser-failed"
	| "parser-crashed"
	| "source-changed"
	| "source-vanished"
	| "source-unreadable"
	| "source-read-failed"
	| "mirror-write-failed"
	| "parser-warning"
	| "duplicate-excluded"
	| "unsupported-file"
	| "stale-index"
	| "deleted-mirror"
	| "search-root-unavailable"
	| "mass-delete-suppressed"
	| "walk-truncated"
	| "directory-unreadable"
	| "pdf-extract-thin"
	| "retrieval-method-failed"
	| "rerank-failed"
	| "jikji-unavailable"
	| "jikji-prepare-failed"
	| "jikji-find-failed"
	| "everything-index-failed"
	| "fsearch-index-failed"
	| "fsearch-binary-missing"
	| "refresh-failed"
	| "refresh-interrupted"
	| "watch-failed"
	| "watch-limited"
	| "unknown-datasource-skill"
	| "no-final-answer"
	| "model-request-failed"
	| "tool-call-burst"
	| "query-routed"
	| "query-route-fallback"
	| "query-decomposition-failed"
	| "follow-up-skipped"
	| "follow-up-check-fallback"
	| "datasources-selected"
	| "datasource-selection-fallback"
	| "web-datasource-id-taken"
	| "web-search-failed"
	| "citation-without-result"
	| "line-reference-unverified"
	| "citation-irrelevant"
	| "citation-relevance-fallback"
	| "evidence-judged"
	| "evidence-judgment-fallback"
	| "memory-search-fallback"
	| "self-config-unavailable"
	| "self-config-rolled-back"
	| "organize-skill-unavailable"
	| "embedding-unavailable";

export interface SearchDocumentDiagnostic {
	readonly code: SearchDocumentDiagnosticCode;
	readonly severity: SearchDocumentDiagnosticSeverity;
	readonly message: string;
	/** Component label (e.g. "sanitizer", "minsync") or opaque virtual path — never a real filesystem path. */
	readonly source?: string;
	/** Structured cause for a component diagnostic, such as a changed source revision. */
	readonly reason?: string;
}

export interface SearchDocumentEvidence {
	readonly excerpt: string;
	/** First line of the chunk in its parsed document, measured by the harness. */
	readonly lineNumber?: number;
	/** Last line of the chunk in its parsed document. */
	readonly lineEnd?: number;
	/** Article or heading the chunk sits in. */
	readonly section?: string;
}

export interface SearchDocumentResult {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly evidence: readonly SearchDocumentEvidence[];
	/** Absent when nothing measured one: derived results carry no model-reported confidence. */
	readonly confidence?: number;
	readonly source?: string;
}

/** One retrieval candidate captured in a run's retrieval trace. */
export interface SearchDocumentRetrievalTraceResult {
	readonly source?: string;
	readonly excerpt: string;
	readonly score?: number;
}

/**
 * What one retrieval tool execution found during a search run. Attached to the
 * degraded fallback response when the agent never wrote a final answer,
 * so an upstream agent can still inspect the raw candidates.
 */
export interface SearchDocumentRetrievalTraceEntry {
	readonly tool: string;
	readonly query?: string;
	readonly resultCount: number;
	readonly results: readonly SearchDocumentRetrievalTraceResult[];
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
	/**
	 * Retrieval candidates gathered during the run. Populated on the degraded
	 * fallback path (missing final emit); absent or empty otherwise.
	 */
	readonly retrievalTrace?: readonly SearchDocumentRetrievalTraceEntry[];
}

/** Cap and shape retrieval results for the run trace (additive tool details). */
export function toRetrievalTraceResults(
	results: readonly RetrievalResult[],
	limit = 5,
	excerptChars = 300,
): readonly SearchDocumentRetrievalTraceResult[] {
	return results.slice(0, limit).map((result) => ({
		source: result.source,
		excerpt: result.content.replace(/\s+/gu, " ").slice(0, excerptChars),
		score: result.score,
	}));
}

export type SearchDocumentsStreamEvent =
	| {
			readonly type: "progress";
			readonly sessionId: string;
			readonly query: string;
			readonly text: string;
	  }
	| {
			/**
			 * Immediate first answer from the minimal-thinking fast phase. Always
			 * yielded before `complete` when the two-phase flow produced one; the
			 * `complete` event's response remains the verified final answer.
			 */
			readonly type: "preliminary";
			readonly response: SearchDocumentsResponse;
	  }
	| {
			readonly type: "complete";
			readonly response: SearchDocumentsResponse;
	  };

function confidenceFrom(score: number): number {
	if (!Number.isFinite(score)) return 0;
	return Math.max(0, Math.min(1, score));
}

function normalizeWarnings(warnings: readonly string[]): SearchDocumentWarning[] {
	return warnings.filter((warning): warning is SearchDocumentWarning => warning === "empty-query");
}

/**
 * Enforce the response invariant: every `[n]` in `answer` resolves to a
 * `results[].number` (issue #1788). Answers the librarian derives from cited
 * evidence already satisfy it; this is the boundary guarantee for external
 * `lite report` curators. Unmatched markers are dropped and reported as a
 * `citation-without-result` diagnostic.
 */
function reconcileCitations(
	answer: string,
	results: readonly { readonly number: number }[],
): { readonly answer: string; readonly diagnostics: readonly SearchDocumentDiagnostic[] } {
	const reconciled = stripUnresolvedCitations(answer, results);
	if (reconciled.unresolved.length === 0) return { answer, diagnostics: [] };
	return {
		answer: reconciled.answer,
		diagnostics: [
			{
				code: "citation-without-result",
				severity: "warning",
				message: `Removed answer citation(s) ${formatCitationList(reconciled.unresolved)} with no matching result; every remaining citation resolves to results[].number.`,
				source: "agent",
			},
		],
	};
}

function toSearchResults(
	details: AutoRAGResultsDetails,
	sourceByNumber: ReadonlyMap<number, string | undefined>,
): SearchDocumentResult[] {
	return details.results.map((result) => ({
		number: result.number,
		title: result.title,
		summary: result.summary,
		evidence: result.evidence.map((evidence) => ({
			excerpt: evidence.excerpt,
			...(evidence.lineNumber !== undefined ? { lineNumber: evidence.lineNumber } : {}),
			...(evidence.lineEnd !== undefined ? { lineEnd: evidence.lineEnd } : {}),
			...(evidence.section !== undefined ? { section: evidence.section } : {}),
		})),
		...(result.confidence !== undefined ? { confidence: confidenceFrom(result.confidence) } : {}),
		source: sourceByNumber.get(result.number),
	}));
}

/**
 * Build the preliminary (fast-phase) search response. Unlike
 * {@link recordStructuredResultsSession} this NEVER touches memory — the final
 * response owns that.
 */
export function createPreliminarySearchDocumentsResponse(
	sessionId: string,
	query: string,
	details: AutoRAGResultsDetails,
	diagnostics: readonly SearchDocumentDiagnostic[] = [],
): SearchDocumentsResponse {
	const sourceByNumber = new Map(details.mapping.map((entry) => [entry.number, entry.source]));
	return {
		sessionId,
		query,
		results: toSearchResults(details, sourceByNumber),
		answer: details.answer,
		searched: details.results.length,
		warnings: [],
		diagnostics: [...diagnostics],
	};
}

export function createEmptySearchDocumentsResponse(
	sessionId: string,
	query: string,
	diagnostics: readonly SearchDocumentDiagnostic[] = [],
): SearchDocumentsResponse {
	return {
		sessionId,
		query,
		results: [],
		answer: "",
		searched: 0,
		warnings: ["empty-query"],
		diagnostics: [...diagnostics],
	};
}

export function normalizeEntryEvidenceRefs(entry: AutoRAGMappingEntry): SessionEvidenceRef[] {
	const rawRefs =
		entry.evidenceRefs.length > 0
			? entry.evidenceRefs
			: [{ method: entry.method, source: entry.source, content: entry.content }];
	const derivedRetrieverMix = Array.from(new Set(rawRefs.map((ref) => ref.method)));
	return rawRefs.map((ref) => {
		if (ref.excerpt === undefined && ref.content === undefined) {
			throw new Error("results: every evidenceRef must include excerpt or content");
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
			retrieverMix: ref.retrieverMix ?? derivedRetrieverMix,
			...(ref.parserType !== undefined ? { parserType: ref.parserType } : {}),
			...(ref.documentType !== undefined ? { documentType: ref.documentType } : {}),
			...(ref.documentArea !== undefined ? { documentArea: ref.documentArea } : {}),
			...(ref.evidenceType !== undefined ? { evidenceType: ref.evidenceType } : {}),
			...(ref.lineEnd !== undefined ? { lineEnd: ref.lineEnd } : {}),
			...(ref.section !== undefined ? { section: ref.section } : {}),
			...(ref.confidence !== undefined ? { confidence: ref.confidence } : {}),
		});
	});
}

export function recordStructuredResultsSession(
	sessionId: string,
	query: string,
	details: AutoRAGResultsDetails,
	memory: RetrievalMemory,
	componentDiagnostics: readonly SearchDocumentDiagnostic[] = [],
	options: {
		/** Skip the retrieval-memory write. */
		readonly isolateMemory?: boolean;
	} = {},
): SearchDocumentsResponse {
	assertResultsMappingOneToOne("results", details.results, details.mapping);

	const memoryResults = [];
	for (const entry of details.mapping) {
		const evidenceRefs = normalizeEntryEvidenceRefs(entry);
		const emittedResult = details.results.find((result) => result.number === entry.number);
		memoryResults.push({
			number: entry.number,
			title: emittedResult?.title ?? `Result ${entry.number}`,
			summary: emittedResult?.summary ?? entry.content,
			content: entry.content,
			method: entry.method,
			source: entry.source,
			...(emittedResult?.confidence !== undefined ? { confidence: confidenceFrom(emittedResult.confidence) } : {}),
			evidenceRefs,
		});
	}
	if (!options.isolateMemory) {
		memory.recordCuratedResultsSession({ sessionId, query, results: memoryResults });
		memory.save();
	}

	const results = toSearchResults(
		details,
		// An empty mapping source means "not reported".
		new Map(details.mapping.map((entry) => [entry.number, entry.source || undefined])),
	);
	const citations = reconcileCitations(details.answer, results);
	const answer = citations.answer;

	const diagnostics: SearchDocumentDiagnostic[] = [...citations.diagnostics];
	// Never silently drop unknown emitted warnings — route them to diagnostics.
	for (const warning of details.warnings) {
		if (warning === "empty-query") continue;
		diagnostics.push({
			code: "unknown-warning",
			severity: "info",
			message: `Unrecognized warning from the search agent: ${warning}`,
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
