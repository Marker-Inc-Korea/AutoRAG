import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";

export const EMIT_AUTORAG_RESULTS_TOOL_NAME = "emit_autorag_results";

const evidenceRefSchema = Type.Object({
	method: Type.String({ description: "Retrieval method namespace for this evidence chunk" }),
	source: Type.String({ description: "Internal opaque source identifier for this evidence chunk" }),
	excerpt: Type.Optional(Type.String({ description: "Evidence excerpt used for stable ID normalization" })),
	content: Type.Optional(Type.String({ description: "Evidence content used for stable ID normalization" })),
	retrievalResultId: Type.Optional(Type.String({ description: "Backend retrieval ID when path-opaque and stable" })),
	chunkIndex: Type.Optional(Type.Integer({ description: "Chunk index, if available" })),
	lineNumber: Type.Optional(Type.Integer({ description: "Line number, if available" })),
	stableEvidenceId: Type.Optional(Type.String({ description: "Stable evidence ID, if already normalized" })),
	retrieverMix: Type.Optional(
		Type.Array(Type.String({ maxLength: 64 }), {
			description: "Retrievers that contributed to this result",
			maxItems: 8,
		}),
	),
	parserType: Type.Optional(Type.String({ description: "Parser that produced this evidence", maxLength: 120 })),
	documentType: Type.Optional(Type.String({ description: "Document type, if known", maxLength: 120 })),
	documentArea: Type.Optional(Type.String({ description: "Document collection area, if known", maxLength: 120 })),
	evidenceType: Type.Optional(Type.String({ description: "Evidence type, if known", maxLength: 120 })),
	evidenceLocation: Type.Optional(Type.String({ description: "Human-readable evidence location", maxLength: 120 })),
	confidence: Type.Optional(Type.Number({ description: "Evidence confidence, 0..1", minimum: 0, maximum: 1 })),
});

const emitResultsSchema = Type.Object({
	answer: Type.String({
		description: "Final curated answer for the caller. Reference results by number (e.g. [1], [2]).",
	}),
	results: Type.Array(
		Type.Object({
			number: Type.Integer({ description: "1-based result number" }),
			title: Type.String({ description: "Short name of the curated knowledge unit" }),
			summary: Type.String({ description: "Key insight: purpose, details, and line range" }),
			evidence: Type.Array(
				Type.Object({
					excerpt: Type.String({ description: "Supporting excerpt" }),
					lineNumber: Type.Optional(Type.Integer({ description: "Line number of the excerpt, if known" })),
				}),
			),
			confidence: Type.Number({ description: "Confidence in this result, 0..1" }),
		}),
		{ description: "Numbered curated knowledge units." },
	),
	mapping: Type.Array(
		Type.Object({
			number: Type.Integer({ description: "Matches the result number this entry maps" }),
			source: Type.String({ description: "Source identifier — a real file path or a datasource id" }),
			method: Type.String({ description: "Retrieval method or tool that produced the source" }),
			content: Type.String({ description: "Raw content snippet for feedback tracking" }),
			evidenceRefs: Type.Optional(
				Type.Array(evidenceRefSchema, {
					description: "Hidden evidence chunk references supporting this curated result",
				}),
			),
		}),
		{ description: "Internal number -> source/method mapping for feedback. One entry per result number." },
	),
	warnings: Type.Optional(Type.Array(Type.String(), { description: "Optional warnings about this result set" })),
});

export interface AutoRAGEmittedEvidence {
	readonly excerpt: string;
	readonly lineNumber?: number;
}

export interface AutoRAGEmittedResult {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly evidence: readonly AutoRAGEmittedEvidence[];
	readonly confidence: number;
}

export interface AutoRAGEvidenceRef {
	readonly method: string;
	readonly source: string;
	readonly excerpt?: string;
	readonly content?: string;
	readonly retrievalResultId?: string;
	readonly chunkIndex?: number;
	readonly lineNumber?: number;
	readonly stableEvidenceId?: string;
	readonly retrieverMix?: readonly string[];
	readonly parserType?: string;
	readonly documentType?: string;
	readonly documentArea?: string;
	readonly evidenceType?: string;
	readonly evidenceLocation?: string;
	readonly confidence?: number;
}

export interface AutoRAGMappingEntry {
	readonly number: number;
	readonly source: string;
	readonly method: string;
	readonly content: string;
	readonly evidenceRefs: readonly AutoRAGEvidenceRef[];
}

export interface AutoRAGResultsDetails {
	readonly answer: string;
	readonly results: readonly AutoRAGEmittedResult[];
	readonly mapping: readonly AutoRAGMappingEntry[];
	readonly warnings: readonly string[];
}

/**
 * Builds the terminating structured-result tool. The model calls this exactly
 * once as its final action; the typed `details` plus `terminate: true` end the
 * Pi Agent run and hand the curated results back through `capture` — no
 * assistant-text parsing involved.
 */
export function createEmitResultsTool(
	capture: (details: AutoRAGResultsDetails) => void,
): AgentTool<typeof emitResultsSchema, AutoRAGResultsDetails> {
	return {
		name: EMIT_AUTORAG_RESULTS_TOOL_NAME,
		label: "Emit AutoRAG Results",
		description:
			"Return the final structured AutoRAG answer. Call this exactly once as your last action after searching, reading, and curating. Put each result's source (file path or datasource id) in the mapping parameter.",
		parameters: emitResultsSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<AutoRAGResultsDetails>> {
			const details: AutoRAGResultsDetails = {
				answer: params.answer,
				results: params.results.map((result) => ({
					number: result.number,
					title: result.title,
					summary: result.summary,
					evidence: result.evidence.map((evidence) =>
						evidence.lineNumber !== undefined
							? { excerpt: evidence.excerpt, lineNumber: evidence.lineNumber }
							: { excerpt: evidence.excerpt },
					),
					confidence: result.confidence,
				})),
				mapping: params.mapping.map((entry) => ({
					number: entry.number,
					source: entry.source,
					method: entry.method,
					content: entry.content,
					evidenceRefs: (
						entry.evidenceRefs ?? [{ method: entry.method, source: entry.source, content: entry.content }]
					).map((evidence) => ({
						method: evidence.method,
						source: evidence.source,
						...(evidence.excerpt !== undefined ? { excerpt: evidence.excerpt } : {}),
						...(evidence.content !== undefined ? { content: evidence.content } : {}),
						...(evidence.retrievalResultId !== undefined
							? { retrievalResultId: evidence.retrievalResultId }
							: {}),
						...(evidence.chunkIndex !== undefined ? { chunkIndex: evidence.chunkIndex } : {}),
						...(evidence.lineNumber !== undefined ? { lineNumber: evidence.lineNumber } : {}),
						...(evidence.stableEvidenceId !== undefined ? { stableEvidenceId: evidence.stableEvidenceId } : {}),
						...(evidence.retrieverMix !== undefined ? { retrieverMix: evidence.retrieverMix } : {}),
						...(evidence.parserType !== undefined ? { parserType: evidence.parserType } : {}),
						...(evidence.documentType !== undefined ? { documentType: evidence.documentType } : {}),
						...(evidence.documentArea !== undefined ? { documentArea: evidence.documentArea } : {}),
						...(evidence.evidenceType !== undefined ? { evidenceType: evidence.evidenceType } : {}),
						...(evidence.evidenceLocation !== undefined ? { evidenceLocation: evidence.evidenceLocation } : {}),
						...(evidence.confidence !== undefined ? { confidence: evidence.confidence } : {}),
					})),
				})),
				warnings: params.warnings ?? [],
			};
			capture(details);
			return {
				content: [{ type: "text", text: "AutoRAG results emitted." }],
				details,
				terminate: true,
			};
		},
	};
}
