import { Type } from "typebox";
import { ANSWER_CITATION_RULE, ANSWER_IMAGE_EMBED_RULE } from "./answer-guidelines.ts";

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

const answerSchema = Type.String({
	description: `Answer for the caller. At most 5 bullet points (plus optional explanation); reference results by bracketed number (e.g. [1], [2]) without file paths or raw chunk text, except the image-embed exception below. ${ANSWER_CITATION_RULE} ${ANSWER_IMAGE_EMBED_RULE}`,
});

const warningsSchema = Type.Optional(
	Type.Array(Type.String(), { description: "Optional warnings about this result set" }),
);

const curatedResultFields = {
	number: Type.Integer({ description: "1-based result number" }),
	title: Type.String({ description: "Short name of the curated knowledge unit" }),
	summary: Type.String({ description: "Key insight: purpose, details, and line range" }),
	evidence: Type.Array(
		Type.Object({
			excerpt: Type.String({ description: "Supporting excerpt" }),
			lineNumber: Type.Optional(Type.Integer({ description: "Line number of the excerpt, if known" })),
		}),
	),
	confidence: Type.Number({ description: "Confidence in this result, 0..1", minimum: 0, maximum: 1 }),
};

/**
 * Report schema with the explicit number -> source mapping. This is the
 * persisted/typed shape and the input contract of `autorag report` and the MCP
 * report tool, where an external agent curated the evidence and no harness
 * ledger exists. The librarian itself never produces this shape from the
 * model: it writes a plain answer and the harness derives the results from the
 * evidence ids the answer cites.
 */
export const reportSchema = Type.Object({
	answer: answerSchema,
	results: Type.Array(Type.Object(curatedResultFields), { description: "Numbered curated knowledge units." }),
	mapping: Type.Array(
		Type.Object({
			number: Type.Integer({ description: "Matches the result number this entry maps" }),
			source: Type.String({ description: "Source identifier — a real file path or a datasource id" }),
			method: Type.String({ description: "Retrieval method or tool that produced the source" }),
			content: Type.String({ description: "Raw content snippet of the evidence behind this result" }),
			evidenceRefs: Type.Optional(
				Type.Array(evidenceRefSchema, {
					description: "Hidden evidence chunk references supporting this curated result",
				}),
			),
		}),
		{ description: "Internal number -> source/method mapping. One entry per result number." },
	),
	warnings: warningsSchema,
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
	/** Absent for results the harness derived from cited evidence: nothing measured a confidence. */
	readonly confidence?: number;
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
