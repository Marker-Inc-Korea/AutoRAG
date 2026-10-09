import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { ANSWER_CITATION_RULE, ANSWER_IMAGE_DELTA_RULE, ANSWER_IMAGE_EMBED_RULE } from "./answer-guidelines.ts";
import { assertCitationsResolve, assertUniqueResultNumbers } from "./citations.ts";
import type { EvidenceLedger } from "./evidence-ledger.ts";

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

const answerSchema = Type.String({
	description: `Answer for the caller. When a first answer was already delivered to the caller during this run, this MUST contain only the corrections and newly verified findings relative to it — never restate the first answer; otherwise it is the complete answer. At most 5 bullet points (plus optional explanation); reference results by bracketed number (e.g. [1], [2]) without file paths or raw chunk text, except the image-embed exception below. ${ANSWER_CITATION_RULE} ${ANSWER_IMAGE_EMBED_RULE} ${ANSWER_IMAGE_DELTA_RULE}`,
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
 * Model-facing schema of `emit_autorag_results`. The model cites evidence by
 * the short ids retrieval tools print next to each result (`refs`); the
 * harness resolves them to the recorded source, method, and chunk, so the
 * model never retypes a path or a chunk.
 */
export const emitResultsSchema = Type.Object({
	answer: answerSchema,
	results: Type.Array(
		Type.Object({
			...curatedResultFields,
			refs: Type.Array(Type.String(), {
				minItems: 1,
				description:
					"Evidence ids supporting this result, copied from the ids shown next to retrieved results (e.g. e3). For a local file you opened yourself with bash, give its absolute path instead.",
			}),
		}),
		{ description: "Numbered curated knowledge units." },
	),
	warnings: warningsSchema,
});

/**
 * Report schema with the explicit number -> source mapping. This is the
 * persisted/typed shape and the input contract of `autorag report` and the MCP
 * report tool, where an external agent curated the evidence and no harness
 * ledger exists.
 */
export const reportSchema = Type.Object({
	answer: answerSchema,
	results: Type.Array(Type.Object(curatedResultFields), { description: "Numbered curated knowledge units." }),
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

export interface EmitResultsToolOptions {
	/** Evidence the run's retrieval tools returned; `refs` resolve against it. */
	readonly ledger: EvidenceLedger;
	/** False in remote sessions: only evidence a tool returned this run may be cited. */
	readonly allowLocalFiles: boolean;
}

/**
 * Builds the terminating structured-result tool. The model calls this exactly
 * once as its final action; the typed `details` plus `terminate: true` end the
 * Pi Agent run and hand the curated results back through `capture` — no
 * assistant-text parsing involved. The number -> source mapping is built here
 * from the harness-held evidence the model cited by id, never from model text.
 */
export function createEmitResultsTool(
	capture: (details: AutoRAGResultsDetails) => void,
	options: EmitResultsToolOptions,
): AgentTool<typeof emitResultsSchema, AutoRAGResultsDetails> {
	return {
		name: EMIT_AUTORAG_RESULTS_TOOL_NAME,
		label: "Emit AutoRAG Results",
		description:
			"Return the final structured AutoRAG answer. Call this exactly once as your last action after searching, reading, and curating. Cite each result's supporting evidence in its refs, using the evidence ids shown next to retrieved results (e.g. e3); the source paths are attached for you. A call whose answer cites a number missing from results, whose result numbers repeat, or whose refs name evidence no tool returned is rejected; fix it and call again.",
		parameters: emitResultsSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<AutoRAGResultsDetails>> {
			assertCitationsResolve(EMIT_AUTORAG_RESULTS_TOOL_NAME, params.answer, params.results);
			assertUniqueResultNumbers(EMIT_AUTORAG_RESULTS_TOOL_NAME, params.results);
			const mapping: AutoRAGMappingEntry[] = params.results.map((result) => {
				const evidenceRefs = options.ledger.resolve(result.refs, {
					label: EMIT_AUTORAG_RESULTS_TOOL_NAME,
					number: result.number,
					fallbackContent: result.evidence.map((evidence) => evidence.excerpt).join("\n") || result.summary,
					allowLocalFiles: options.allowLocalFiles,
				});
				const primary = evidenceRefs[0];
				if (primary === undefined) {
					throw new Error(
						`${EMIT_AUTORAG_RESULTS_TOOL_NAME}: result ${result.number} resolved no evidence. Give it at least one valid ref.`,
					);
				}
				return {
					number: result.number,
					source: primary.source,
					method: primary.method,
					content: primary.content ?? "",
					evidenceRefs,
				};
			});
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
				mapping,
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
