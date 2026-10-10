import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import { ANSWER_CITATION_RULE, ANSWER_IMAGE_EMBED_RULE } from "./answer-guidelines.ts";
import { assertCitationsResolve } from "./citations.ts";
import type { AutoRAGEvidenceRef } from "./emit-results-tool.ts";
import type { EvidenceLedger } from "./evidence-ledger.ts";

export const EMIT_FAST_ANSWER_TOOL_NAME = "emit_fast_answer";

const fastAnswerSchema = Type.Object({
	answer: Type.String({
		description: `Complete, self-contained first answer for the caller in at most 5 bullet points (plus optional explanation), produced immediately from baseline retrieval evidence. Reference results by bracketed number (e.g. [1], [2]) without file paths or raw chunk text, except the image-embed exception below. ${ANSWER_CITATION_RULE} ${ANSWER_IMAGE_EMBED_RULE}`,
	}),
	results: Type.Array(
		Type.Object({
			number: Type.Integer({ description: "1-based result number" }),
			title: Type.String({ description: "Short name of the knowledge unit" }),
			summary: Type.String({ description: "Key insight backing the first answer" }),
			evidence: Type.Optional(
				Type.Array(
					Type.Object({
						excerpt: Type.String({ description: "Supporting excerpt" }),
						lineNumber: Type.Optional(Type.Integer({ description: "Line number of the excerpt, if known" })),
					}),
				),
			),
			confidence: Type.Optional(
				Type.Number({ description: "Confidence in this result, 0..1", minimum: 0, maximum: 1 }),
			),
			refs: Type.Optional(
				Type.Array(Type.String(), {
					description:
						"Evidence ids from the baseline evidence that support this result (e.g. e3). The source path is attached for you. Omit only when no baseline evidence backs the result.",
				}),
			),
		}),
		{ description: "Numbered knowledge units backing the first answer." },
	),
});

export interface AutoRAGFastAnswerResult {
	readonly number: number;
	readonly title: string;
	readonly summary: string;
	readonly evidence: readonly { readonly excerpt: string; readonly lineNumber?: number }[];
	readonly confidence?: number;
}

export interface AutoRAGFastAnswerDetails {
	readonly answer: string;
	readonly results: readonly AutoRAGFastAnswerResult[];
	/** Number -> primary source, derived from `evidenceRefs`. */
	readonly sources: readonly { readonly number: number; readonly source: string }[];
	/** Harness-recorded evidence each result cited; never model-written text. */
	readonly evidenceRefs: readonly { readonly number: number; readonly refs: readonly AutoRAGEvidenceRef[] }[];
}

export interface EmitFastAnswerToolOptions {
	/** Evidence the baseline retrieval produced; `refs` resolve against it. */
	readonly ledger: EvidenceLedger;
}

/**
 * Builds the non-terminating fast-answer tool used by the two-phase search
 * flow. The model calls this exactly once during the thinking-off fast phase
 * to deliver an immediate, complete first answer; the run then continues into
 * the verification phase, which ends with emit_autorag_results. Sources are
 * derived from the evidence ids the model cites, never typed by the model.
 */
export function createEmitFastAnswerTool(
	capture: (details: AutoRAGFastAnswerDetails) => void,
	options: EmitFastAnswerToolOptions,
): AgentTool<typeof fastAnswerSchema, AutoRAGFastAnswerDetails> {
	return {
		name: EMIT_FAST_ANSWER_TOOL_NAME,
		label: "Emit Fast Answer",
		description:
			"Deliver the immediate first answer to the user. Call this exactly once during the fast phase with a complete, self-contained answer built only from the baseline retrieval evidence. Do not call any other tool before this one. The run continues afterwards for verification. A call whose answer cites a number missing from results, or whose refs name evidence that was not provided, is rejected; fix it and call again.",
		parameters: fastAnswerSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<AutoRAGFastAnswerDetails>> {
			assertCitationsResolve(EMIT_FAST_ANSWER_TOOL_NAME, params.answer, params.results);
			const evidenceRefs: { number: number; refs: AutoRAGEvidenceRef[] }[] = [];
			for (const result of params.results) {
				if (result.refs === undefined || result.refs.length === 0) continue;
				const refs = options.ledger.resolve(result.refs, {
					label: EMIT_FAST_ANSWER_TOOL_NAME,
					number: result.number,
					fallbackContent: result.summary,
					allowLocalFiles: false,
				});
				if (refs.length > 0) evidenceRefs.push({ number: result.number, refs });
			}
			const details: AutoRAGFastAnswerDetails = {
				answer: params.answer,
				results: params.results.map((result) => ({
					number: result.number,
					title: result.title,
					summary: result.summary,
					evidence: (result.evidence ?? []).map((evidence) =>
						evidence.lineNumber !== undefined
							? { excerpt: evidence.excerpt, lineNumber: evidence.lineNumber }
							: { excerpt: evidence.excerpt },
					),
					...(result.confidence !== undefined ? { confidence: result.confidence } : {}),
				})),
				sources: evidenceRefs.flatMap((entry) => {
					const [primary] = entry.refs;
					return primary === undefined ? [] : [{ number: entry.number, source: primary.source }];
				}),
				evidenceRefs,
			};
			capture(details);
			return {
				content: [
					{
						type: "text",
						text: "Fast answer delivered to the user. Stop now; a deeper verification pass follows separately.",
					},
				],
				details,
			};
		},
	};
}
