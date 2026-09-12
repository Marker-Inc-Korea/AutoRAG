import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";

export const EMIT_FAST_ANSWER_TOOL_NAME = "emit_fast_answer";

const fastAnswerSchema = Type.Object({
	answer: Type.String({
		description:
			"Complete, self-contained first answer for the caller, produced immediately from the baseline retrieval evidence.",
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
		}),
		{ description: "Numbered knowledge units backing the first answer." },
	),
	sources: Type.Optional(
		Type.Array(
			Type.Object({
				number: Type.Integer({ description: "Matches the result number this source belongs to" }),
				source: Type.String({ description: "Source identifier — a real file path or a datasource id" }),
			}),
			{ description: "Optional number -> source mapping for the first answer." },
		),
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
	readonly sources: readonly { readonly number: number; readonly source: string }[];
}

/**
 * Builds the non-terminating fast-answer tool used by the two-phase search
 * flow. The model calls this exactly once during the thinking-off fast phase
 * to deliver an immediate, complete first answer; the run then continues into
 * the verification phase, which ends with emit_autorag_results.
 */
export function createEmitFastAnswerTool(
	capture: (details: AutoRAGFastAnswerDetails) => void,
): AgentTool<typeof fastAnswerSchema, AutoRAGFastAnswerDetails> {
	return {
		name: EMIT_FAST_ANSWER_TOOL_NAME,
		label: "Emit Fast Answer",
		description:
			"Deliver the immediate first answer to the user. Call this exactly once during the fast phase with a complete, self-contained answer built only from the baseline retrieval evidence. Do not call any other tool before this one. The run continues afterwards for verification.",
		parameters: fastAnswerSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<AutoRAGFastAnswerDetails>> {
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
				sources: (params.sources ?? []).map((entry) => ({ number: entry.number, source: entry.source })),
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
