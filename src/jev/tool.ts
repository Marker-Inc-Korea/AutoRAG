import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { type Static, Type } from "typebox";
import {
	JEV_TOOL_NAME,
	type JevAnswerSummary,
	type JevEvaluation,
	type JevEvaluator,
	JevInputError,
	type JevQuestion,
	type JevState,
} from "./types.ts";

const jevQuestionSchema = Type.Object({
	type: Type.Union([Type.Literal("noul"), Type.Literal("choice"), Type.Literal("score")], {
		description: "noul = yes/no probability, choice = pick one option, score = position on an ordered rubric",
	}),
	instructions: Type.String({
		minLength: 1,
		description: "One narrow, atomic question. Jev answers exactly what is asked.",
	}),
	criteria: Type.Optional(
		Type.Union([Type.Record(Type.String(), Type.String()), Type.Array(Type.String())], {
			description: "choice: option id → description. score: ordered rubric, index 0 lowest (at least two levels).",
		}),
	),
	true: Type.Optional(Type.String({ description: "noul: description of the yes outcome" })),
	false: Type.Optional(Type.String({ description: "noul: description of the no outcome" })),
});

const jevParameters = Type.Object({
	label: Type.Optional(
		Type.String({ description: "Short human label for this judgment, shown in the tool result header" }),
	),
	state: Type.Optional(
		Type.Unknown({
			description: "Text, JSON object, or JSON array Jev should judge. Keep it small and self-contained.",
		}),
	),
	questions: Type.Record(Type.String(), jevQuestionSchema, {
		description: "Named questions, all evaluated against the same state in one batched request.",
	}),
});

type JevQuestionParams = Static<typeof jevQuestionSchema>;

function toJevQuestion(name: string, raw: JevQuestionParams): JevQuestion {
	switch (raw.type) {
		case "noul": {
			if ((raw.true === undefined) !== (raw.false === undefined)) {
				throw new JevInputError(
					`Jev question "${name}" is a noul and must describe both the true and false outcomes, or neither`,
				);
			}
			return {
				type: "noul",
				instructions: raw.instructions,
				...(raw.true !== undefined && raw.false !== undefined ? { true: raw.true, false: raw.false } : {}),
			};
		}
		case "choice": {
			if (raw.criteria === undefined || Array.isArray(raw.criteria)) {
				throw new JevInputError(
					`Jev question "${name}" is a choice and requires criteria as option id → description`,
				);
			}
			return { type: "choice", instructions: raw.instructions, criteria: { ...raw.criteria } };
		}
		case "score": {
			if (raw.criteria === undefined || !Array.isArray(raw.criteria)) {
				throw new JevInputError(`Jev question "${name}" is a score and requires criteria as an ordered list`);
			}
			const criteria = [...raw.criteria];
			if (criteria.length < 2) {
				throw new JevInputError(`Jev question "${name}" is a score and needs at least two criteria levels`);
			}
			return { type: "score", instructions: raw.instructions, criteria };
		}
		default:
			throw new JevInputError(`Jev question "${name}" has an unsupported type`);
	}
}

function toJevState(state: unknown): JevState {
	if (state === undefined || state === null) return null;
	if (typeof state === "string" || Array.isArray(state)) return state;
	if (typeof state === "object") {
		// `typeof` checked; a plain JSON object is indexable by the SDK contract.
		return state as Record<string, unknown>;
	}
	throw new JevInputError("state must be text, a JSON object, a JSON array, or omitted");
}

function formatProbability(value: number): string {
	return value.toFixed(3);
}

function formatAnswer(answer: JevAnswerSummary): string {
	switch (answer.type) {
		case "noul":
			return `noul=${formatProbability(answer.noul ?? 0)}`;
		case "choice": {
			const confidence =
				answer.confidence !== undefined ? ` confidence=${formatProbability(answer.confidence)}` : "";
			return `choice=${answer.choice ?? "?"}${confidence}`;
		}
		case "score": {
			const confidence =
				answer.confidence !== undefined ? ` confidence=${formatProbability(answer.confidence)}` : "";
			return `score=${(answer.score ?? 0).toFixed(2)}${confidence}`;
		}
	}
}

function formatEvaluation(evaluation: JevEvaluation, label: string | undefined): string {
	const header = `jev → ${evaluation.provider}/${evaluation.model}${label !== undefined ? ` · ${label}` : ""}`;
	const lines = Object.entries(evaluation.answers).map(([name, answer]) => `- ${name}: ${formatAnswer(answer)}`);
	return [
		header,
		...lines,
		`usage: ${evaluation.usage.totalTokens} tokens, $${evaluation.usage.costUsd.toFixed(6)}`,
	].join("\n");
}

/**
 * The `jev` decision tool. Returns calibrated probabilities for typed
 * questions instead of prose, so callers can branch on a code-owned threshold.
 */
export function createJevTool(evaluator: JevEvaluator): AgentTool<typeof jevParameters, JevEvaluation> {
	return {
		name: JEV_TOOL_NAME,
		label: "Jev Decision",
		description: [
			"Ask the Jev judgment model typed questions about a state and get calibrated probabilities.",
			"Use for classification, triage, comparison, ranking, and yes/no checks where a number beats prose.",
			"Batch every question about one state into a single call; keep each question narrow and atomic.",
			"Returns probabilities (0..1), a chosen option, or a rubric score — never generated text.",
		].join(" "),
		parameters: jevParameters,
		async execute(
			_toolCallId: string,
			params: Static<typeof jevParameters>,
			signal?: AbortSignal,
		): Promise<AgentToolResult<JevEvaluation>> {
			const questions: Record<string, JevQuestion> = {};
			for (const [name, raw] of Object.entries(params.questions)) {
				questions[name] = toJevQuestion(name, raw);
			}
			if (Object.keys(questions).length === 0) {
				throw new JevInputError("At least one Jev question is required");
			}
			const evaluation = await evaluator.evaluate({
				state: toJevState(params.state),
				questions,
				...(signal !== undefined ? { signal } : {}),
			});
			return {
				content: [{ type: "text", text: formatEvaluation(evaluation, params.label) }],
				details: evaluation,
			};
		},
	};
}
