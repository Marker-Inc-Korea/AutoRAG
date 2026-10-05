import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Jev, type Question, type State } from "jev-use";
import { type Static, Type } from "typebox";
import { toToolDefinition } from "./pi-session.ts";

/** `jev` tool name, kept in AutoRAG's reserved set and system prompt. */
export const JEV_TOOL_NAME = "jev";

/**
 * Jev backends the `jev-use` engine resolves from the environment, in
 * auto-select order: TypeSafe -> OpenRouter -> Vercel AI Gateway.
 */
export type JevBackendName = "typesafe" | "openrouter" | "vercel";

/**
 * OpenRouter's live Jev model id. `jev-use` 0.8.0 ships `typesafe/jev-latest`,
 * which OpenRouter rejects with HTTP 400 ("Model ... does not exist"), so the
 * OpenRouter path is pinned here.
 */
export const OPENROUTER_DEFAULT_MODEL = "typesafe/jev-1.13";

/** Per-backend model overrides applied when the caller names none. */
const BACKEND_DEFAULT_MODEL: Record<string, string> = { openrouter: OPENROUTER_DEFAULT_MODEL };

/**
 * Options for the `jev` decision tool. Secrets never appear here: `jev-use`
 * reads the backend credential from its own environment variable
 * (`TYPESAFE_API_KEY`, `OPENROUTER_API_KEY`, or `AI_GATEWAY_API_KEY`).
 */
export interface JevToolOptions {
	/** Force one backend; omit to let the first credential present win. */
	readonly backend?: JevBackendName;
	/**
	 * Wire model id sent with every call. Omit for the backend default;
	 * OpenRouter is pinned to {@link OPENROUTER_DEFAULT_MODEL} because
	 * `jev-use` 0.8.0's own default (`typesafe/jev-latest`) is not a live
	 * OpenRouter model id.
	 */
	readonly model?: string;
	/** Escalate verdicts below this confidence. Default: per-source thresholds. */
	readonly confidenceThreshold?: number;
}

const jevQuestionSchema = Type.Object({
	id: Type.Optional(Type.String({ description: "Caller-assigned id, echoed back in the verdict" })),
	type: Type.Union([Type.Literal("noul"), Type.Literal("choice"), Type.Literal("score")], {
		description: "noul = P(yes), choice = pick one option, score = ordered level index",
	}),
	question: Type.String({
		minLength: 1,
		description: "One narrow question about the state. Jev answers exactly this.",
	}),
	options: Type.Optional(
		Type.Union([Type.Array(Type.String()), Type.Record(Type.String(), Type.String())], {
			description: "choice only: at least two option labels (or label -> meaning)",
		}),
	),
	levels: Type.Optional(
		Type.Array(Type.String(), {
			description: "score only: at least two ordered level descriptions; the answer indexes into them",
		}),
	),
	criteria: Type.Optional(
		Type.Object(
			{ true: Type.String(), false: Type.String() },
			{ description: "noul only: what yes and no mean, to sharpen calibration" },
		),
	),
});

const jevParameters = Type.Object({
	state: Type.Unknown({
		description: "Text, JSON object, or JSON array Jev should judge. Keep it small and self-contained.",
	}),
	questions: Type.Array(jevQuestionSchema, {
		minItems: 1,
		description: "Named questions, all evaluated against the same state in one batched request.",
	}),
	confidence_threshold: Type.Optional(
		Type.Number({
			minimum: 0,
			maximum: 1,
			description: "Escalate any verdict below this confidence; omitted uses jev-use's per-source defaults",
		}),
	),
});

type JevParams = Static<typeof jevParameters>;

/**
 * Builds the pi extension that registers the `jev` decision tool on top of the
 * [`jev-use`](https://www.npmjs.com/package/jev-use) engine.
 *
 * Jev is TypeSafe's judgment model: typed questions about a state in, calibrated
 * probabilities out — no generated text. `jev-use` owns backend auto-selection,
 * request screening, and response validation; this extension only adapts the
 * pi tool surface. The client is built lazily, so no credential resolves at
 * extension load time.
 */
export function createJevExtension(options: JevToolOptions = {}): ExtensionFactory {
	let client: Jev | undefined;
	return (pi) => {
		const tool: AgentTool<typeof jevParameters, unknown> = {
			name: JEV_TOOL_NAME,
			label: "Jev Decision",
			description: [
				"Ask the Jev judgment model typed questions about a state and get calibrated probabilities.",
				"Use for classification, triage, comparison, ranking, and yes/no checks where a number beats prose.",
				"Batch every question about one state into a single call; keep each question narrow and atomic.",
				"Verdicts with escalate=true are handed back to you; check confidence before acting on a close call.",
			].join(" "),
			parameters: jevParameters,
			async execute(_toolCallId: string, params: JevParams) {
				try {
					client ??= new Jev({
						...(options.backend !== undefined ? { backend: options.backend } : {}),
						...(options.model !== undefined ? { model: options.model } : {}),
						...(options.confidenceThreshold !== undefined
							? { confidenceThreshold: options.confidenceThreshold }
							: {}),
					});
					const state = (params.state ?? "") as State;
					const model = options.model ?? BACKEND_DEFAULT_MODEL[client.backend.name];
					const judgment = await client.judge(state, params.questions as Question[], {
						...(model !== undefined ? { model } : {}),
						...(params.confidence_threshold !== undefined
							? { confidenceThreshold: params.confidence_threshold }
							: {}),
					});
					return { content: [{ type: "text", text: JSON.stringify(judgment, null, 2) }], details: judgment };
				} catch (error) {
					// Missing credentials are surfaced, never silent: hand the step
					// back to the model with jev-use's escalation vocabulary.
					const hint = error instanceof Error ? error.message : String(error);
					return {
						content: [{ type: "text", text: `Jev unavailable: ${hint}` }],
						details: { escalated: true, reason: "unreachable", hint },
					};
				}
			},
		};
		pi.registerTool(toToolDefinition(tool));
	};
}
