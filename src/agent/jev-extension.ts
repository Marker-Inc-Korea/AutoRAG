import type { AgentTool } from "@earendil-works/pi-agent-core";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Jev, type JevBackend, type Judgment, type Question, type State } from "jev-use";
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
	/**
	 * Force one backend by name; omit to let the first credential present win.
	 * A ready-made `jev-use` backend instance is a programmatic seam (tests,
	 * custom transports); config only accepts names.
	 */
	readonly backend?: JevBackendName | JevBackend;
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

/** {@link JevToolOptions} plus the environment `jev-use` resolves credentials from. */
export interface JevJudgeOptions extends JevToolOptions {
	/** Environment consulted for the backend credential. Default `process.env`. */
	readonly env?: Record<string, string | undefined>;
}

/**
 * One batched Jev judgment. Rejects only when no backend can be built (for
 * example a missing credential); an unreachable backend resolves with
 * `escalate: true` verdicts, per `jev-use`'s contract.
 */
export type JevJudge = (
	state: State,
	questions: Question[],
	options?: { readonly confidenceThreshold?: number },
) => Promise<Judgment>;

/**
 * Builds the Jev judge shared by the `jev` tool and the query router. The
 * `jev-use` client is created lazily on the first call, so no credential
 * resolves at construction time, and the OpenRouter model pin applies to
 * every caller.
 */
export function createJevJudge(options: JevJudgeOptions = {}): JevJudge {
	let client: Jev | undefined;
	return async (state, questions, callOptions = {}) => {
		client ??= new Jev({
			...(options.backend !== undefined ? { backend: options.backend } : {}),
			...(options.env !== undefined ? { env: options.env } : {}),
			...(options.model !== undefined ? { model: options.model } : {}),
			...(options.confidenceThreshold !== undefined ? { confidenceThreshold: options.confidenceThreshold } : {}),
		});
		const model = options.model ?? BACKEND_DEFAULT_MODEL[client.backend.name];
		return client.judge(state, questions, {
			...(model !== undefined ? { model } : {}),
			...(callOptions.confidenceThreshold !== undefined
				? { confidenceThreshold: callOptions.confidenceThreshold }
				: {}),
		});
	};
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
 * pi tool surface over {@link createJevJudge}. Pass the agent's shared judge
 * so the tool and the query router reuse one lazily built client.
 */
export function createJevExtension(judgeOrOptions: JevJudge | JevToolOptions = {}): ExtensionFactory {
	const judge = typeof judgeOrOptions === "function" ? judgeOrOptions : createJevJudge(judgeOrOptions);
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
					const state = (params.state ?? "") as State;
					const judgment = await judge(state, params.questions as Question[], {
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
