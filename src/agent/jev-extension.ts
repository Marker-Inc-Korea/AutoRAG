import type { AgentTool } from "@earendil-works/pi-agent-core";
import { getEnvApiKey } from "@earendil-works/pi-ai/compat";
import type { ExtensionFactory } from "@earendil-works/pi-coding-agent";
import { Jev, type JevBackend, type Judgment, type Question, type State } from "jev-use";
import { type Static, Type } from "typebox";
import { isDazziSignedIn } from "../cloud/credential.ts";
import { DazziJevBackend } from "../cloud/jev-backend.ts";
import { DECISION_PROVIDER_ORDER, PiDecisionBackend } from "./decision-backends.ts";
import { toToolDefinition } from "./pi-session.ts";
import type { SllmDecisionModel } from "./sllm-decision.ts";

/** `jev` tool name, kept in AutoRAG's reserved set and system prompt. */
export const JEV_TOOL_NAME = "jev";

/**
 * Options for the decision model. Secrets never appear here: a pi provider's credential comes from
 * its environment variable (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `TYPESAFE_API_KEY`,
 * `AI_GATEWAY_API_KEY`, `CLOUDFLARE_API_KEY`, `OPENCODE_API_KEY`) or pi's `auth.json`, and the
 * hosted `dazzi` backend uses the `/login` credential.
 */
export interface JevToolOptions {
	/**
	 * The decision backend: `dazzi`, or any pi provider whose catalog carries decision models
	 * (`openrouter`, `openai`, `typesafe`, `vercel-ai-gateway`, `cloudflare-workers-ai`, or `opencode`).
	 * Omit to take the Dazzi plan when signed in, else the first of
	 * {@link DECISION_PROVIDER_ORDER} whose API key is set. A ready-made `jev-use` backend instance is a
	 * programmatic seam (tests, custom transports); config only accepts names.
	 */
	readonly backend?: string | JevBackend;
	/** Decision model id within the backend, e.g. `openai/gpt-6-luna-decisions` on OpenRouter. Omit for the backend default. */
	readonly model?: string;
	/** Escalate verdicts below this confidence. Default: per-source thresholds. */
	readonly confidenceThreshold?: number;
	/**
	 * Small chat LLM (sLLM) the session moves to when the session check finds the decision model
	 * unavailable; a `jev-use` backend instance is the programmatic seam. Omit for no fallback.
	 */
	readonly fallback?: SllmDecisionModel | JevBackend;
}

/** {@link JevToolOptions} plus where credentials are read from. */
export interface JevJudgeOptions extends JevToolOptions {
	/** Environment consulted for the backend credential, ahead of `process.env`. */
	readonly env?: Record<string, string | undefined>;
	/** Pi agent home: `auth.json` (credentials, the Dazzi `/login`) and `models.json`. Default: Pi's agent dir. */
	readonly agentDir?: string;
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

/** The backend named by `options.backend`, or the first one with a credential. Throws when there is none. */
function resolveBackend(options: JevJudgeOptions): JevBackend {
	const { backend, agentDir } = options;
	if (backend !== undefined && typeof backend !== "string") return backend;
	const env = options.env ?? process.env;
	const providerEnv = Object.fromEntries(
		Object.entries(env).filter((entry): entry is [string, string] => entry[1] !== undefined),
	);
	const name =
		backend ??
		(isDazziSignedIn(agentDir) ? "dazzi" : undefined) ??
		DECISION_PROVIDER_ORDER.find((provider) => getEnvApiKey(provider, providerEnv) !== undefined);
	if (name === undefined) {
		throw new Error(
			"No Jev credentials found: set OPENROUTER_API_KEY, OPENAI_API_KEY, TYPESAFE_API_KEY, AI_GATEWAY_API_KEY, CLOUDFLARE_API_KEY, or OPENCODE_API_KEY, or sign in to Dazzi Models with /login.",
		);
	}
	if (name === "dazzi") return new DazziJevBackend({ env, agentDir });
	return new PiDecisionBackend({
		provider: name,
		...(agentDir !== undefined ? { agentDir } : {}),
		...(options.env !== undefined ? { env: options.env } : {}),
	});
}

/**
 * Builds a Jev judge over one decision backend. The backend is built lazily on the first call, so
 * no credential resolves at construction time; a missing one rejects that call with a named error.
 */
export function createJevJudge(options: JevJudgeOptions = {}): JevJudge {
	let client: Jev | undefined;
	return async (state, questions, callOptions = {}) => {
		client ??= new Jev({
			backend: resolveBackend(options),
			...(options.model !== undefined ? { model: options.model } : {}),
			...(options.confidenceThreshold !== undefined ? { confidenceThreshold: options.confidenceThreshold } : {}),
		});
		return client.judge(state, questions, {
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
