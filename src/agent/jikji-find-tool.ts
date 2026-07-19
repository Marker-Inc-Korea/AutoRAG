import type { AgentTool, AgentToolResult } from "@earendil-works/pi-agent-core";
import { Type } from "typebox";
import type {
	JikjiAnswerPack,
	JikjiCandidate,
	JikjiDiagnostic,
	JikjiEvidence,
	JikjiHandoffAction,
	JikjiNextRead,
} from "../jikji/index.ts";

export const JIKJI_FIND_TOOL_NAME = "jikji_find";

/**
 * Run-scoped merged policy assembled from per-root Jikji answer packs using
 * least-privilege (restrictive-wins) semantics. Set on the agent after each
 * `findJikji` call and consumed by the bash gate.
 */
export interface MergedJikjiPolicy {
	readonly handoffAction: JikjiHandoffAction;
	readonly stopAfterFind: boolean;
	readonly forbiddenTools: readonly string[];
	readonly allowedFollowups: readonly string[];
	readonly agentShouldNotRerank: boolean;
	readonly rawFallbackAllowed: boolean;
}

/**
 * Per-root policy summary for one successful (ok) root, captured BEFORE the
 * least-privilege merge. `root` is the configured search path string.
 */
export interface JikjiFindPerRootPolicy {
	readonly root: string;
	readonly handoffAction: JikjiHandoffAction;
	readonly stopAfterFind: boolean;
	readonly forbiddenTools: readonly string[];
	readonly allowedFollowups: readonly string[];
	readonly agentShouldNotRerank: boolean;
}

/** Result returned by the agent's `findJikji` provider method. */
export interface JikjiFindProviderResult {
	/** Merged answer pack with normalized paths, or undefined when unavailable. */
	readonly answerPack: JikjiAnswerPack | undefined;
	/** Merged policy, or undefined when jikji is unavailable (bash allowed). */
	readonly policy: MergedJikjiPolicy | undefined;
	readonly diagnostics: readonly JikjiDiagnostic[];
	/** Search roots that were queried. */
	readonly roots: readonly string[];
	/** Per-root policy summaries for ok roots, before the merge. */
	readonly perRoot: readonly JikjiFindPerRootPolicy[];
}

export interface JikjiFindProvider {
	findJikji(
		query: string,
		opts?: { readonly topK?: number; readonly first?: boolean },
	): Promise<JikjiFindProviderResult>;
}

export interface JikjiFindDetails {
	readonly method: "jikji_find";
	readonly answerCount: number;
	readonly handoffAction: JikjiHandoffAction | undefined;
	readonly stopAfterFind: boolean | undefined;
	readonly rawFallbackAllowed: boolean | undefined;
	readonly sources: readonly string[];
	readonly candidates: readonly JikjiCandidate[];
	readonly evidencePack: readonly JikjiEvidence[];
	readonly forbiddenTools: readonly string[];
	readonly allowedFollowups: readonly string[];
	readonly perRoot: readonly JikjiFindPerRootPolicy[];
}

const jikjiFindSchema = Type.Object({
	query: Type.String({ description: "Query for Jikji local file discovery." }),
	topK: Type.Optional(Type.Integer({ description: "Maximum number of answer paths / candidates to return." })),
	first: Type.Optional(Type.Boolean({ description: "Return only the first/best answer." })),
});

/**
 * LLM-facing tool that calls Jikji `find` for local file discovery. The schema
 * is model-safe and exposes no root paths: `{ query, topK?, first? }`. When
 * Jikji is unavailable or all roots failed, the tool returns a short fallback
 * message so the agent falls back to bash, and no policy is set.
 */
export function createJikjiFindTool(provider: JikjiFindProvider): AgentTool<typeof jikjiFindSchema, JikjiFindDetails> {
	return {
		name: JIKJI_FIND_TOOL_NAME,
		label: "Jikji Find",
		description:
			"Discover local files via Jikji. Returns answer_paths with per-candidate next_read hints and a tool-call policy directive. Call this FIRST for local file discovery when Jikji is configured.",
		parameters: jikjiFindSchema,
		async execute(_toolCallId, params): Promise<AgentToolResult<JikjiFindDetails>> {
			const query = String(params.query ?? "").trim();
			if (query.length === 0) {
				return {
					content: [{ type: "text", text: "Jikji query was empty; nothing searched." }],
					details: {
						method: "jikji_find",
						answerCount: 0,
						handoffAction: undefined,
						stopAfterFind: undefined,
						rawFallbackAllowed: undefined,
						sources: [],
						candidates: [],
						evidencePack: [],
						forbiddenTools: [],
						allowedFollowups: [],
						perRoot: [],
					},
				};
			}
			const { answerPack, policy, diagnostics, perRoot } = await provider.findJikji(query, {
				topK: params.topK,
				first: params.first,
			});

			if (answerPack === undefined) {
				const diagMsg =
					diagnostics.length > 0
						? diagnostics.map((d) => d.message).join(" ")
						: "Jikji is unavailable or all roots failed.";
				return {
					content: [
						{
							type: "text",
							text: `jikji-unavailable; use bash to explore.\n${diagMsg}`.trim(),
						},
					],
					details: {
						method: "jikji_find",
						answerCount: 0,
						handoffAction: undefined,
						stopAfterFind: undefined,
						rawFallbackAllowed: undefined,
						sources: [],
						candidates: [],
						evidencePack: [],
						forbiddenTools: [],
						allowedFollowups: [],
						perRoot,
					},
				};
			}

			const directive = formatDirective(policy);
			const body = formatAnswerPack(answerPack);
			const text = `${body}\n\n${directive}`;
			return {
				content: [{ type: "text", text }],
				details: {
					method: "jikji_find",
					answerCount: answerPack.answerPaths.length,
					handoffAction: policy?.handoffAction,
					stopAfterFind: policy?.stopAfterFind,
					rawFallbackAllowed: policy?.rawFallbackAllowed,
					sources: [...answerPack.answerPaths],
					candidates: [...answerPack.candidates],
					evidencePack: [...answerPack.evidencePack],
					forbiddenTools: policy?.forbiddenTools ?? [],
					allowedFollowups: policy?.allowedFollowups ?? [],
					perRoot,
				},
			};
		},
	};
}

function formatAnswerPack(pack: JikjiAnswerPack): string {
	const lines: string[] = [];
	if (pack.answerPaths.length === 0) {
		lines.push("No answer paths returned by Jikji.");
	} else {
		lines.push("answer_paths:");
		for (const path of pack.answerPaths) {
			const hint = nextReadHintFor(pack.candidates, path);
			lines.push(hint ? `- ${path} (next_read: ${hint})` : `- ${path}`);
		}
	}
	return lines.join("\n");
}

function nextReadHintFor(candidates: readonly JikjiCandidate[], path: string): JikjiNextRead | undefined {
	const match = candidates.find((c) => c.path === path);
	return match?.nextRead;
}

function formatDirective(policy: MergedJikjiPolicy | undefined): string {
	if (policy === undefined) {
		return "directive: jikji unavailable; use bash to explore the collection.";
	}
	if (policy.stopAfterFind) {
		return "directive: stop_after_find — answer from these paths; raw shell is disallowed.";
	}
	switch (policy.handoffAction) {
		case "direct_use":
			return "directive: direct_use — use the answer_paths directly; raw shell is disallowed.";
		case "jikji_retry":
			return "directive: jikji_retry — retry jikji_find with a refined query before falling back; raw shell is disallowed.";
		case "raw_fallback_after_retry":
			return policy.rawFallbackAllowed
				? "directive: raw fallback allowed — you may use bash to explore if the answer_paths are insufficient."
				: "directive: raw_fallback_after_retry — retry jikji_find once more; raw shell is allowed only after a second jikji_find.";
		default:
			return "directive: honor the Jikji policy.";
	}
}
