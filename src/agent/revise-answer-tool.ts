import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import type { Theme } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { type Static, Type } from "typebox";
import type { AnswerRevisionSession, Revision } from "./answer-revision.ts";
import type { TerminalTool } from "./pi-session.ts";

export const REVISE_ANSWER_TOOL_NAME = "revise_answer";

const reviseAnswerSchema = Type.Object({
	op: Type.Union([Type.Literal("confirm"), Type.Literal("retract"), Type.Literal("replace"), Type.Literal("add")], {
		description:
			"confirm: the claim is right as written. retract: the claim is wrong or unsupported; it is struck out. replace: the claim is wrong and this is the corrected claim. add: a new verified finding the first answer does not already state; never a source, date or explanation for a claim it already makes.",
	}),
	claimId: Type.Optional(
		Type.String({
			description: "confirm, retract, replace: the id of the first-answer claim (c1, c2, ...). Not for add.",
		}),
	),
	text: Type.Optional(
		Type.String({
			description:
				"replace, add: the corrected claim or the new finding, written like a first-answer claim and citing the evidence that supports it with its [eN] id. Not for confirm or retract.",
		}),
	),
	reason: Type.String({
		description:
			"One short line, in the user's language, saying why: the user reads it next to the change (for example the newer source that contradicts the claim).",
	}),
});

export type ReviseAnswerParams = Static<typeof reviseAnswerSchema>;

export interface ReviseAnswerDetails {
	/**
	 * The applied revision as the terminal shows it: the claim and its replacement as the model wrote them, with the
	 * model's own `[eN]` citations, which are what the first answer carried in the transcript. Absent when the call
	 * was rejected.
	 */
	readonly revision?: Revision;
}

export type ReviseAnswerTool = TerminalTool<typeof reviseAnswerSchema, ReviseAnswerDetails>;

export interface ReviseAnswerToolOptions {
	/** The revision session of the first answer the user was shown; undefined when none reached the user. */
	readonly session: () => AnswerRevisionSession | undefined;
	/** Called once for every applied revision. */
	readonly announce: (revision: Revision) => void;
	/** Sources among the text's citations that do not back the text. Empty when they all do. */
	readonly unsupported: (text: string) => Promise<readonly string[]>;
}

/** Concise verification-turn rule for correcting an already-delivered first answer. */
export const REVISE_ANSWER_PROMPT_RULE = `Apply corrections to the first answer via ${REVISE_ANSWER_TOOL_NAME} (one claim at a time) rather than rewriting it in your reply: retract or replace only factually wrong, unsupported, or outdated claims, and add only genuinely new verified findings (not extra citations or phrasing tweaks for claims already stated).`;
function reply(text: string, details: ReviseAnswerDetails, isError = false): AgentToolResult<ReviseAnswerDetails> {
	return { content: [{ type: "text", text }], details, ...(isError ? { isError: true } : {}) };
}

/**
 * The model's way to correct an answer the user has already read. The harness keeps the claims of that answer, so a
 * revision names the claim it changes and the user is shown the text they actually read struck out, not the model's
 * retelling of it. A rejected revision comes back as a tool error the model can fix in the same turn.
 */
export function createReviseAnswerTool(options: ReviseAnswerToolOptions): ReviseAnswerTool {
	return {
		name: REVISE_ANSWER_TOOL_NAME,
		label: "Revise answer",
		description:
			"Correct the first answer the user has already read, one claim at a time. The first answer is split into claims c1, c2, ...; name the claim you change. The user sees the original struck through, the corrected text, and your reason. Use it only for a claim that is factually wrong, unsupported or outdated, or for a new verified finding the first answer does not already state. Wording, style, number format and translation choices are not a correction: do not use it to rephrase a claim that says the right thing, and do not restate claims that are right.",
		parameters: reviseAnswerSchema,
		executionMode: "sequential",
		renderCall: (args, theme) =>
			new Text(
				`${theme.style(REVISE_ANSWER_TOOL_NAME, { fg: "toolTitle", bold: true })} ${theme.fg("muted", `${args.op ?? ""}${args.claimId === undefined ? "" : ` ${args.claimId}`}`)}`,
				0,
				0,
			),
		renderResult: (result, _options, theme, context) => {
			const revision = result.details?.revision;
			if (context.isError || revision === undefined) {
				const message = result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
				return new Text(theme.fg(context.isError ? "error" : "muted", message), 0, 0);
			}
			return new Text(formatRevision(revision, theme).join("\n"), 0, 0);
		},
		async execute(_toolCallId, params) {
			const session = options.session();
			if (session === undefined) {
				return reply(
					"There is no first answer to revise: none reached the user. Write the complete answer as your reply.",
					{},
					true,
				);
			}
			const request = {
				op: params.op,
				reason: params.reason,
				...(params.claimId === undefined ? {} : { claimId: params.claimId }),
				...(params.text === undefined ? {} : { text: params.text }),
			};
			const checked = session.check(request);
			if (!checked.ok) return reply(checked.message, {}, true);
			if (params.text !== undefined && (params.op === "replace" || params.op === "add")) {
				const unsupported = await options.unsupported(params.text);
				if (unsupported.length > 0) {
					return reply(
						`The cited evidence does not support this text: ${unsupported.join(", ")}. Cite evidence that does, or retract the claim instead.`,
						{},
						true,
					);
				}
			}
			const applied = session.apply(request);
			if (!applied.ok) return reply(applied.message, {}, true);
			options.announce(applied.revision);
			const left = session.unreviewed();
			return reply(
				`Applied (${applied.revision.op}${applied.revision.claimId === undefined ? "" : ` ${applied.revision.claimId}`}). ${
					left.length === 0 ? "Every claim has been reviewed." : `Not yet reviewed: ${left.join(", ")}.`
				}`,
				{ revision: { ...applied.revision, ...applied.written } },
			);
		},
	};
}

function markLines(
	text: string,
	mark: string,
	style: (line: string) => string,
	styleMark: (mark: string) => string,
): string[] {
	return text.split("\n").map((line) => styleMark(mark) + style(line));
}

/**
 * A revision as terminal lines: a headline with the reason, the original struck out in red, and the new text in
 * green. Every line of a multi-line claim carries its own mark and styling, so a struck block stays struck.
 */
export function formatRevision(revision: Revision, theme: Theme): string[] {
	const reason = revision.reason === "" ? "" : ` ${theme.fg("muted", `— ${revision.reason}`)}`;
	const subject = revision.claimId === undefined ? "" : ` ${revision.claimId}`;
	if (revision.op === "confirm") return [`${theme.fg("success", `✓ Checked${subject}`)}${reason}`];
	const headline = { replace: "Corrected", retract: "Retracted", add: "Added" }[revision.op];
	const lines = [`${theme.style(`✎ ${headline}${subject}`, { fg: "warning", bold: true })}${reason}`];
	if (revision.original !== undefined) {
		lines.push(
			...markLines(
				revision.original,
				// U+2212, not "-": a claim that is a bullet starts with "- " itself, and "- - " reads as a typo.
				"− ",
				// `style`, not `theme.strikethrough`: that one goes through chalk, which prints nothing at colour level 0.
				(line) => theme.style(line, { fg: "error", strikethrough: true }),
				(mark) => theme.fg("error", mark),
			),
		);
	}
	if (revision.replacement !== undefined) {
		lines.push(
			...markLines(
				revision.replacement,
				"+ ",
				(line) => theme.fg("success", line),
				(mark) => theme.fg("success", mark),
			),
		);
	}
	return lines;
}
