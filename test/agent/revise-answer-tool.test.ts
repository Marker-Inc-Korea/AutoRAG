import type { AgentToolResult } from "@earendil-works/pi-agent-core";
import { initTheme, type Theme, ToolExecutionComponent } from "@earendil-works/pi-coding-agent";
import chalk from "chalk";
import { describe, expect, it } from "vitest";
import { AnswerRevisionSession, type Revision, type RevisionCitations } from "../../src/agent/answer-revision.ts";
import { toToolDefinition } from "../../src/agent/pi-session.ts";
import {
	createReviseAnswerTool,
	formatRevision,
	REVISE_ANSWER_PROMPT_RULE,
	REVISE_ANSWER_TOOL_NAME,
	type ReviseAnswerDetails,
	type ReviseAnswerParams,
	type ReviseAnswerTool,
} from "../../src/agent/revise-answer-tool.ts";

const citations: RevisionCitations = {
	inspect(text) {
		const unresolved: string[] = [];
		let resolved = 0;
		for (const match of text.matchAll(/\[(e\d+|\d+)\]/gu)) {
			if (/^e[1-9]$/u.test(match[1] ?? "")) resolved += 1;
			else unresolved.push(match[0]);
		}
		return { resolved, unresolved };
	},
	display: (text) => text.replace(/\[e(\d)\]/gu, "[$1]"),
};

/**
 * Wraps every styled run in a visible tag so a test can read the styling back. `style` mirrors pi's
 * `theme.style`, the one styling call that does not depend on chalk's colour level.
 */
const tagTheme = {
	fg: (color: string, text: string) => `<${color}>${text}</${color}>`,
	style: (text: string, options: { fg?: string; bold?: boolean; strikethrough?: boolean }) => {
		let out = text;
		if (options.strikethrough) out = `<s>${out}</s>`;
		if (options.bold) out = `<b>${out}</b>`;
		return options.fg === undefined ? out : `<${options.fg}>${out}</${options.fg}>`;
	},
} as unknown as Theme;

const fast = "- Refunds need team-lead approval [e1]\n- The cap is 50만 원 [e2]";

interface Harness {
	readonly tool: ReviseAnswerTool;
	readonly announced: Revision[];
	readonly unsupportedCalls: string[];
	setSession(session: AnswerRevisionSession | undefined): void;
}

function harness(options: { unsupported?: (text: string) => readonly string[] } = {}): Harness {
	let current: AnswerRevisionSession | undefined = new AnswerRevisionSession(fast, citations);
	const announced: Revision[] = [];
	const unsupportedCalls: string[] = [];
	const tool = createReviseAnswerTool({
		session: () => current,
		announce: (revision) => announced.push(revision),
		unsupported: async (text) => {
			unsupportedCalls.push(text);
			return options.unsupported?.(text) ?? [];
		},
	});
	return {
		tool,
		announced,
		unsupportedCalls,
		setSession: (session) => {
			current = session;
		},
	};
}

function textOf(result: AgentToolResult<ReviseAnswerDetails>): string {
	return result.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

function run(h: Harness, params: ReviseAnswerParams) {
	return h.tool.execute("call-1", params);
}

describe("revise_answer tool", () => {
	it("tells the model that wording, style and translation choices are not corrections", () => {
		const tool = harness().tool;
		for (const text of [tool.description, REVISE_ANSWER_PROMPT_RULE]) {
			expect(text).toMatch(/wording|rephras/i);
			expect(text).toMatch(/not (a|for) /i);
		}
		expect(REVISE_ANSWER_PROMPT_RULE).toMatch(/same fact/i);
	});

	it("tells the model an addition must be a fact the first answer does not already state", () => {
		const tool = harness().tool;
		for (const text of [tool.description, REVISE_ANSWER_PROMPT_RULE]) {
			expect(text).toMatch(/already (state|say|contain|in)/i);
		}
		// The model reads the serialized schema, so that is what is asserted on.
		expect(JSON.stringify(tool.parameters)).toMatch(/does not already state/i);
	});

	it("is named revise_answer", () => {
		expect(REVISE_ANSWER_TOOL_NAME).toBe("revise_answer");
		expect(harness().tool.name).toBe("revise_answer");
	});

	it("applies a replacement, announces it once, and tells the model what is left unreviewed", async () => {
		const h = harness();
		const result = await run(h, {
			op: "replace",
			claimId: "c1",
			text: "- Refunds need director approval [e3]",
			reason: "July policy",
		});
		expect(result.isError).not.toBe(true);
		expect(h.announced).toEqual([
			{
				op: "replace",
				claimId: "c1",
				original: "- Refunds need team-lead approval [1]",
				replacement: "- Refunds need director approval [3]",
				reason: "July policy",
			},
		]);
		// The terminal shows the first answer with the model's own [eN] citations, so its view is the claim as written.
		expect(result.details).toEqual({
			revision: {
				op: "replace",
				claimId: "c1",
				original: "- Refunds need team-lead approval [e1]",
				replacement: "- Refunds need director approval [e3]",
				reason: "July policy",
			},
		});
		expect(textOf(result)).toContain("c2");
	});

	it("returns a rejected revision as a tool error the model can fix, and announces nothing", async () => {
		const h = harness();
		const result = await run(h, { op: "retract", claimId: "c9", reason: "x" });
		expect(result.isError).toBe(true);
		expect(textOf(result)).toContain("c1, c2");
		expect(h.announced).toEqual([]);
	});

	it("refuses when no first answer reached the user", async () => {
		const h = harness();
		h.setSession(undefined);
		const result = await run(h, { op: "retract", claimId: "c1", reason: "x" });
		expect(result.isError).toBe(true);
		expect(textOf(result)).toMatch(/first answer/i);
		expect(h.announced).toEqual([]);
	});

	it("asks the support check about a replacement or an addition, and rejects what it names", async () => {
		const h = harness({ unsupported: () => ["/policy/refunds.txt"] });
		const replaced = await run(h, { op: "replace", claimId: "c1", text: "- Directors approve [e3]", reason: "x" });
		expect(replaced.isError).toBe(true);
		expect(textOf(replaced)).toContain("/policy/refunds.txt");
		const added = await run(h, { op: "add", text: "- Logged [e4]", reason: "x" });
		expect(added.isError).toBe(true);
		expect(h.unsupportedCalls).toEqual(["- Directors approve [e3]", "- Logged [e4]"]);
		expect(h.announced).toEqual([]);
		// The rejected replacement did not use up the claim: it can still be corrected.
		h.tool.execute("call-2", { op: "retract", claimId: "c1", reason: "wrong" });
	});

	it("skips the support check for ops that add no new claim, and for a replacement the session already rejects", async () => {
		const h = harness();
		await run(h, { op: "retract", claimId: "c1", reason: "x" });
		await run(h, { op: "confirm", claimId: "c2", reason: "x" });
		await run(h, { op: "replace", claimId: "c2", text: "no citation at all", reason: "x" });
		expect(h.unsupportedCalls).toEqual([]);
	});

	it("runs one call at a time, so two corrections of one claim cannot race the support check", () => {
		expect(harness().tool.executionMode).toBe("sequential");
	});

	it("is handed to pi with its terminal renderers", () => {
		const definition = toToolDefinition(harness().tool);
		expect(definition.renderCall).toBeTypeOf("function");
		expect(definition.renderResult).toBeTypeOf("function");
	});
});

describe("formatRevision (the terminal view of a correction)", () => {
	it("shows a replacement as the struck original above the new text, with the reason", () => {
		const lines = formatRevision(
			{
				op: "replace",
				claimId: "c1",
				original: "- Refunds need team-lead approval [1]",
				replacement: "- Refunds need director approval [3]",
				reason: "July policy",
			},
			tagTheme,
		);
		expect(lines).toEqual([
			"<warning><b>✎ Corrected c1</b></warning> <muted>— July policy</muted>",
			"<error>− </error><error><s>- Refunds need team-lead approval [1]</s></error>",
			"<success>+ </success><success>- Refunds need director approval [3]</success>",
		]);
	});

	it("shows a retraction as the struck original alone", () => {
		const lines = formatRevision(
			{ op: "retract", claimId: "c2", original: "The cap is 50만 원 [2]", reason: "no source" },
			tagTheme,
		);
		expect(lines).toEqual([
			"<warning><b>✎ Retracted c2</b></warning> <muted>— no source</muted>",
			"<error>− </error><error><s>The cap is 50만 원 [2]</s></error>",
		]);
	});

	it("shows an addition as new text only", () => {
		const lines = formatRevision(
			{ op: "add", replacement: "Exceptions are logged [4]", reason: "audit policy" },
			tagTheme,
		);
		expect(lines).toEqual([
			"<warning><b>✎ Added</b></warning> <muted>— audit policy</muted>",
			"<success>+ </success><success>Exceptions are logged [4]</success>",
		]);
	});

	it("shows a confirmation as one quiet line", () => {
		const lines = formatRevision(
			{ op: "confirm", claimId: "c3", original: "Paid monthly [3]", reason: "payroll policy" },
			tagTheme,
		);
		expect(lines).toEqual(["<success>✓ Checked c3</success> <muted>— payroll policy</muted>"]);
	});

	it("keeps the change marks distinct from a claim's own bullet, so a struck bullet is not read as '- - '", () => {
		const lines = formatRevision(
			{
				op: "replace",
				claimId: "c4",
				original: "- Paid monthly [3]",
				replacement: "- Paid monthly, net 60 [3]",
				reason: "amendment",
			},
			tagTheme,
		);
		const plain = lines.map((line) => line.replace(/<\/?[a-z]+>/gu, ""));
		expect(plain.slice(1)).toEqual(["− - Paid monthly [3]", "+ - Paid monthly, net 60 [3]"]);
		expect(plain.join("\n")).not.toMatch(/^- - /mu);
	});

	it("marks every line of a multi-line claim, so a struck block stays struck", () => {
		const lines = formatRevision({ op: "retract", claimId: "c1", original: "- a\n  b", reason: "x" }, tagTheme);
		expect(lines.slice(1)).toEqual([
			"<error>− </error><error><s>- a</s></error>",
			"<error>− </error><error><s>  b</s></error>",
		]);
	});
});

describe("formatRevision with pi's real theme", () => {
	/** The theme pi hands a renderer: the real one, so the escape codes are what a terminal receives. */
	function realTheme(): Theme {
		initTheme("dark");
		let captured: Theme | undefined;
		const probe = {
			name: "probe",
			label: "probe",
			description: "",
			parameters: {},
			renderCall: (_args: unknown, theme: Theme) => {
				captured = theme;
				return { render: () => [""], invalidate: () => undefined };
			},
			execute: async () => ({ content: [], details: {} }),
		};
		new ToolExecutionComponent(
			"probe",
			"t",
			{},
			{},
			probe as never,
			{ requestRender: () => undefined } as never,
			process.cwd(),
		).render(40);
		if (captured === undefined) throw new Error("pi never called the renderer");
		return captured;
	}

	it("strikes the original out even when chalk's colour level is 0, as in a plain bun terminal", () => {
		const theme = realTheme();
		const level = chalk.level;
		chalk.level = 0;
		try {
			const lines = formatRevision(
				{ op: "replace", claimId: "c1", original: "old", replacement: "new", reason: "r" },
				theme,
			);
			expect(lines[1]).toContain("\x1b[9m");
			expect(lines[1]).toContain("\x1b[29m");
			expect(lines[0]).toContain("\x1b[1m");
			// The replacement is plain colour: only the original is struck.
			expect(lines[2]).not.toContain("\x1b[9m");
		} finally {
			chalk.level = level;
		}
	});
});

describe("revise_answer renderers", () => {
	const revision: Revision = {
		op: "replace",
		claimId: "c1",
		original: "old claim [1]",
		replacement: "new claim [2]",
		reason: "fresher source",
	};

	function renderedResult(result: AgentToolResult<ReviseAnswerDetails>, isError = false): string {
		const definition = toToolDefinition(harness().tool);
		const context = { isError } as never;
		const component = definition.renderResult?.(result, { expanded: false, isPartial: false }, tagTheme, context);
		return component?.render(200).join("\n") ?? "";
	}

	it("renders an applied revision as its terminal view", () => {
		const text = renderedResult({ content: [{ type: "text", text: "ok" }], details: { revision } });
		expect(text).toContain("<error><s>old claim [1]</s></error>");
		expect(text).toContain("<success>new claim [2]</success>");
		expect(text).toContain("fresher source");
	});

	it("renders a rejected revision as the tool's message, not as a correction", () => {
		const text = renderedResult({ content: [{ type: "text", text: "Unknown claim c9." }], details: {} }, true);
		expect(text).toContain("Unknown claim c9.");
		expect(text).not.toContain("Corrected");
	});

	it("names the revision in the call line before the result lands", () => {
		const definition = toToolDefinition(harness().tool);
		const component = definition.renderCall?.(
			{ op: "replace", claimId: "c1", reason: "x" } as never,
			tagTheme,
			{} as never,
		);
		expect(component?.render(200).join("\n")).toContain("replace c1");
	});
});
