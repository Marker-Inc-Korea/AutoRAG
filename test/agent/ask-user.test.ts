import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxResponseStep, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { ASK_USER_TOOL_NAME, resolveAskUserExtensionPath } from "../../src/agent/ask-user.ts";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-ask-user-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

interface Dialogs {
	readonly selects: { title: string; options: readonly string[] }[];
	readonly inputs: string[];
}

/** The interactive UI a TUI binds. `custom` yields undefined (as in RPC mode), so the tool uses select/input dialogs. */
function ui(
	dialogs: Dialogs,
	answers: { select?: string | undefined; input?: string | undefined },
): ExtensionUIContext {
	const noop = () => undefined;
	return {
		select: async (title: string, options: readonly string[]) => {
			dialogs.selects.push({ title, options });
			return answers.select;
		},
		input: async (title: string) => {
			dialogs.inputs.push(title);
			return answers.input;
		},
		confirm: async () => false,
		notify: noop,
		onTerminalInput: () => noop,
		setStatus: noop,
		setWorkingMessage: noop,
		setWorkingVisible: noop,
		setWorkingIndicator: noop,
		setHiddenThinkingLabel: noop,
		setWidget: noop,
		setFooter: noop,
		setHeader: noop,
		setTitle: noop,
		custom: async () => undefined as never,
		pasteToEditor: noop,
		setEditorText: noop,
		getEditorText: () => "",
		editor: async () => undefined,
		addAutocompleteProvider: noop,
		setEditorComponent: noop,
		getEditorComponent: noop,
		theme: { fg: (_color: string, text: string) => text, bold: (text: string) => text } as never,
		getAllThemes: () => [],
		getTheme: noop,
		setTheme: () => ({ success: false, error: "UI not available" }),
		getToolsExpanded: () => false,
		setToolsExpanded: noop,
	} as unknown as ExtensionUIContext;
}

const askThenSay = (args: Parameters<typeof fauxToolCall>[1]): FauxResponseStep[] => [
	fauxAssistantMessage([fauxToolCall(ASK_USER_TOOL_NAME, args)], { stopReason: "toolUse" }),
	fauxAssistantMessage("noted", { stopReason: "stop" }),
	// The first question of a turn runs a second, verification phase.
	fauxAssistantMessage("verified", { stopReason: "stop" }),
];

async function tui(
	responses: FauxResponseStep[],
	answers: { select?: string | undefined; input?: string | undefined } = {},
	options: Partial<AutoRAGAgentOptions> = {},
) {
	const registration = registerFauxProvider({ api: `faux-ask-${randomUUID()}`, models: [{ id: "ask" }] });
	registration.setResponses(responses);
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	const model = registration.getModel();
	writeFileSync(
		join(agentDir, "settings.json"),
		JSON.stringify({ defaultProvider: model.provider, defaultModel: model.id }),
	);
	const agent = new AutoRAGAgent({
		model,
		apiKey: "test-key",
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: false,
		...options,
	});
	const hosted = await agent.createPiInteractiveRuntime();
	const dialogs: Dialogs = { selects: [], inputs: [] };
	await hosted.runtime.session.bindExtensions({ uiContext: ui(dialogs, answers) });
	return {
		agent,
		session: hosted.runtime.session,
		dialogs,
		dispose: async () => {
			await hosted.dispose();
			registration.unregister();
		},
	};
}

/** The text of the first `ask_user` tool result in the finished session. */
function askResult(messages: readonly { role: string; toolName?: string; content?: unknown; isError?: boolean }[]) {
	const result = messages.find((message) => message.role === "toolResult" && message.toolName === ASK_USER_TOOL_NAME);
	const text = Array.isArray(result?.content)
		? (result.content as { type: string; text?: string }[]).map((part) => part.text ?? "").join("")
		: "";
	return { text, isError: result?.isError === true, found: result !== undefined };
}

describe("ask_user in the interactive session", () => {
	it("resolves the pi-ask-user extension entry file", () => {
		expect(existsSync(resolveAskUserExtensionPath())).toBe(true);
	});

	it("is active for the model and described in the system prompt", async () => {
		const t = await tui([]);
		try {
			expect(t.session.getActiveToolNames()).toContain(ASK_USER_TOOL_NAME);
			expect(t.session.systemPrompt).toContain(`**${ASK_USER_TOOL_NAME}**`);
		} finally {
			await t.dispose();
		}
	});

	it("tells the model the phase prompts' no-tools rule does not cover ask_user, so it asks instead of guessing", async () => {
		const t = await tui([]);
		try {
			expect(t.agent.buildFastAnswerPrompt("q", {}, "baseline")).toContain(ASK_USER_TOOL_NAME);
		} finally {
			await t.dispose();
		}
	});

	it("does not mention ask_user in a phase prompt when the tool is not offered", async () => {
		const t = await tui([], {}, { discordRemote: true });
		try {
			expect(t.agent.buildFastAnswerPrompt("q", {}, "baseline")).not.toContain(ASK_USER_TOOL_NAME);
		} finally {
			await t.dispose();
		}
	});

	it("shows a multiple-choice question and returns the picked option to the model", async () => {
		const t = await tui(
			askThenSay({
				question: "Which report do you mean?",
				options: [{ title: "Q1 report" }, { title: "Q2 report", description: "newer" }],
				allowFreeform: false,
			}),
			{ select: "Q2 report" },
		);
		try {
			await t.session.prompt("Summarize the report", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(t.dialogs.selects).toHaveLength(1);
			expect(t.dialogs.selects[0]?.title).toContain("Which report do you mean?");
			expect(t.dialogs.selects[0]?.options).toEqual(["Q1 report", "Q2 report"]);
			const result = askResult(t.session.messages as never);
			expect(result.isError).toBe(false);
			expect(result.text).toBe("User answered: Q2 report");
		} finally {
			await t.dispose();
		}
	});

	it("asks an open question and returns the typed answer", async () => {
		const t = await tui(askThenSay({ question: "What is the project codename?" }), { input: "Bluebird" });
		try {
			await t.session.prompt("Find the codename docs", { source: "interactive" });
			await t.agent.waitForIdle();
			expect(t.dialogs.inputs).toHaveLength(1);
			expect(t.dialogs.inputs[0]).toContain("What is the project codename?");
			expect(askResult(t.session.messages as never).text).toBe("User answered: Bluebird");
		} finally {
			await t.dispose();
		}
	});

	it("tells the model the user cancelled instead of failing the turn", async () => {
		const t = await tui(askThenSay({ question: "Which one?", options: [{ title: "A" }, { title: "B" }] }), {
			select: undefined,
		});
		try {
			await t.session.prompt("pick", { source: "interactive" });
			await t.agent.waitForIdle();
			const result = askResult(t.session.messages as never);
			expect(result.isError).toBe(false);
			expect(result.text).toBe("User cancelled the question");
		} finally {
			await t.dispose();
		}
	});

	it("is left out of a session driven from a chat bridge, where nobody sits at the terminal", async () => {
		const t = await tui([], {}, { discordRemote: true });
		try {
			expect(t.session.getActiveToolNames()).not.toContain(ASK_USER_TOOL_NAME);
			expect(t.session.systemPrompt).not.toContain(`**${ASK_USER_TOOL_NAME}**`);
		} finally {
			await t.dispose();
		}
	});
});
