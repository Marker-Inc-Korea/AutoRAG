/**
 * Live E2E: a real model decides to ask the person at the terminal, and uses the answer.
 *
 * Gated so the offline suite never calls a model or spends credits:
 *
 *   AUTORAG_ASK_USER_LIVE=1 [AUTORAG_ASK_USER_LIVE_MODEL=anthropic/claude-haiku-5.5] OPENROUTER_API_KEY=... \
 *     bunx vitest run test/live-e2e/ask-user.test.ts
 *
 * Acceptance bar: given a request that cannot be done without a choice only the user can make, the real model
 * (1) calls `ask_user` with real options instead of guessing, (2) receives the option the user picks and
 * builds on it, and (3) given a free-text question, receives the typed answer verbatim. The librarian, pi
 * session, the pi-ask-user extension and the model are real; only the terminal dialogs are scripted.
 *
 * The model must be a 2026+ release (see the global model-usage rule).
 */
import { mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionUIContext } from "@earendil-works/pi-coding-agent";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { ASK_USER_TOOL_NAME } from "../../src/agent/ask-user.ts";
import { resolveAgentModel, resolveConfig } from "../../src/cli/config.ts";

const LIVE = process.env.AUTORAG_ASK_USER_LIVE === "1" && Boolean(process.env.OPENROUTER_API_KEY);
const MODEL_ID = process.env.AUTORAG_ASK_USER_LIVE_MODEL ?? "anthropic/claude-haiku-5.5";

let root: string;

beforeEach(() => {
	root = realpathSync(mkdtempSync(join(tmpdir(), "autorag-ask-user-live-")));
});

afterEach(() => {
	rmSync(root, { recursive: true, force: true });
});

interface Scripted {
	/** What `ui.select` answers: the option title to pick, or undefined to cancel. */
	readonly pick?: (options: readonly string[]) => string | undefined;
	readonly typed?: string;
}

async function session(script: Scripted) {
	const agentDir = join(root, "agent");
	mkdirSync(agentDir, { recursive: true });
	process.env.PI_CODING_AGENT_DIR = agentDir;
	const docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	writeFileSync(join(root, "config.json"), JSON.stringify({ searchPaths: [docs] }));
	const config = resolveConfig({
		flags: { "model-provider": "openrouter", "model-id": MODEL_ID },
		env: { ...process.env, AUTORAG_CONFIG: join(root, "config.json") },
		cwd: root,
		readOnly: true,
	});
	const resolved = await resolveAgentModel(config);
	const agent = new AutoRAGAgent({
		model: resolved.model,
		...(resolved.apiKey !== undefined ? { apiKey: resolved.apiKey } : {}),
		...(resolved.providerApiKeys !== undefined ? { providerApiKeys: resolved.providerApiKeys } : {}),
		searchPaths: [docs],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
		webSearch: false,
		piAgentDir: agentDir,
		piSessionDir: join(root, "sessions"),
		jev: false,
	});
	const hosted = await agent.createPiInteractiveRuntime();
	const shown: { question: string; options: readonly string[] }[] = [];
	const noop = () => undefined;
	await hosted.runtime.session.bindExtensions({
		uiContext: {
			// `custom` yields undefined, as in RPC mode, so pi-ask-user falls back to select/input dialogs.
			custom: async () => undefined as never,
			select: async (title: string, options: readonly string[]) => {
				shown.push({ question: title, options });
				return script.pick?.(options);
			},
			input: async (title: string) => {
				shown.push({ question: title, options: [] });
				return script.typed;
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
			theme: { fg: (_color: string, text: string) => text, bold: (text: string) => text },
		} as unknown as ExtensionUIContext,
	});
	return {
		agent,
		session: hosted.runtime.session,
		shown,
		dispose: () => hosted.dispose(),
	};
}

/** Every assistant text block in the finished session, joined. */
function assistantText(messages: readonly { role: string; content?: unknown }[]): string {
	return messages
		.filter((message) => message.role === "assistant")
		.flatMap((message) =>
			Array.isArray(message.content) ? (message.content as { type: string; text?: string }[]) : [],
		)
		.filter((block) => block.type === "text")
		.map((block) => block.text ?? "")
		.join("\n");
}

function askCalls(messages: readonly { role: string; toolName?: string; content?: unknown }[]): string[] {
	return messages
		.filter((message) => message.role === "toolResult" && message.toolName === ASK_USER_TOOL_NAME)
		.map((message) =>
			(message.content as { type: string; text?: string }[]).map((block) => block.text ?? "").join(""),
		);
}

describe.skipIf(!LIVE)("ask_user with a live model", () => {
	it("asks a multiple-choice question for a decision only the user can make, then builds on the pick", async () => {
		const t = await session({ pick: (options) => options.find((option) => /final/i.test(option)) ?? options[0] });
		try {
			await t.session.prompt(
				"Pick a file format for my quarterly report. I have not decided between a draft and a final version " +
					"and you cannot know which I want. Ask me which one before you say anything else, with the two " +
					"choices 'Draft' and 'Final', then tell me in one sentence which one I chose.",
				{ source: "interactive" },
			);
			await t.agent.waitForIdle();
			const messages = t.session.messages as never as { role: string; toolName?: string; content?: unknown }[];
			expect(t.shown.length, "the model must ask instead of guessing").toBeGreaterThanOrEqual(1);
			expect(t.shown[0]?.options.some((option) => /final/i.test(option))).toBe(true);
			expect(askCalls(messages)[0]).toMatch(/^User answered: .*final/i);
			expect(assistantText(messages)).toMatch(/final/i);
		} finally {
			await t.dispose();
		}
	}, 300_000);

	it("hands the model a typed answer verbatim", async () => {
		const t = await session({ typed: "zebra-4417" });
		try {
			await t.session.prompt(
				"Ask me, as an open question with no options, for my project's secret codename. " +
					"Then repeat the codename I give you exactly.",
				{ source: "interactive" },
			);
			await t.agent.waitForIdle();
			const messages = t.session.messages as never as { role: string; toolName?: string; content?: unknown }[];
			expect(askCalls(messages)).toEqual(["User answered: zebra-4417"]);
			expect(assistantText(messages)).toContain("zebra-4417");
		} finally {
			await t.dispose();
		}
	}, 300_000);

	it("treats a cancelled question as no answer rather than inventing one", async () => {
		const t = await session({ pick: () => undefined });
		try {
			await t.session.prompt(
				"Ask me which city to plan a trip to, with options 'Seoul' and 'Busan'. If I do not answer, say " +
					"exactly 'NO ANSWER' and nothing else.",
				{ source: "interactive" },
			);
			await t.agent.waitForIdle();
			const messages = t.session.messages as never as { role: string; toolName?: string; content?: unknown }[];
			expect(askCalls(messages)).toEqual(["User cancelled the question"]);
			expect(assistantText(messages)).toMatch(/NO ANSWER/i);
		} finally {
			await t.dispose();
		}
	}, 300_000);
});
