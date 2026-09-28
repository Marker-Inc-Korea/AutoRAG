import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { AgentTool, Skill } from "@earendil-works/pi-agent-core";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { Type } from "typebox";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
	AutoRAGAgent,
	type AutoRAGAgentOptions,
	type AutoRAGChatSessionOptions,
	type AutoRAGPersonaOptions,
} from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { buildSystemPrompt, type SystemPromptConfig } from "../../src/agent/system-prompt.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-persona-chat-"));
	docs = join(root, "docs");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(
		join(docs, "refund-policy.txt"),
		[
			"Refund exceptions require director approval before payout.",
			"Finance acknowledged the policy in the July review.",
		].join("\n"),
	);
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(reasoning: boolean, ...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({
		api: `faux-${randomUUID()}`,
		models: [{ id: "faux-model", reasoning }],
	});
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

function stopText(text: string): FauxResponseStep {
	return fauxAssistantMessage(text, { stopReason: "stop" });
}

function finalEmitCall(answer: string): FauxResponseStep {
	return fauxAssistantMessage(
		[
			fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
				answer,
				results: [
					{
						number: 1,
						title: "Verified refund approval rule",
						summary: "Verified: refund exceptions require director approval before payout.",
						evidence: [{ excerpt: "Refund exceptions require director approval before payout.", lineNumber: 1 }],
						confidence: 0.95,
					},
				],
				mapping: [
					{
						number: 1,
						source: join(docs, "refund-policy.txt"),
						method: "bash",
						content: "Refund exceptions require director approval before payout.",
					},
				],
			}),
		],
		{ stopReason: "toolUse" },
	);
}

function agentOptions(model: ReturnType<typeof fauxModel>): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: false,
		jikji: false,
	};
}

function memoryEntryCount(path: string): number {
	if (!existsSync(path)) return 0;
	const data = JSON.parse(readFileSync(path, "utf8")) as Record<string, unknown>;
	const arrays = ["curatedResults", "evidenceChunks", "feedbackSignals", "insights", "pendingInsightSignals"];
	return arrays.reduce((sum, key) => sum + (Array.isArray(data[key]) ? (data[key] as unknown[]).length : 0), 0);
}

function makeFakeTool(): AgentTool {
	return {
		name: "fake_tool",
		label: "fake_tool",
		description: "Fake tool for chat-session tests",
		parameters: Type.Object({}),
		async execute() {
			return { content: [{ type: "text", text: "ok" }], details: {} };
		},
	};
}

const PINNED_SKILL: Skill = {
	name: "settings-assistant",
	description: "Guides application settings changes.",
	filePath: "/tmp/skills/settings-assistant/SKILL.md",
	content: "FULL SKILL BODY: always confirm before changing a setting.",
};

describe("persona system prompt composition", () => {
	it("keeps the librarian prompt byte-identical when no persona is configured", () => {
		const agent = new AutoRAGAgent(agentOptions(fauxModel(false)));
		const internals = agent as unknown as { currentSystemPromptConfig(): SystemPromptConfig };
		expect(agent.getSystemPrompt()).toBe(buildSystemPrompt(internals.currentSystemPromptConfig()));
		expect(agent.getSystemPrompt()).toContain("## Search Strategy");
	});

	it("replaces the librarian prompt verbatim with a persona string", () => {
		const persona: AutoRAGPersonaOptions = { systemPrompt: "You are the settings assistant." };
		const agent = new AutoRAGAgent({ ...agentOptions(fauxModel(false)), persona });
		expect(agent.getSystemPrompt()).toBe("You are the settings assistant.");
		expect(agent.getSystemPrompt()).not.toContain("## Search Strategy");
	});

	it("passes the live config to a persona function that can extend the librarian prompt", () => {
		let received: SystemPromptConfig | undefined;
		const persona: AutoRAGPersonaOptions = {
			systemPrompt: (config) => {
				received = config;
				return `${buildSystemPrompt(config)}\n\n## Persona Addendum\nSettings mode is active.`;
			},
		};
		const agent = new AutoRAGAgent({ ...agentOptions(fauxModel(false)), persona });
		expect(received).toBeDefined();
		expect(received?.toolNames).toContain("bash");
		expect(received?.toolNames).toContain("search_all_documents");
		expect(received?.toolNames).toContain("check_memory");
		expect(agent.getSystemPrompt()).toContain("## Search Strategy");
		expect(agent.getSystemPrompt()).toContain("## Persona Addendum");
	});

	it("appends pinned skills to the constructor prompt and the per-search session prompt", async () => {
		let markReady: (() => void) | undefined;
		let release: (() => void) | undefined;
		const sessionReady = new Promise<void>((resolve) => {
			markReady = resolve;
		});
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		const gatedFinalEmit: FauxResponseStep = async () => {
			markReady?.();
			await gate;
			return finalEmitCall("Final answer: refund exceptions require director approval.") as ReturnType<
				typeof fauxAssistantMessage
			>;
		};
		const persona: AutoRAGPersonaOptions = { systemPrompt: "Persona base prompt.", skills: [PINNED_SKILL] };
		const agent = new AutoRAGAgent({ ...agentOptions(fauxModel(true, gatedFinalEmit)), persona });

		const constructorPrompt = agent.getSystemPrompt();
		expect(constructorPrompt).toContain("Persona base prompt.");
		expect(constructorPrompt).toContain("## Always-Loaded Skills");
		expect(constructorPrompt).toContain("settings-assistant");
		expect(constructorPrompt).toContain("FULL SKILL BODY: always confirm before changing a setting.");

		const searching = agent.searchDocuments("refund approval");
		await sessionReady;
		try {
			const session = (agent as unknown as { activeSession?: { agent: { state: { systemPrompt: string } } } })
				.activeSession;
			expect(session?.agent.state.systemPrompt).toContain("Persona base prompt.");
			expect(session?.agent.state.systemPrompt).toContain("## Always-Loaded Skills");
			expect(session?.agent.state.systemPrompt).toContain("settings-assistant");
			expect(session?.agent.state.systemPrompt).toContain(
				"FULL SKILL BODY: always confirm before changing a setting.",
			);
		} finally {
			release?.();
		}
		await searching;
	});
});

describe("createChatSession", () => {
	it("runs chat turns with exactly the given tools and never touches retrieval memory", async () => {
		const model = fauxModel(false, stopText("Chat reply one."), stopText("Chat reply two."));
		const memoryPath = join(root, "chat-memory.json");
		const agent = new AutoRAGAgent({ ...agentOptions(model), memoryPath });
		const fakeTool = makeFakeTool();
		const options: AutoRAGChatSessionOptions = { tools: [fakeTool] };

		const session = agent.createChatSession(options);

		expect(session.agent.state.tools).toHaveLength(1);
		expect(session.agent.state.tools[0]).toBe(fakeTool);

		await session.prompt("hello");
		await session.prompt("again");

		const roles = session.agent.state.messages
			.map((message) => message.role)
			.filter((role) => role === "user" || role === "assistant");
		expect(roles).toEqual(["user", "assistant", "user", "assistant"]);
		expect(memoryEntryCount(memoryPath)).toBe(0);
	});

	it("constructs with empty searchPaths and resolves getRefreshStatus", async () => {
		const agent = new AutoRAGAgent({
			searchPaths: [],
			workspacePath: root,
			minSync: false,
			jikji: false,
			dupey: false,
			webSearch: false,
			memoryPath: join(root, "empty-memory.json"),
		});
		await expect(agent.getRefreshStatus()).resolves.toMatchObject({ state: "idle" });
	});

	it("keeps two instances in one process writing only to their own memory files", async () => {
		const modelA = fauxModel(
			false,
			stopText("alpha chat reply one"),
			finalEmitCall("Alpha final answer."),
			stopText("alpha chat reply two"),
		);
		const modelB = fauxModel(
			false,
			stopText("bravo chat reply one"),
			stopText("bravo chat reply two"),
			finalEmitCall("Bravo final answer."),
		);
		const memoryPathA = join(root, "memory-a.json");
		const memoryPathB = join(root, "memory-b.json");
		const agentA = new AutoRAGAgent({ ...agentOptions(modelA), memoryPath: memoryPathA });
		const agentB = new AutoRAGAgent({ ...agentOptions(modelB), memoryPath: memoryPathB });
		const chatA = agentA.createChatSession();
		const chatB = agentB.createChatSession();

		await chatA.prompt("alpha chat one");
		await chatB.prompt("bravo chat one");
		await agentA.searchDocuments("alpha-token refund approval");
		await chatB.prompt("bravo chat two");
		await agentB.searchDocuments("bravo-token refund approval");
		await chatA.prompt("alpha chat two");

		const memoryA = readFileSync(memoryPathA, "utf8");
		const memoryB = readFileSync(memoryPathB, "utf8");
		expect(memoryA).toContain("alpha-token");
		expect(memoryA).not.toContain("bravo-token");
		expect(memoryB).toContain("bravo-token");
		expect(memoryB).not.toContain("alpha-token");
	});
});
