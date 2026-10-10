import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type AssistantMessage,
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
	type TranscriptContext,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { type JevBackend, MockBackend } from "jev-use";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";
import { EVIDENCE_QUESTION_ID_PREFIX } from "../../src/agent/evidence-judgment.ts";
import { EMIT_FAST_ANSWER_TOOL_NAME } from "../../src/agent/fast-answer-tool.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
} from "../../src/agent/query-routing.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

const SKILL_MARKER = "SETUP-SKILL-FULL-TEXT-MARKER";

let root: string;
let docs: string;
let configPath: string;
let skillPath: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-self-config-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
	configPath = join(root, "config.json");
	writeFileSync(configPath, `${JSON.stringify({ model: { provider: "openrouter", id: "old/model" } }, null, 2)}\n`);
	skillPath = join(root, "SKILL.md");
	writeFileSync(
		skillPath,
		`---\nname: autorag-setup\ndescription: test\n---\n\n# AutoRAG setup\n\n${SKILL_MARKER}\n\nSecond paragraph of the skill.\n`,
	);
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model" }] });
	registration.setResponses(responses);
	registrations.push(registration);
	return registration.getModel();
}

function lastUserText(context: TranscriptContext): string {
	const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
	if (lastUser === undefined) return "";
	return typeof lastUser.content === "string"
		? lastUser.content
		: lastUser.content.map((part) => (part.type === "text" ? part.text : "")).join("");
}

interface Seen {
	prompts: string[];
	toolNames: string[][];
}

/** Replays system-message tool declarations in order, as providers do, to get the tools the model sees. */
function visibleToolNames(context: TranscriptContext): string[] {
	const names = new Set<string>();
	for (const message of context.messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) names.add(tool.name);
		for (const reference of message.toolsRemoved ?? []) names.delete(reference.name);
	}
	return [...names];
}

function capture(step: FauxResponseStep, seen: Seen): FauxResponseStep {
	return (context) => {
		seen.prompts.push(lastUserText(context));
		seen.toolNames.push(visibleToolNames(context));
		return step as AssistantMessage;
	};
}

function emitConfigReport(answer: string): FauxResponseStep {
	return fauxAssistantMessage([fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, { answer, results: [], mapping: [] })], {
		stopReason: "toolUse",
	});
}

function jevRouting(route: QueryRoute | "config"): MockBackend {
	const distribution = { local: 0.05, web: 0.05, direct: 0.05, config: 0.05, [route]: 0.85 };
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: { answer: route, distribution, confidence: 0.9 },
		[DECOMPOSE_QUESTION_ID]: { answer: 0.9 },
		[FOLLOW_UP_QUESTION_ID]: { answer: 0.9 },
	});
}

function agentWith(options: Partial<AutoRAGAgentOptions> & Pick<AutoRAGAgentOptions, "model">): AutoRAGAgent {
	return new AutoRAGAgent({
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		piAgentDir: join(root, "pi-agent"),
		minSync: false,
		jikji: false,
		webSearch: false,
		...options,
	});
}

function recordingMinSync() {
	const queries: string[] = [];
	return {
		queries,
		method: {
			isReady: () => true,
			isBinaryMissing: () => false,
			retrieve: async (query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> => {
				queries.push(query);
				return [];
			},
		},
	};
}

async function collect(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

describe("Jev config branch: the agent configures itself", () => {
	it("loads the full setup skill, skips emit_fast_answer, and reports through emit_autorag_results", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(capture(emitConfigReport("- Default model is now `new/model`."), seen));
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath },
		});
		const minSync = recordingMinSync();
		// minSyncMethod is the private seam the sibling Jev pipeline tests inject through.
		Object.assign(agent, { minSyncMethod: minSync.method });

		const events = await collect(agent, "change the default model to new/model");

		expect(seen.prompts).toHaveLength(1);
		expect(seen.prompts[0]).toContain(SKILL_MARKER);
		expect(seen.prompts[0]).toContain("Second paragraph of the skill.");
		expect(seen.prompts[0]).not.toContain("description: test");
		expect(seen.prompts[0]).toContain(configPath);
		expect(seen.prompts[0]).toContain("change the default model to new/model");
		expect(seen.toolNames[0]).not.toContain(EMIT_FAST_ANSWER_TOOL_NAME);
		expect(seen.toolNames[0]).toContain(EMIT_AUTORAG_RESULTS_TOOL_NAME);
		expect(seen.toolNames[0]).toEqual(expect.arrayContaining(["bash", "read", "edit", "write"]));
		expect(minSync.queries).toEqual([]);
		expect(events.some((event) => event.type === "preliminary")).toBe(false);
		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("expected a complete event");
		expect(complete.response.answer).toBe("- Default model is now `new/model`.");
		expect(complete.response.results).toEqual([]);
		expect(complete.response.diagnostics?.some((diagnostic) => diagnostic.code === "missing-final-emit")).toBe(false);
		expect(
			complete.response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "query-routed" && diagnostic.message.includes("config"),
			),
		).toBe(true);
	});

	it("makes bash usable when the configured workspace directory does not exist yet", async () => {
		const workspace = join(root, "not-created-yet");
		const bashResults: boolean[] = [];
		const probeBash: FauxResponseStep = (context) => {
			const last = context.messages.at(-1);
			if (last?.role === "toolResult") {
				bashResults.push(last.isError === true);
				return emitConfigReport("- checked.") as AssistantMessage;
			}
			return fauxAssistantMessage([fauxToolCall("bash", { command: "echo ok" })], {
				stopReason: "toolUse",
			}) as AssistantMessage;
		};
		const model = fauxModel(probeBash, probeBash);
		const agent = agentWith({
			model,
			workspacePath: workspace,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath },
		});

		const response = await agent.searchDocuments("check the provider health");

		expect(bashResults).toEqual([false]);
		expect(existsSync(workspace)).toBe(true);
		expect(response.answer).toBe("- checked.");
	});

	it("lets the model edit the config file with its write tool and report the change", async () => {
		const next = `${JSON.stringify({ model: { provider: "openrouter", id: "new/model" } }, null, 2)}\n`;
		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall("write", { path: configPath, content: next })], {
				stopReason: "toolUse",
			}),
			emitConfigReport("- model.id: old/model → new/model"),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath },
		});

		const response = await agent.searchDocuments("switch the agent model to new/model");

		expect(JSON.parse(readFileSync(configPath, "utf8")).model.id).toBe("new/model");
		expect(response.answer).toBe("- model.id: old/model → new/model");
	});

	it("restores the previous config and says so when the edited config no longer validates", async () => {
		const before = readFileSync(configPath, "utf8");
		const broken = `${JSON.stringify({ model: { provider: "openrouter", id: "glm-5.3-flash" } }, null, 2)}\n`;
		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall("write", { path: configPath, content: broken })], {
				stopReason: "toolUse",
			}),
			emitConfigReport("- model.id: old/model → glm-5.3-flash"),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: {
				configPath,
				skillPath,
				validate: async () => "Model openrouter/glm-5.3-flash is not in the catalog",
			},
		});

		const response = await agent.searchDocuments("switch the agent model to glm 5.3 flash");

		expect(readFileSync(configPath, "utf8")).toBe(before);
		expect(response.answer).toContain("- model.id: old/model → glm-5.3-flash");
		expect(response.answer).toMatch(/restored/iu);
		expect(response.answer).toContain("not in the catalog");
		expect(response.diagnostics?.some((d) => d.code === "self-config-rolled-back" && d.severity === "warning")).toBe(
			true,
		);
	});

	it("keeps a valid edit and does not append a rollback notice", async () => {
		const next = `${JSON.stringify({ model: { provider: "openrouter", id: "new/model" } }, null, 2)}\n`;
		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall("write", { path: configPath, content: next })], {
				stopReason: "toolUse",
			}),
			emitConfigReport("- model.id: old/model → new/model"),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath, validate: async () => undefined },
		});

		const response = await agent.searchDocuments("switch the agent model to new/model");

		expect(readFileSync(configPath, "utf8")).toBe(next);
		expect(response.answer).toBe("- model.id: old/model → new/model");
		expect(response.diagnostics?.some((d) => d.code === "self-config-rolled-back")).toBe(false);
	});

	it("does not roll back when the config was left unchanged, even if it was already invalid", async () => {
		const before = readFileSync(configPath, "utf8");
		const model = fauxModel(emitConfigReport("- Nothing changed."));
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath, validate: async () => "was already broken" },
		});

		const response = await agent.searchDocuments("how is my model configured?");

		expect(readFileSync(configPath, "utf8")).toBe(before);
		expect(response.answer).toBe("- Nothing changed.");
		expect(response.diagnostics?.some((d) => d.code === "self-config-rolled-back")).toBe(false);
	});

	it("reminds the model to emit when it ends the configuration turn with prose", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(
			fauxAssistantMessage("I changed the model.", { stopReason: "stop" }),
			capture(emitConfigReport("- Changed the model."), seen),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath },
		});

		const response = await agent.searchDocuments("switch the agent model");

		expect(seen.prompts[0]).toContain(EMIT_AUTORAG_RESULTS_TOOL_NAME);
		expect(response.answer).toBe("- Changed the model.");
	});

	it("never judges or stores a configuration report as retrieval evidence", async () => {
		const routing = jevRouting("config");
		const evidenceQuestionIds: string[] = [];
		// Answers every evidence-support question with p=0.9, so anything judged would be stored.
		const backend: JevBackend = {
			name: "recording",
			async judge(request) {
				const evidence = request.questions.filter((question) =>
					question.id.startsWith(EVIDENCE_QUESTION_ID_PREFIX),
				);
				if (evidence.length === 0) return routing.judge(request);
				evidenceQuestionIds.push(...evidence.map((question) => question.id));
				return {
					...(await routing.judge({ ...request, questions: [] })),
					answers: evidence.map(() => ({ answer: 0.9 })),
				};
			},
		};
		const setting = '"id": "new/model"';
		const model = fauxModel(
			fauxAssistantMessage(
				[
					fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
						answer: "The default model is new/model [1].",
						results: [
							{
								number: 1,
								title: "Config",
								summary: "model.id",
								evidence: [{ excerpt: setting }],
								confidence: 0.9,
							},
						],
						mapping: [{ number: 1, source: configPath, method: "read", content: setting }],
					}),
				],
				{ stopReason: "toolUse" },
			),
		);
		const agent = agentWith({ model, jev: { backend }, selfConfig: { configPath, skillPath } });

		const response = await agent.searchDocuments("what is the default model now?");

		expect(evidenceQuestionIds).toEqual([]);
		expect(response.diagnostics?.some((diagnostic) => diagnostic.source === "memory")).toBe(false);
		const memory = new RetrievalMemory({ storagePath: join(root, "memory.json") });
		memory.load();
		expect(memory.getJudgedEvidence()).toEqual([]);
	});

	it("falls back to local search with a diagnostic when the setup skill cannot be loaded", async () => {
		const source = join(docs, "x.txt");
		const model = fauxModel(
			fauxAssistantMessage(
				[
					fauxToolCall(EMIT_FAST_ANSWER_TOOL_NAME, {
						answer: "Fast.",
						results: [],
					}),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage("done", { stopReason: "stop" }),
			fauxAssistantMessage(
				[
					fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
						answer: "Verified.",
						results: [
							{
								number: 1,
								title: "T",
								summary: "S",
								evidence: [{ excerpt: "E" }],
								confidence: 0.9,
							},
						],
						mapping: [{ number: 1, source, method: "bash", content: "E" }],
					}),
				],
				{ stopReason: "toolUse" },
			),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("config") },
			selfConfig: { configPath, skillPath: join(root, "missing", "SKILL.md") },
		});

		const response = await agent.searchDocuments("switch the agent model");

		expect(response.answer).toBe("Verified.");
		expect(
			response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "self-config-unavailable" && diagnostic.severity === "warning",
			),
		).toBe(true);
	});
});
