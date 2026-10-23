import { randomUUID } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
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
import { EVIDENCE_QUESTION_ID_PREFIX } from "../../src/agent/evidence-judgment.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
} from "../../src/agent/query-routing.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import { RetrievalMemory } from "../../src/memory/memory.ts";
import type { RetrievalOptions, RetrievalResult } from "../../src/retrieval/types.ts";

let root: string;
let docs: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-organize-route-"));
	docs = join(root, "docs");
	mkdirSync(docs, { recursive: true });
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

function report(answer: string): FauxResponseStep {
	return fauxAssistantMessage(answer, { stopReason: "stop" });
}

function jevRouting(route: QueryRoute): MockBackend {
	const distribution = { local: 0.05, direct: 0.05, organize: 0.05, extra: 0.05, [route]: 0.85 };
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
		minSync: { autoInstall: false },
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

describe.each(["organize", "extra"] as const)(
	"Jev %s branch: acts on the request without a search pipeline",
	(route) => {
		it("runs one tool-capable turn: no retrieval, no fast answer, no verification, no decomposition", async () => {
			const seen: Seen = { prompts: [], toolNames: [] };
			const model = fauxModel(capture(report("- Done."), seen));
			const agent = agentWith({ model, jev: { backend: jevRouting(route) } });
			const minSync = recordingMinSync();
			Object.assign(agent, { minSyncMethod: minSync.method });

			const events = await collect(agent, "move the PDFs in Downloads into a pdf folder");

			expect(seen.prompts).toHaveLength(1);
			expect(seen.prompts[0]).toContain("move the PDFs in Downloads into a pdf folder");
			expect(seen.prompts[0]).not.toContain("Baseline retrieval evidence");
			// The model can act: bash and the file tools stay available.
			expect(seen.toolNames[0]).toEqual(expect.arrayContaining(["bash", "read", "edit", "write"]));
			expect(minSync.queries).toEqual([]);
			expect(events.some((event) => event.type === "preliminary")).toBe(false);
			const complete = events.find((event) => event.type === "complete");
			if (complete?.type !== "complete") throw new Error("expected a complete event");
			expect(complete.response.answer).toBe("- Done.");
			expect(complete.response.results).toEqual([]);
			expect(complete.response.diagnostics?.some((diagnostic) => diagnostic.code === "no-final-answer")).toBe(false);
			expect(
				complete.response.diagnostics?.some(
					(diagnostic) => diagnostic.code === "query-routed" && diagnostic.message.includes(route),
				),
			).toBe(true);
		});

		it("never judges or stores the turn's report as retrieval evidence", async () => {
			const routing = jevRouting(route);
			const evidenceQuestionIds: string[] = [];
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
			const kept = join(docs, "a.txt");
			writeFileSync(kept, "alpha");
			const model = fauxModel(report(`Moved [file:${kept}].`));
			const agent = agentWith({ model, jev: { backend } });

			const response = await agent.searchDocuments("rename a.txt to b.txt");

			expect(evidenceQuestionIds).toEqual([]);
			expect(response.diagnostics?.some((diagnostic) => diagnostic.source === "memory")).toBe(false);
			const memory = new RetrievalMemory({ storagePath: join(root, "memory.json") });
			memory.load();
			expect(memory.getJudgedEvidence()).toEqual([]);
			expect(memory.getSchema().curatedResults).toEqual([]);
		});

		it("lets the model really change files with bash on this turn", async () => {
			const source = join(docs, "draft.txt");
			const target = join(docs, "final.txt");
			writeFileSync(source, "hello");
			const step: FauxResponseStep = (context) => {
				const last = context.messages.at(-1);
				if (last?.role === "toolResult") return report(`- ${source} → ${target}`) as AssistantMessage;
				return fauxAssistantMessage([fauxToolCall("bash", { command: `mv '${source}' '${target}'` })], {
					stopReason: "toolUse",
				}) as AssistantMessage;
			};
			const agent = agentWith({ model: fauxModel(step, step), jev: { backend: jevRouting(route) } });

			const response = await agent.searchDocuments("rename draft.txt to final.txt");

			expect(existsSync(source)).toBe(false);
			expect(existsSync(target)).toBe(true);
			expect(response.answer).toContain("final.txt");
		});
	},
);

describe("Jev organize branch prompt", () => {
	it("carries the file-organization rules and keeps conversion tools visible", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(capture(report("- Done."), seen));
		const agent = agentWith({ model, jev: { backend: jevRouting("organize") } });

		await agent.searchDocuments("convert every docx in Downloads to pdf");

		expect(seen.prompts[0]).toMatch(/organiz/iu);
		expect(seen.prompts[0]).toMatch(/old path → new path/u);
		expect(seen.prompts[0]).toMatch(/trash/iu);
		expect(seen.toolNames[0]).toEqual(expect.arrayContaining(["convert_files", "list_file_conversions"]));
		expect(seen.toolNames[0]).not.toContain("search_all_documents");
	});
});

describe("Jev extra branch prompt", () => {
	it("is the direct path with tools allowed: no search tools are required, none are forbidden", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(capture(report("Committed."), seen));
		const agent = agentWith({ model, jev: { backend: jevRouting("extra") } });

		await agent.searchDocuments("commit my changes and open a PR");

		expect(seen.prompts[0]).not.toMatch(/Do NOT call any/u);
		expect(seen.prompts[0]).toContain("commit my changes and open a PR");
	});
});
