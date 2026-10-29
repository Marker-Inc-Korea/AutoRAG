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
import { packagedSkillLocation } from "../../src/agent/packaged-skill.ts";
import {
	DECOMPOSE_QUESTION_ID,
	FOLLOW_UP_QUESTION_ID,
	QUERY_ROUTE_QUESTION_ID,
	type QueryRoute,
} from "../../src/agent/query-routing.ts";
import { SEARCH_ALL_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-all-tool.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import { SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME } from "../../src/agent/search-minsync-tool.ts";
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
	return lastUser === undefined ? "" : textOf(lastUser.content);
}

/** One content part as text; empty for anything that is not a text part. */
function partText(part: unknown): string {
	if (typeof part !== "object" || part === null) return "";
	const type = "type" in part ? part.type : undefined;
	const text = "text" in part ? part.text : undefined;
	return type === "text" && typeof text === "string" ? text : "";
}

/** Message content as one string, whether it is plain text or text parts. */
function textOf(content: unknown): string {
	if (typeof content === "string") return content;
	if (!Array.isArray(content)) return "";
	return content.map(partText).join("");
}

/**
 * A faux step answering `answer(ids)`, where `ids` are the evidence ids the
 * tool results in the transcript have printed so far.
 */
function citingStep(answer: (ids: readonly string[]) => string): FauxResponseStep {
	return (context) => {
		const ids = context.messages.flatMap((message) =>
			message.role === "toolResult"
				? [...textOf(message.content).matchAll(/\[(e\d+)\]/gu)].map((match) => match[1] as string)
				: [],
		);
		return fauxAssistantMessage(answer(ids), { stopReason: "stop" });
	};
}

/** The routing backend plus every evidence-judgment question Jev answered. */
function recordingEvidenceJudgment(route: QueryRoute): { backend: JevBackend; evidenceQuestionIds: string[] } {
	const routing = jevRouting(route);
	const evidenceQuestionIds: string[] = [];
	return {
		evidenceQuestionIds,
		backend: {
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
		},
	};
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

function jevRouting(route: QueryRoute, decomposeProbability = 0.9): MockBackend {
	const distribution = { local: 0.05, organize: 0.05, extra: 0.05, [route]: 0.85 };
	return new MockBackend({
		[QUERY_ROUTE_QUESTION_ID]: { answer: route, distribution, confidence: 0.9 },
		[DECOMPOSE_QUESTION_ID]: { answer: decomposeProbability },
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

function recordingMinSync(results: RetrievalResult[] = []) {
	const queries: string[] = [];
	return {
		queries,
		method: {
			isReady: () => true,
			isBinaryMissing: () => false,
			describe: () => ({ name: "minsync" }),
			retrieve: async (query: string, _options: RetrievalOptions): Promise<RetrievalResult[]> => {
				queries.push(query);
				return results;
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
	it("carries the organize skill invocation, the tool scope, and the file-organization rules", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(capture(report("- Done."), seen));
		const agent = agentWith({ model, jev: { backend: jevRouting("organize") } });

		await agent.searchDocuments("convert every docx in Downloads to pdf");

		const prompt = seen.prompts[0] ?? "";
		// The rules arrive as a loaded skill with a path-opaque location: the turn
		// prompt never names the file the skill came from.
		expect(prompt).toContain(`<skill name="organize" location="${packagedSkillLocation("organize")}">`);
		expect(prompt).toContain("</skill>");
		expect(prompt).toMatch(/old path → new path/u);
		expect(prompt).toMatch(/trash/iu);
		expect(prompt).not.toContain("SKILL.md");
		expect(prompt).not.toContain("internal-skills");
		// The prompt, not the tool list, scopes this turn: it names the tools to
		// use and the retrieval/web tools not to call.
		expect(prompt).toMatch(/work with bash/u);
		expect(prompt).toContain("convert_files");
		expect(prompt).toMatch(/do NOT call the retrieval, datasource, or web tools/u);
		expect(prompt).toContain(SEARCH_ALL_DOCUMENTS_TOOL_NAME);
		expect(prompt).toContain(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME);
		// Nothing is withheld any more: the converter and the search tools are both
		// visible to the model on an organize turn.
		expect(seen.toolNames[0]).toEqual(
			expect.arrayContaining([
				"convert_files",
				"list_file_conversions",
				SEARCH_ALL_DOCUMENTS_TOOL_NAME,
				SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
			]),
		);
	});

	it("falls back to local search with a diagnostic when the organize skill cannot be loaded", async () => {
		const decompositionModel = fauxModel(
			fauxAssistantMessage('["move the PDFs into a pdf folder"]', { stopReason: "stop" }),
		);
		const model = fauxModel(
			fauxAssistantMessage("Fast.", { stopReason: "stop" }),
			fauxAssistantMessage("Verified.", { stopReason: "stop" }),
		);
		const agent = agentWith({
			model,
			jev: { backend: jevRouting("organize") },
			queryDecomposition: { model: decompositionModel },
			organizeSkillPath: join(root, "missing", "SKILL.md"),
		});

		const response = await agent.searchDocuments("move the PDFs in Downloads into a pdf folder");

		expect(response.answer).toBe("Verified.");
		expect(
			response.diagnostics?.some(
				(diagnostic) => diagnostic.code === "organize-skill-unavailable" && diagnostic.severity === "warning",
			),
		).toBe(true);
	});
});

describe("no Jev route changes the active tool set", () => {
	/** A TUI-hosted agent on the faux model: the bound session outlives the run, so a swap would show here. */
	async function hostedAgent(route: QueryRoute, ...responses: FauxResponseStep[]) {
		const registration = registerFauxProvider({ api: `faux-hosted-${randomUUID()}`, models: [{ id: "faux-model" }] });
		registration.setResponses(responses);
		registrations.push(registration);
		const agent = agentWith({
			model: registration.getModel(),
			apiKey: "test-key",
			piSessionDir: join(root, "sessions"),
			jev: { backend: jevRouting(route) },
		});
		const hosted = await agent.createPiInteractiveRuntime();
		return { agent, session: hosted.runtime.session, dispose: () => hosted.dispose() };
	}

	it("leaves an interactive session's tools exactly as they were across an organize turn", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const hosted = await hostedAgent("organize", capture(report("- Done."), seen));
		try {
			const before = hosted.session.getActiveToolNames();
			expect(before).toEqual(
				expect.arrayContaining([
					"bash",
					"convert_files",
					SEARCH_ALL_DOCUMENTS_TOOL_NAME,
					SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME,
				]),
			);

			const response = await hosted.agent.searchDocuments("move the PDFs in Downloads into a pdf folder");

			expect(
				response.diagnostics?.some(
					(diagnostic) => diagnostic.code === "query-routed" && diagnostic.message.includes("organize"),
				),
			).toBe(true);
			// After the run, and during it: the same set the session started with.
			expect(hosted.session.getActiveToolNames()).toEqual(before);
			expect([...(seen.toolNames[0] ?? [])].sort()).toEqual([...before].sort());
		} finally {
			await hosted.dispose();
		}
	});
});

describe("Jev organize branch memory", () => {
	it("never judges or stores the turn's report as retrieval evidence", async () => {
		const { backend, evidenceQuestionIds } = recordingEvidenceJudgment("organize");
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
});

describe("Jev extra branch prompt, tools, and memory", () => {
	it("is the tool path with no search required, none forbidden, and the answer rules intact", async () => {
		const seen: Seen = { prompts: [], toolNames: [] };
		const model = fauxModel(capture(report("Committed."), seen));
		const agent = agentWith({ model, jev: { backend: jevRouting("extra") } });

		await agent.searchDocuments("commit my changes and open a PR");

		expect(seen.prompts[0]).not.toMatch(/Do NOT call any/u);
		expect(seen.prompts[0]).toContain("commit my changes and open a PR");
		// A knowledge answer, a lookup, or a task: all three are allowed, and the
		// tools that can settle a lookup are named.
		expect(seen.prompts[0]).toMatch(/Answer from your own knowledge/u);
		expect(seen.prompts[0]).toContain(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME);
		expect(seen.prompts[0]).toContain(SEARCH_ALL_DOCUMENTS_TOOL_NAME);
		// The same answer rules the other answer prompts use.
		expect(seen.prompts[0]).toMatch(/Answer language/u);
		expect(seen.prompts[0]).toMatch(/Citations/u);
	});

	it("keeps every tool the session has: the same set a local run's fast phase sees", async () => {
		const extraSeen: Seen = { prompts: [], toolNames: [] };
		const extra = agentWith({
			model: fauxModel(capture(report("Paris."), extraSeen)),
			jev: { backend: jevRouting("extra") },
		});
		await extra.searchDocuments("what is the capital of France?");

		const localSeen: Seen = { prompts: [], toolNames: [] };
		const local = agentWith({
			model: fauxModel(capture(report("The policy is in refund.md."), localSeen)),
			// No decomposition, so the fast phase is this run's first model call.
			jev: { backend: jevRouting("local", 0.1) },
		});
		await local.searchDocuments("where is the refund policy?");

		expect(extraSeen.toolNames[0]).toEqual(expect.arrayContaining(["bash", "read", "edit", "write"]));
		expect(extraSeen.toolNames[0]?.slice().sort()).toEqual(localSeen.toolNames[0]?.slice().sort());
	});

	it("returns real tool results when the extra turn searches, and its cited evidence becomes results and memory", async () => {
		const { backend, evidenceQuestionIds } = recordingEvidenceJudgment("extra");
		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME, { query: "refund policy" })], {
				stopReason: "toolUse",
			}),
			citingStep((ids) => `Refunds need director approval [${ids[0]}].`),
		);
		const agent = agentWith({ model, jev: { backend } });
		const minSync = recordingMinSync([
			{
				id: "c1",
				source: "/docs/refund.md",
				content: "Refund exceptions require director approval before payout.",
				score: 1,
				metadata: { method: "bm25" },
			},
		]);
		Object.assign(agent, { minSyncMethod: minSync.method });

		const response = await agent.searchDocuments("who approves refund exceptions?");

		// The tool really ran: the query reached retrieval and the model could cite its evidence.
		expect(minSync.queries).toEqual(["refund policy"]);
		expect(response.answer).toBe("Refunds need director approval [1].");
		expect(response.results.map((result) => [result.number, result.title])).toEqual([[1, "refund.md"]]);
		// Cited retrieved evidence feeds retrieval memory exactly like a local run.
		expect(evidenceQuestionIds.length).toBeGreaterThan(0);
		const memory = new RetrievalMemory({ storagePath: join(root, "memory.json") });
		memory.load();
		expect(memory.getJudgedEvidence()).toHaveLength(1);
		expect(memory.getSchema().curatedResults).toHaveLength(1);
	});

	it("stays isolated when the extra answer cites no retrieved evidence", async () => {
		const { backend, evidenceQuestionIds } = recordingEvidenceJudgment("extra");
		const model = fauxModel(
			fauxAssistantMessage([fauxToolCall(SEARCH_MINSYNC_DOCUMENTS_TOOL_NAME, { query: "refund policy" })], {
				stopReason: "toolUse",
			}),
			report("- Committed the change."),
		);
		const agent = agentWith({ model, jev: { backend } });
		const minSync = recordingMinSync([
			{
				id: "c1",
				source: "/docs/refund.md",
				content: "Refund exceptions require director approval before payout.",
				score: 1,
				metadata: { method: "bm25" },
			},
		]);
		Object.assign(agent, { minSyncMethod: minSync.method });

		const response = await agent.searchDocuments("commit my changes");

		expect(response.answer).toBe("- Committed the change.");
		expect(response.results).toEqual([]);
		expect(evidenceQuestionIds).toEqual([]);
		expect(response.diagnostics?.some((diagnostic) => diagnostic.source === "memory")).toBe(false);
		const memory = new RetrievalMemory({ storagePath: join(root, "memory.json") });
		memory.load();
		expect(memory.getJudgedEvidence()).toEqual([]);
		expect(memory.getSchema().curatedResults).toEqual([]);
	});
});
