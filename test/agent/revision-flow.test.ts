import { randomUUID } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import {
	type FauxProviderRegistration,
	type FauxResponseStep,
	fauxAssistantMessage,
	fauxToolCall,
} from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent, type AutoRAGAgentOptions } from "../../src/agent/agent.ts";
import type { SearchDocumentsStreamEvent } from "../../src/agent/search-documents.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";
import { baselineJev } from "../helpers/baseline-jev.ts";

let root: string;
let docs: string;
let policy: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-revision-"));
	docs = join(root, "docs");
	policy = join(docs, "refund-policy.txt");
	registrations = [];
	mkdirSync(docs, { recursive: true });
	writeFileSync(policy, "Refund exceptions require director approval before payout.\nThe cap is 500000 won.");
});

afterEach(() => {
	for (const reg of registrations) reg.unregister();
	rmSync(root, { recursive: true, force: true });
});

function fauxModel(...responses: FauxResponseStep[]) {
	const reg = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "faux-model", reasoning: true }] });
	reg.setResponses(responses);
	registrations.push(reg);
	return reg.getModel();
}

function options(model: ReturnType<typeof fauxModel>): AutoRAGAgentOptions {
	return {
		model,
		searchPaths: [docs],
		memoryPath: join(root, "memory.json"),
		workspacePath: root,
		minSync: { autoInstall: false },
		jikji: false,
		jev: baselineJev(),
	};
}

/** The fast phase answers from the baseline evidence it was handed, citing the id the baseline printed for the policy. */
const fastCitingBaseline: FauxResponseStep = (context) => {
	const text = context.messages
		.flatMap((message) => (message.role === "user" && Array.isArray(message.content) ? message.content : []))
		.map((part) => (part.type === "text" ? part.text : ""))
		.join("\n");
	const id = text.match(/\[(e\d+)\] \/docs\/policy\.md/)?.[1];
	if (id === undefined) throw new Error("baseline context printed no evidence id");
	return fauxAssistantMessage(
		`- Refund exceptions need team-lead approval [${id}]\n- The cap is 50만 원 [${id}]\n- Paid in the next cycle`,
		{ stopReason: "stop" },
	);
};

function withBaseline(agent: AutoRAGAgent): AutoRAGAgent {
	// The MinSync retrieval method is private; the suite replaces it the same way to feed the baseline.
	const internals = agent as unknown as { minSyncMethod: unknown };
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (): Promise<RetrievalResult[]> => [
			{
				id: "1",
				source: "/docs/policy.md",
				content: "Refund exceptions require director approval.",
				score: 1,
				metadata: {},
			},
		],
	};
	return agent;
}

function revise(args: Record<string, string>): FauxResponseStep {
	return fauxAssistantMessage([fauxToolCall("revise_answer", args)], { stopReason: "toolUse" });
}

async function collect(agent: AutoRAGAgent, query: string): Promise<SearchDocumentsStreamEvent[]> {
	const events: SearchDocumentsStreamEvent[] = [];
	for await (const event of agent.searchDocumentsStream(query)) events.push(event);
	return events;
}

function kinds(events: readonly SearchDocumentsStreamEvent[]): string[] {
	return events.map((event) => event.type);
}

describe("revise_answer in the two-phase search", () => {
	it("offers the first answer's claims to the verification phase, with the model's own citation ids", async () => {
		const prompts: string[] = [];
		const capture: FauxResponseStep = (context) => {
			const lastUser = [...context.messages].reverse().find((message) => message.role === "user");
			const content = lastUser?.content;
			prompts.push(
				typeof content === "string"
					? content
					: (content ?? []).map((part) => (part.type === "text" ? part.text : "")).join(""),
			);
			return fauxAssistantMessage("Nothing needs correcting.", { stopReason: "stop" });
		};
		const agent = withBaseline(new AutoRAGAgent(options(fauxModel(fastCitingBaseline, capture))));

		await collect(agent, "who approves refund exceptions?");

		const prompt = prompts.at(-1) ?? "";
		expect(prompt).toMatch(/c1: - Refund exceptions need team-lead approval \[e\d+\]/);
		expect(prompt).toContain("c3: - Paid in the next cycle");
		expect(prompt).toContain("revise_answer");
		// The delta-only rewriting contract is gone: corrections are tool calls now.
		expect(prompt).not.toContain("DELTA ONLY");
	});

	it("streams each revision as an event between the preliminary and the complete answer", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({ op: "retract", claimId: "c3", reason: "No source states a payout date" }),
			fauxAssistantMessage("Removed the unsupported payout claim.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const order = kinds(events).filter((type) => type !== "progress");
		expect(order).toEqual(["preliminary", "revision", "complete"]);
		const revision = events.find((event) => event.type === "revision");
		if (revision?.type !== "revision") throw new Error("no revision event");
		expect(revision.revision).toMatchObject({
			op: "retract",
			claimId: "c3",
			original: "- Paid in the next cycle",
			reason: "No source states a payout date",
		});
	});

	it("makes the complete answer the revised answer, not a delta", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({ op: "retract", claimId: "c3", reason: "unsupported" }),
			fauxAssistantMessage("Removed the payout claim.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).toContain("Refund exceptions need team-lead approval [1]");
		expect(complete.response.answer).toContain("The cap is 50만 원 [1]");
		expect(complete.response.answer).not.toContain("Paid in the next cycle");
		// The model's closing note is not appended to the answer: the revisions carry the changes.
		expect(complete.response.answer).not.toContain("Removed the payout claim");
		expect(complete.response.results[0]?.number).toBe(1);
	});

	it("applies a replacement with a new citation and numbers it like any other answer", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({
				op: "replace",
				claimId: "c1",
				text: `- Refund exceptions need director approval [file:${policy}]`,
				reason: "The policy file says director",
			}),
			fauxAssistantMessage("Corrected the approver.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const revision = events.find((event) => event.type === "revision");
		if (revision?.type !== "revision") throw new Error("no revision event");
		expect(revision.revision.original).toMatch(/team-lead approval \[1\]/);
		expect(revision.revision.replacement).toMatch(/director approval \[2\]/);
		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).toContain("director approval [2]");
		expect(complete.response.answer).not.toContain("team-lead");
		expect(complete.response.results.map((result) => result.source)).toContain(policy);
	});

	it("tells the model a rejected revision failed, and no revision is announced", async () => {
		const results: string[] = [];
		const watch: FauxResponseStep = (context) => {
			const last = context.messages.at(-1);
			if (last?.role === "toolResult") {
				results.push(last.content.map((part) => (part.type === "text" ? part.text : "")).join(""));
			}
			return fauxAssistantMessage("Done.", { stopReason: "stop" });
		};
		const model = fauxModel(fastCitingBaseline, revise({ op: "retract", claimId: "c99", reason: "x" }), watch);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		expect(results.join("\n")).toContain("Unknown claim c99");
		expect(kinds(events)).not.toContain("revision");
		// Nothing was applied, so the model's own reply is the answer.
		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).toBe("Done.");
	});

	it("answers with the model's own reply, as before, when it corrected nothing through the tool", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			fauxAssistantMessage("Verified; nothing to correct.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(kinds(events)).not.toContain("revision");
		expect(complete.response.answer).toBe("Verified; nothing to correct.");
	});

	it("answers with the model's reply when it only confirmed claims: a confirmation changes nothing", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({ op: "confirm", claimId: "c1", reason: "matches the policy" }),
			fauxAssistantMessage("The first answer holds.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(kinds(events)).toContain("revision");
		expect(complete.response.answer).toBe("The first answer holds.");
	});

	it("drops the closing note once the tool corrected something: the revisions carry the change", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({ op: "retract", claimId: "c3", reason: "unsupported" }),
			fauxAssistantMessage("Removed the payout claim.", { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).not.toContain("Removed the payout claim");
	});

	it("keeps a closing reply that cites evidence after a correction: the model also wrote something new", async () => {
		const model = fauxModel(
			fastCitingBaseline,
			revise({ op: "retract", claimId: "c3", reason: "unsupported" }),
			fauxAssistantMessage(`Exceptions also need finance sign-off [file:${policy}].`, { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).toContain("team-lead approval");
		expect(complete.response.answer).not.toContain("Paid in the next cycle");
		expect(complete.response.answer).toContain("also need finance sign-off");
		expect(complete.response.answer).not.toContain("[file:");
	});

	it("answers with the closing reply alone when every claim was retracted", async () => {
		const model = fauxModel(
			fauxAssistantMessage("Only claim, wrong.", { stopReason: "stop" }),
			revise({ op: "retract", claimId: "c1", reason: "unsupported" }),
			fauxAssistantMessage(`Nothing is documented about this [file:${policy}].`, { stopReason: "stop" }),
		);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		const events = await collect(agent, "who approves refund exceptions?");

		const complete = events.find((event) => event.type === "complete");
		if (complete?.type !== "complete") throw new Error("no complete event");
		expect(complete.response.answer).not.toContain("Only claim");
		expect(complete.response.answer).toContain("Nothing is documented");
	});

	it("does not offer revise_answer when no first answer reached the caller", async () => {
		const results: string[] = [];
		const watch: FauxResponseStep = (context) => {
			const last = context.messages.at(-1);
			if (last?.role === "toolResult") {
				results.push(last.content.map((part) => (part.type === "text" ? part.text : "")).join(""));
			}
			return fauxAssistantMessage("The complete answer, written out.", { stopReason: "stop" });
		};
		const model = fauxModel(fastCitingBaseline, revise({ op: "retract", claimId: "c1", reason: "x" }), watch);
		const agent = withBaseline(new AutoRAGAgent(options(model)));

		// searchDocuments installs no preliminary consumer, so the first answer never reaches a caller.
		const response = await agent.searchDocuments("who approves refund exceptions?");

		expect(results.join("\n")).toMatch(/no first answer/i);
		expect(response.answer).toContain("The complete answer, written out.");
	});
});
