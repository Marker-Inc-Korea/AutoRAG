import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { Context } from "@earendil-works/pi-ai";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import type { JikjiAnswerPack } from "../../src/jikji/index.ts";
import type { RetrievalResult } from "../../src/retrieval/types.ts";

/** Bench-only view of the agent's private Jikji provider, MinSync method and follow-up check. */
type AgentInternals = {
	minSyncMethod: unknown;
	jikjiClient: unknown;
	findJikji: (query: string, opts?: { readonly signal?: AbortSignal }) => Promise<unknown>;
	shouldFollowUp: (query: string, fastAnswer: unknown) => Promise<boolean>;
	interactive: { steerInterrupt: AbortController };
};

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-jikji-bg-"));
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function userText(context: Context): string {
	return context.messages
		.flatMap((message) => {
			if (message.role !== "user") return [];
			const content = message.content;
			if (typeof content === "string") return [content];
			return content.flatMap((part) =>
				part && typeof part === "object" && "text" in part ? [String(part.text)] : [],
			);
		})
		.join("\n");
}

function pack(paths: readonly string[]): JikjiAnswerPack {
	return {
		answerPaths: paths,
		paths,
		candidates: paths.map((path) => ({ path, nextRead: "original" })),
		evidencePack: [],
		handoffAction: "direct_use",
		toolCallPolicy: { stopAfterFind: false, forbiddenTools: [], allowedFollowups: [] },
		agentShouldNotRerank: false,
	};
}

function agentFor(events: string[], prompts: string[]): AutoRAGAgent {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "bg-model" }] });
	registrations.push(registration);
	registration.setResponses([
		(context) => {
			events.push("fast-model");
			prompts.push(userText(context));
			return fauxAssistantMessage("first answer", { stopReason: "stop" });
		},
		(context) => {
			events.push("verify-model");
			prompts.push(userText(context));
			return fauxAssistantMessage("final answer", { stopReason: "stop" });
		},
	]);
	return new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: [root],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		minSync: { autoInstall: false },
		jikji: false,
		everything: false,
		fsearch: false,
	});
}

function wire(agent: AutoRAGAgent, find: AgentInternals["findJikji"], events: string[]): AgentInternals {
	const internals = agent as unknown as AgentInternals;
	internals.jikjiClient = {};
	internals.findJikji = find;
	internals.minSyncMethod = {
		isReady: () => true,
		isBinaryMissing: () => false,
		retrieve: async (): Promise<RetrievalResult[]> => [
			{ id: "c", source: "/docs/chunk.md", content: "a chunk", score: 0.9, metadata: {} },
		],
	};
	const followUp = internals.shouldFollowUp.bind(agent);
	internals.shouldFollowUp = (query, fastAnswer) => {
		events.push("follow-up-check");
		return followUp(query, fastAnswer);
	};
	return internals;
}

describe("Jikji find runs in the background of the fast answer (#2060)", () => {
	it("does not hold the fast answer; it settles before the follow-up check and the verification prompt", async () => {
		const events: string[] = [];
		const prompts: string[] = [];
		const agent = agentFor(events, prompts);
		let release = (): void => undefined;
		const gate = new Promise<void>((resolve) => {
			release = resolve;
		});
		wire(
			agent,
			async () => {
				events.push("jikji-start");
				await gate;
				events.push("jikji-done");
				return { answerPack: pack(["/docs/0417.md"]) };
			},
			events,
		);

		for await (const event of agent.searchDocumentsStream("refund approval", {})) {
			// The first answer reaches the caller while Jikji is still running.
			if (event.type === "preliminary") {
				events.push("preliminary");
				release();
			}
		}

		expect(events).toEqual([
			"jikji-start",
			"fast-model",
			"preliminary",
			"jikji-done",
			"follow-up-check",
			"verify-model",
		]);
		expect(prompts[0]).not.toContain("Jikji's file discovery");
		expect(prompts[1]).toContain("start exploring from them.\n- /docs/0417.md");
	});

	it("a failing Jikji find neither blocks nor breaks the run, and adds no starting points", async () => {
		const events: string[] = [];
		const prompts: string[] = [];
		const agent = agentFor(events, prompts);
		wire(
			agent,
			async () => {
				throw new Error("jikji exploded");
			},
			events,
		);

		const response = await agent.searchDocuments("refund approval");

		expect(response.answer).toContain("final answer");
		expect(prompts[1]).not.toContain("Jikji's file discovery");
	});

	it("gives Jikji the run's stop signal, not the steer signal, so a steer does not cancel it", async () => {
		const events: string[] = [];
		const prompts: string[] = [];
		const agent = agentFor(events, prompts);
		const seen: { query: string; signal: AbortSignal | undefined; steerSignal: AbortSignal }[] = [];
		const internals = wire(
			agent,
			async (query, opts) => {
				seen.push({ query, signal: opts?.signal, steerSignal: internals.interactive.steerInterrupt.signal });
				return { answerPack: pack([]) };
			},
			events,
		);

		await agent.searchDocuments("refund approval");

		expect(seen.map((entry) => entry.query)).toEqual(["refund approval"]);
		expect(seen[0]?.signal).toBeInstanceOf(AbortSignal);
		expect(seen[0]?.signal).not.toBe(seen[0]?.steerSignal);
	});
});
