import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, type Message } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-llm-transcript-"));
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

/** A grounded final answer: plain text citing the local file the harness resolves. */
function groundedAnswer(answer: string) {
	const source = join(root, "grounded.txt");
	writeFileSync(source, answer);
	return fauxAssistantMessage(`${answer} [file:${source}]`, { stopReason: "stop" });
}

function toolNamesDeclaredToModel(messages: readonly Message[]): string[] {
	const names: string[] = [];
	for (const message of messages) {
		if (message.role !== "system") continue;
		for (const tool of message.toolsAdded ?? []) names.push(tool.name);
	}
	return names;
}

const isNoFinalAnswer = (diagnostic: { code: string }): boolean => diagnostic.code === "no-final-answer";

describe("provider transcript", () => {
	it("declares the system prompt and the retrieval tools, no emit tool", async () => {
		const registration = registerFauxProvider({
			api: `faux-${randomUUID()}`,
			models: [{ id: "transcript-model" }],
		});
		const providerTranscripts: { roles: string[]; declaredTools: string[] }[] = [];
		registration.setResponses([
			fauxAssistantMessage("Initial pass.", { stopReason: "stop" }),
			(context) => {
				providerTranscripts.push({
					roles: context.messages.map((message) => message.role),
					declaredTools: toolNamesDeclaredToModel(context.messages),
				});
				return groundedAnswer("grounded answer");
			},
		]);
		registrations.push(registration);

		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
			minSync: { autoInstall: false },
		});

		const response = await agent.searchDocuments("find the grounded answer");

		expect(providerTranscripts.length).toBeGreaterThan(0);
		for (const transcript of providerTranscripts) {
			expect(transcript.roles).toContain("system");
			expect(transcript.declaredTools).toContain("search_all_documents");
			expect(transcript.declaredTools).not.toContain("emit_autorag_results");
		}
		expect(response.answer).toBe("grounded answer [1]");
		expect(response.results).toHaveLength(1);
		expect(response.results[0]?.source).toBe(join(root, "grounded.txt"));
		expect(response.diagnostics?.some(isNoFinalAnswer) ?? false).toBe(false);
	});
});
