import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-model-request-error-"));
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function agentAnswering(respond: () => ReturnType<typeof fauxAssistantMessage>): AutoRAGAgent {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "error-model" }] });
	registration.setResponses(Array.from({ length: 8 }, () => respond));
	registrations.push(registration);
	return new AutoRAGAgent({
		model: registration.getModel(),
		searchPaths: ["test/fixtures/sample-project"],
		workspacePath: root,
		memoryPath: join(root, "memory.json"),
		jikji: false,
		minSync: false,
	});
}

const RATE_LIMIT =
	"429 Rate limit reached for gpt-6-luna on tokens per min (TPM): Limit 200000, Used 199000, Requested 22000.";

describe("degraded answer when the model request fails", () => {
	it("carries the provider error text instead of saying the agent did not record why", async () => {
		const agent = agentAnswering(() => fauxAssistantMessage([], { stopReason: "error", errorMessage: RATE_LIMIT }));

		const response = await agent.searchDocuments("refund approval");

		expect(response.results).toEqual([]);
		expect(response.answer).toContain(RATE_LIMIT);
		expect(response.answer).not.toContain("did not record why it stopped");
		expect(response.answer).not.toContain("broaden the configured searchPaths");
		expect(response.diagnostics).toContainEqual(
			expect.objectContaining({
				code: "model-request-failed",
				severity: "error",
				message: `The model request failed: ${RATE_LIMIT}`,
			}),
		);
	});

	it("keeps the old wording when the model stopped normally without emitting", async () => {
		const agent = agentAnswering(() => fauxAssistantMessage("Still looking.", { stopReason: "stop" }));

		const response = await agent.searchDocuments("refund approval");

		expect(response.answer).toContain("broaden the configured searchPaths");
		expect(response.diagnostics?.some((diagnostic) => diagnostic.code === "model-request-failed")).toBe(false);
	});
});
