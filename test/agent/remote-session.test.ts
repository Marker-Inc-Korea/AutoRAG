import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

const FIXTURE_DIR = "test/fixtures/sample-project";
let tmpDir: string | undefined;
const registrations: FauxProviderRegistration[] = [];

interface AgentInternals {
	innerAgent: {
		state: {
			tools: Array<{ name: string }>;
		};
	};
}

afterEach(() => {
	for (const registration of registrations.splice(0)) registration.unregister();
	if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
	tmpDir = undefined;
});

function textOnlyModel() {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "text-only" }] });
	registration.setResponses([
		() =>
			fauxAssistantMessage([{ type: "text", text: "I could not find anything relevant." }], { stopReason: "stop" }),
	]);
	registrations.push(registration);
	return registration.getModel();
}

describe("AutoRAGAgent remote-session tool surface", () => {
	it("keeps the normal local tools during a remote peer search", () => {
		tmpDir = mkdtempSync(join(tmpdir(), "autorag-remote-session-"));
		const agent = new AutoRAGAgent({
			searchPaths: [FIXTURE_DIR],
			memoryPath: join(tmpDir, "memory.json"),
			remoteSession: true,
		});
		const names = (agent as unknown as AgentInternals).innerAgent.state.tools.map((tool) => tool.name);

		for (const name of ["bash", "jikji_find", "check_memory"]) {
			expect(names, name).toContain(name);
		}
		for (const name of [
			"semantic_search_local_docs",
			"search_all_documents",
			"search_datasource_documents",
			"emit_autorag_results",
		]) {
			expect(names, name).toContain(name);
		}

		const prompt = agent.getSystemPrompt();
		for (const name of ["bash", "jikji_find", "check_memory"]) {
			expect(prompt).toContain(`- **${name}**:`);
		}
	});

	it("returns a structured no-verified-results response when a remote session emits nothing", async () => {
		tmpDir = mkdtempSync(join(tmpdir(), "autorag-remote-empty-"));
		const agent = new AutoRAGAgent({
			model: textOnlyModel(),
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			remoteSession: true,
			minSync: false,
			jikji: false,
			thinking: false,
		});

		const response = await agent.searchDocuments("is there anything about unicorns?");

		expect(response.results).toEqual([]);
		expect(response.answer).toMatch(/no verified results/i);
		expect(response.diagnostics).toContainEqual(expect.objectContaining({ code: "no-verified-results" }));
	});

	it("still throws when a local session emits nothing", async () => {
		tmpDir = mkdtempSync(join(tmpdir(), "autorag-local-empty-"));
		const agent = new AutoRAGAgent({
			model: textOnlyModel(),
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: false,
			jikji: false,
			thinking: false,
		});

		await expect(agent.searchDocuments("is there anything about unicorns?")).rejects.toThrow(
			/completed without emitting structured results/,
		);
	});
});
