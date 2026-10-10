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

afterEach(() => {
	for (const registration of registrations.splice(0)) registration.unregister();
	if (tmpDir) rmSync(tmpDir, { recursive: true, force: true });
	tmpDir = undefined;
});

/** Tool names the transcript declares to the model (system messages carry `toolsAdded`). */
function declaredToolNames(messages: readonly unknown[]): string[] {
	const names = new Set<string>();
	for (const message of messages) {
		if (typeof message !== "object" || message === null || !("role" in message) || message.role !== "system")
			continue;
		const added = "toolsAdded" in message && Array.isArray(message.toolsAdded) ? message.toolsAdded : [];
		for (const tool of added) {
			if (typeof tool === "object" && tool !== null && "name" in tool && typeof tool.name === "string") {
				names.add(tool.name);
			}
		}
	}
	return [...names];
}

describe("AutoRAGAgent model requests", () => {
	it("declares the agent's tools to the model so it can call them", async () => {
		tmpDir = mkdtempSync(join(tmpdir(), "autorag-tool-declarations-"));
		const seen: string[][] = [];
		const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "tool-check" }] });
		const respond = (context: { messages: readonly unknown[] }) => {
			seen.push(declaredToolNames(context.messages));
			return fauxAssistantMessage([{ type: "text", text: "done" }], { stopReason: "stop" });
		};
		registration.setResponses([respond, respond, respond]);
		registrations.push(registration);

		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: [FIXTURE_DIR],
			workspacePath: tmpDir,
			memoryPath: join(tmpDir, "memory.json"),
			minSync: { autoInstall: false },
			jikji: false,
			webSearch: false,
		});
		await agent.searchDocuments("where is the report?");

		expect(seen.length).toBeGreaterThan(0);
		expect(seen[0]).toEqual(expect.arrayContaining(["bash", "search_all_documents", "emit_autorag_results"]));
	});
});
