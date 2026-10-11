import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

let root: string;
let registration: FauxProviderRegistration;

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-lifecycle-"));
	const source = join(root, "a.txt");
	writeFileSync(source, "a");
	registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "single-agent" }] });
	registration.setResponses([fauxAssistantMessage(`answer [file:${source}]`, { stopReason: "stop" })]);
});

afterEach(() => {
	registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

describe("AutoRAGAgent lifecycle", () => {
	it("forwards events from the direct in-flight agent", async () => {
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});
		const events: string[] = [];
		const unsubscribe = agent.subscribe((event) => {
			events.push(event.type);
		});

		await agent.searchDocuments("Meeting");
		unsubscribe();

		expect(events.length).toBeGreaterThan(0);
	});
});
