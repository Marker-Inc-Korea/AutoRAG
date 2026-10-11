import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";

let root: string;
let registrations: FauxProviderRegistration[];

beforeEach(() => {
	root = mkdtempSync(join(tmpdir(), "autorag-search-documents-"));
	registrations = [];
});

afterEach(() => {
	for (const registration of registrations) registration.unregister();
	rmSync(root, { recursive: true, force: true });
});

function groundedSource(answer: string): string {
	const source = join(root, "grounded-answer.txt");
	writeFileSync(source, answer);
	return source;
}

/** Two-phase script: a fast plain answer, then a verified answer citing the local file it grounded on. */
function modelFor(answer = "grounded answer") {
	const source = groundedSource(answer);
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "single-agent" }] });
	registration.setResponses([
		fauxAssistantMessage("Initial pass.", { stopReason: "stop" }),
		fauxAssistantMessage(`${answer} [file:${source}]`, { stopReason: "stop" }),
	]);
	registrations.push(registration);
	return registration.getModel();
}

describe("AutoRAGAgent searchDocuments", () => {
	it("returns structured results without any child-agent dispatch", async () => {
		const agent = new AutoRAGAgent({
			model: modelFor(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});

		const response = await agent.searchDocuments("find the grounded answer");

		expect(response.answer).toBe("grounded answer [1]");
		expect(response.results).toHaveLength(1);
		expect(agent.getResultRegistry(response.sessionId).get(1)?.source).toBe(join(root, "grounded-answer.txt"));
	});

	it("passes programmatic provider credentials to the model request", async () => {
		const apiKey = "programmatic-test-api-key";
		const authSource = join(root, "auth.txt");
		writeFileSync(authSource, "authenticated");
		const registration = registerFauxProvider({
			api: `faux-${randomUUID()}`,
			provider: `credential-provider-${randomUUID()}`,
			models: [{ id: "credential-model" }],
		});
		registration.setResponses([
			fauxAssistantMessage("Initial pass.", { stopReason: "stop" }),
			(_context, options) => {
				expect(options?.apiKey).toBe(apiKey);
				return fauxAssistantMessage(`authenticated [file:${authSource}]`, { stopReason: "stop" });
			},
		]);
		registrations.push(registration);
		const model = registration.getModel();
		const agent = new AutoRAGAgent({
			model,
			apiKey,
			providerApiKeys: { [model.provider]: apiKey },
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});

		await expect(agent.searchDocuments("authenticated search")).resolves.toMatchObject({
			answer: "authenticated [1]",
		});
	});

	it("rejects concurrent searches and recovers after completion", async () => {
		const agent = new AutoRAGAgent({
			model: modelFor("first"),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});

		const first = agent.searchDocuments("first");
		await expect(agent.searchDocuments("second")).rejects.toThrow(/busy/i);
		await expect(first).resolves.toMatchObject({ answer: "first [1]" });
	});

	it("returns an empty structured response for blank queries", async () => {
		const agent = new AutoRAGAgent({
			model: modelFor(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});

		await expect(agent.searchDocuments("  ")).resolves.toMatchObject({ query: "", answer: "", results: [] });
	});

	it("preserves startup diagnostics for blank queries", async () => {
		const agent = new AutoRAGAgent({
			model: modelFor(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
			startupDiagnostics: [
				{
					code: "unknown-datasource-skill",
					severity: "warning",
					message: "Unknown datasource skill(s) in config were skipped: dropbox",
					source: "datasources",
				},
			],
		});

		await expect(agent.searchDocuments("  ")).resolves.toMatchObject({
			diagnostics: [
				expect.objectContaining({
					code: "unknown-datasource-skill",
					severity: "warning",
					source: "datasources",
				}),
			],
		});
	});

	it("yields assistant progress before the structured completion", async () => {
		const source = groundedSource("확인된 답변");
		const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "streaming-agent" }] });
		registration.setResponses([
			// The fast phase ends in a plain answer (never progress).
			fauxAssistantMessage("초기 확인 중입니다.", { stopReason: "stop" }),
			// A message that ends on a tool call is a progress note, not an answer.
			fauxAssistantMessage(
				[
					{ type: "text", text: "류동현 선임은 오픈소스 과제 담당자로 보입니다. 추가 자료를 확인하겠습니다." },
					fauxToolCall("check_memory", { query: "류동현 선임" }),
				],
				{ stopReason: "toolUse" },
			),
			fauxAssistantMessage(`확인된 답변 [file:${source}]`, { stopReason: "stop" }),
		]);
		registrations.push(registration);
		const agent = new AutoRAGAgent({
			model: registration.getModel(),
			searchPaths: ["test/fixtures/sample-project"],
			workspacePath: root,
			memoryPath: join(root, "memory.json"),
			jikji: false,
		});

		const events = [];
		for await (const event of agent.searchDocumentsStream("류동현 선임 전화번호")) events.push(event);

		expect(events[0]).toMatchObject({ type: "progress", text: "Reviewing the query." });
		const progress = events.flatMap((event) => (event.type === "progress" ? [event.text] : []));
		expect(progress).toContain("류동현 선임은 오픈소스 과제 담당자로 보입니다. 추가 자료를 확인하겠습니다.");
		expect(events.at(-1)).toMatchObject({ type: "complete", response: { answer: "확인된 답변 [1]" } });
	});
});
