import { randomUUID } from "node:crypto";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { type FauxProviderRegistration, fauxAssistantMessage, fauxToolCall } from "@earendil-works/pi-ai";
import { registerFauxProvider } from "@earendil-works/pi-ai/compat";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { AutoRAGAgent } from "../../src/agent/agent.ts";
import { EMIT_AUTORAG_RESULTS_TOOL_NAME } from "../../src/agent/emit-results-tool.ts";

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

function modelFor(answer = "grounded answer") {
	const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "single-agent" }] });
	registration.setResponses([
		fauxAssistantMessage(
			[
				fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
					answer: `[1] ${answer}`,
					results: [
						{
							number: 1,
							title: "Result",
							summary: answer,
							evidence: [{ excerpt: answer }],
							confidence: 0.9,
						},
					],
					mapping: [{ number: 1, source: "/docs/a.txt", method: "bash", content: answer }],
				}),
			],
			{ stopReason: "toolUse" },
		),
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

		expect(response.answer).toBe("[1] grounded answer");
		expect(response.results).toHaveLength(1);
		expect(agent.getResultRegistry(response.sessionId).get(1)?.source).toBe("/docs/a.txt");
	});

	it("passes programmatic provider credentials to the model request", async () => {
		const apiKey = "programmatic-test-api-key";
		const registration = registerFauxProvider({
			api: `faux-${randomUUID()}`,
			provider: `credential-provider-${randomUUID()}`,
			models: [{ id: "credential-model" }],
		});
		registration.setResponses([
			(_context, options) => {
				expect(options?.apiKey).toBe(apiKey);
				return fauxAssistantMessage(
					[
						fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
							answer: "[1] authenticated",
							results: [
								{
									number: 1,
									title: "Authenticated result",
									summary: "authenticated",
									evidence: [{ excerpt: "authenticated" }],
									confidence: 1,
								},
							],
							mapping: [
								{
									number: 1,
									source: "/docs/auth.txt",
									method: "bash",
									content: "authenticated",
								},
							],
						}),
					],
					{ stopReason: "toolUse" },
				);
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
			answer: "[1] authenticated",
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
		await expect(first).resolves.toMatchObject({ answer: "[1] first" });
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
		const registration = registerFauxProvider({ api: `faux-${randomUUID()}`, models: [{ id: "streaming-agent" }] });
		registration.setResponses([
			fauxAssistantMessage([{ type: "text", text: "류동현 선임은 오픈소스 과제 담당자로 보입니다. 추가 자료를 확인하겠습니다." }], {
				stopReason: "stop",
			}),
			fauxAssistantMessage(
				[
					fauxToolCall(EMIT_AUTORAG_RESULTS_TOOL_NAME, {
						answer: "[1] 확인된 답변",
						results: [
							{
								number: 1,
								title: "확인된 결과",
								summary: "확인된 답변",
								evidence: [{ excerpt: "확인된 답변" }],
								confidence: 1,
							},
						],
						mapping: [{ number: 1, source: "/docs/a.txt", method: "bash", content: "확인된 답변" }],
					}),
				],
				{ stopReason: "toolUse" },
			),
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

		expect(events[0]).toMatchObject({ type: "progress" });
		expect(events.at(-1)).toMatchObject({ type: "complete", response: { answer: "[1] 확인된 답변" } });
	});
});
